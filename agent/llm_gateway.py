"""The single entry point for LLM calls.

Nodes ask for a route by name, and this module owns everything about providers. A call follows
these steps:

1. Skip any provider with no key or an open circuit breaker.
2. Try the remaining models in route order, each with its own timeout within the route's
   total deadline.
3. Count a failure against the provider's breaker. A refusal is not counted, because the
   provider is healthy and it was only this request that was declined.
4. If text had already been streamed, reset the draft before the next model starts.
5. If every model fails, raise `AllModelsFailedError`, and the calling node uses its own
   fallback: a default plan in the supervisor, and cited excerpts in the response agent.

Fallback is a plain loop rather than `with_fallbacks`, so that each attempt can be reported.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from config.settings import CircuitBreakerLimits, EnvironmentSettings, LlmLimits
from resilience.circuit_breaker import CircuitBreaker

logger = structlog.get_logger()

# The `FAULTS` value that fails every provider. A provider's own name fails only that one.
ALL_PROVIDERS_FAULT = "llm"

TextHandler = Callable[[str], Awaitable[None]]
DraftResetHandler = Callable[[], Awaitable[None]]


class AllModelsFailedError(Exception):
    """Every model on a route failed or was skipped, so the caller must use its own fallback."""

    def __init__(self, route_name: str, failures: list[str]) -> None:
        """Record which models failed, for the logs and the degraded list."""
        super().__init__(f"All models failed on route {route_name}: {failures}")
        self.route_name = route_name
        self.failures = failures


class ModelRefusedError(Exception):
    """The model declined the request, which the next model on the route may still answer."""


class SimulatedLlmFaultError(RuntimeError):
    """Raised when `FAULTS` asks for a provider to fail during a demo or a test."""


@dataclass(frozen=True)
class GatewayResult[ResultValue]:
    """A successful call: the value, the model that produced it, and any models that failed."""

    value: ResultValue
    model_spec: str
    failed_model_specs: list[str] = field(default_factory=list)

    def describe_fallbacks(self, reason_prefix: str) -> list[str]:
        """Return one `degraded` entry per failed model, naming the step that was affected."""
        return [f"{reason_prefix}:{model_spec}" for model_spec in self.failed_model_specs]


class ChatModelFactory(Protocol):
    """Creates a chat model, so that tests can supply a fake."""

    def __call__(
        self,
        model_spec: str,
        api_key: str,
        max_tokens: int,
        max_retries: int,
        **provider_options: Any,
    ) -> Any:
        """Return a LangChain chat model."""
        ...


def create_chat_model(
    model_spec: str,
    api_key: str,
    max_tokens: int,
    max_retries: int,
    **provider_options: Any,
) -> Any:
    """Create a LangChain chat model. This is the only place that does so."""
    return init_chat_model(
        model_spec,
        api_key=api_key,
        max_tokens=max_tokens,
        max_retries=max_retries,
        **provider_options,
    )


def _provider_of(model_spec: str) -> str:
    """Return the provider part of a spec such as "anthropic:claude-sonnet-5-5"."""
    return model_spec.split(":", maxsplit=1)[0]


class LlmGateway:
    """Runs LLM calls along a route of models. Created once and shared by all requests."""

    def __init__(
        self,
        routes: dict[str, list[str]],
        settings: EnvironmentSettings,
        llm_limits: LlmLimits,
        breaker_limits: CircuitBreakerLimits,
        answer_effort: str,
        chat_model_factory: ChatModelFactory = create_chat_model,
    ) -> None:
        """Keep the routes and keys. A provider with no key is skipped, not treated as an error."""
        self._routes = routes
        self._limits = llm_limits
        self._answer_effort = answer_effort
        self._enabled_faults = settings.enabled_faults
        self._chat_model_factory = chat_model_factory
        self._api_keys = {
            "anthropic": settings.anthropic_api_key.get_secret_value(),
            "openai": settings.openai_api_key.get_secret_value(),
            "groq": settings.groq_api_key.get_secret_value(),
        }
        self._breakers = {
            provider: CircuitBreaker(
                breaker_limits.failure_threshold, breaker_limits.recovery_seconds
            )
            for provider in self._api_keys
        }

    def _usable_model_specs(self, route_name: str) -> list[str]:
        """Return the route's models, leaving out providers with no key or an open breaker."""
        usable_model_specs = []
        for model_spec in self._routes[route_name]:
            provider = _provider_of(model_spec)
            if self._api_keys.get(provider) and self._breakers[provider].allows_request():
                usable_model_specs.append(model_spec)
            else:
                logger.warning("llm_model_skipped", route=route_name, model=model_spec)
        return usable_model_specs

    def _raise_if_fault_enabled(self, provider: str) -> None:
        """Fail on purpose when `FAULTS` names this provider or all of them."""
        if ALL_PROVIDERS_FAULT in self._enabled_faults or provider in self._enabled_faults:
            raise SimulatedLlmFaultError(f"Simulated {provider} failure from FAULTS")

    def _create_model(self, model_spec: str, max_tokens: int, **provider_options: Any) -> Any:
        """Create a client for one attempt, with its provider's key."""
        return self._chat_model_factory(
            model_spec,
            self._api_keys[_provider_of(model_spec)],
            max_tokens,
            self._limits.max_retries_per_model,
            **provider_options,
        )

    def _answer_options(self, model_spec: str) -> dict[str, Any]:
        """Return the options for a streamed answer. Only Anthropic models take an effort."""
        if _provider_of(model_spec) == "anthropic":
            return {"output_config": {"effort": self._answer_effort}}
        return {}

    def _attempt_timeout(self, deadline: float) -> float:
        """Return the seconds for one attempt, which may not pass the route's deadline."""
        seconds_left_on_route = deadline - asyncio.get_running_loop().time()
        return min(self._limits.attempt_timeout_s, seconds_left_on_route)

    def _record_failure(
        self, route_name: str, model_spec: str, error: Exception, failures: list[str]
    ) -> None:
        """Count a failure against the provider, and note it for the caller."""
        if not isinstance(error, ModelRefusedError):
            self._breakers[_provider_of(model_spec)].record_failure()
        failures.append(model_spec)
        logger.warning(
            "llm_attempt_failed", route=route_name, model=model_spec, error=type(error).__name__
        )

    async def generate_structured[OutputSchema: BaseModel](
        self,
        route_name: str,
        messages: Sequence[BaseMessage],
        output_schema: type[OutputSchema],
    ) -> GatewayResult[OutputSchema]:
        """Return a validated pydantic object from the first model that can produce one."""
        deadline = asyncio.get_running_loop().time() + self._limits.route_deadline_s
        failures: list[str] = []

        for model_spec in self._usable_model_specs(route_name):
            try:
                self._raise_if_fault_enabled(_provider_of(model_spec))
                chat_model = self._create_model(model_spec, self._limits.structured_max_tokens)
                # Sonnet 5.5 rejects forced tool calls, which is the default method here.
                structured_model = chat_model.with_structured_output(
                    output_schema, method="json_schema"
                )
                async with asyncio.timeout(self._attempt_timeout(deadline)):
                    structured_output = await structured_model.ainvoke(list(messages))
            except Exception as error:  # Any failure means trying the next model.
                self._record_failure(route_name, model_spec, error, failures)
                continue

            self._breakers[_provider_of(model_spec)].record_success()
            return GatewayResult(structured_output, model_spec, failures)

        raise AllModelsFailedError(route_name, failures)

    async def _stream_one_model(
        self,
        chat_model: Any,
        messages: Sequence[BaseMessage],
        on_text: TextHandler,
    ) -> str:
        """Stream one model's reply, and return the whole text."""
        collected_text: list[str] = []
        async for message_chunk in chat_model.astream(list(messages)):
            # A refusal can arrive after some text, so it is checked on every chunk.
            if message_chunk.response_metadata.get("stop_reason") == "refusal":
                raise ModelRefusedError("The model declined this request.")
            chunk_text = message_chunk.text
            if chunk_text:
                collected_text.append(chunk_text)
                await on_text(chunk_text)

        full_text = "".join(collected_text)
        if not full_text:
            raise ModelRefusedError("The model returned no text.")
        return full_text

    async def stream_text(
        self,
        route_name: str,
        messages: Sequence[BaseMessage],
        on_text: TextHandler,
        on_draft_reset: DraftResetHandler,
    ) -> GatewayResult[str]:
        """Stream text from the first model that can finish a reply.

        If a model fails after streaming some text, `on_draft_reset` runs before the next model
        starts, so that a draft from a failed model never joins a later one.
        """
        deadline = asyncio.get_running_loop().time() + self._limits.route_deadline_s
        failures: list[str] = []

        for model_spec in self._usable_model_specs(route_name):
            text_was_streamed = False

            async def forward_text(text: str) -> None:
                nonlocal text_was_streamed
                text_was_streamed = True
                await on_text(text)

            try:
                self._raise_if_fault_enabled(_provider_of(model_spec))
                chat_model = self._create_model(
                    model_spec, self._limits.answer_max_tokens, **self._answer_options(model_spec)
                )
                async with asyncio.timeout(self._attempt_timeout(deadline)):
                    full_text = await self._stream_one_model(chat_model, messages, forward_text)
            except Exception as error:  # Any failure means trying the next model.
                self._record_failure(route_name, model_spec, error, failures)
                if text_was_streamed:
                    await on_draft_reset()
                continue

            self._breakers[_provider_of(model_spec)].record_success()
            return GatewayResult(full_text, model_spec, failures)

        raise AllModelsFailedError(route_name, failures)

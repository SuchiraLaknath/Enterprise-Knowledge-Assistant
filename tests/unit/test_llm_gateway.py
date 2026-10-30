import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.llm_gateway import AllModelsFailedError, LlmGateway
from config.settings import CircuitBreakerLimits, EnvironmentSettings, LlmLimits
from tests.fakes.fake_llm import FakeChatModel

ROUTES = {"response": ["anthropic:first", "anthropic:second", "openai:third"]}
MESSAGES = [HumanMessage("Hello")]
LLM_LIMITS = LlmLimits(
    attempt_timeout_s=1.0,
    route_deadline_s=2.0,
    max_retries_per_model=0,
    structured_max_tokens=100,
    answer_max_tokens=100,
)
BREAKER_LIMITS = CircuitBreakerLimits(failure_threshold=2, recovery_seconds=30.0)


created_model_options: dict = {}


class Reply(BaseModel):
    text: str


def make_gateway(models, faults="", openai_key="openai-key", breaker_limits=BREAKER_LIMITS):
    settings = EnvironmentSettings(
        _env_file=None,
        pinecone_api_key="test",
        pinecone_index="test",
        anthropic_api_key="anthropic-key",
        openai_api_key=openai_key,
        groq_api_key="",
        faults=faults,
    )

    def create_fake_model(model_spec, api_key, max_tokens, max_retries, **provider_options):
        created_model_options[model_spec] = provider_options
        return models[model_spec]

    return LlmGateway(ROUTES, settings, LLM_LIMITS, breaker_limits, "low", create_fake_model)


class TokenCollector:
    def __init__(self):
        self.tokens = []
        self.reset_count = 0

    async def on_text(self, text):
        self.tokens.append(text)

    async def on_draft_reset(self):
        self.reset_count += 1


def healthy_models(**overrides):
    models = {
        "anthropic:first": FakeChatModel("first reply"),
        "anthropic:second": FakeChatModel("second reply"),
        "openai:third": FakeChatModel("third reply"),
    }
    return models | {name.replace("_", ":"): model for name, model in overrides.items()}


async def test_structured_call_uses_the_first_model_and_json_schema():
    models = healthy_models()

    result = await make_gateway(models).generate_structured("response", MESSAGES, Reply)

    assert result.value == "first reply"
    assert result.model_spec == "anthropic:first"
    assert result.failed_model_specs == []
    # Sonnet 5.5 rejects the default forced-tool method with a 400.
    assert models["anthropic:first"].structured_method == "json_schema"


async def test_route_order_is_followed_when_a_model_fails():
    models = healthy_models(anthropic_first=FakeChatModel(fails=True))

    result = await make_gateway(models).generate_structured("response", MESSAGES, Reply)

    assert result.value == "second reply"
    assert result.failed_model_specs == ["anthropic:first"]


async def test_a_provider_with_no_key_is_skipped():
    models = healthy_models(
        anthropic_first=FakeChatModel(fails=True), anthropic_second=FakeChatModel(fails=True)
    )
    gateway = make_gateway(models, openai_key="")

    with pytest.raises(AllModelsFailedError):
        await gateway.generate_structured("response", MESSAGES, Reply)

    assert models["openai:third"].call_count == 0


async def test_an_open_breaker_skips_the_provider_on_later_calls():
    # Two failures open the Anthropic breaker, so the next call goes straight to OpenAI.
    models = healthy_models(
        anthropic_first=FakeChatModel(fails=True), anthropic_second=FakeChatModel(fails=True)
    )
    gateway = make_gateway(models)

    await gateway.generate_structured("response", MESSAGES, Reply)
    calls_before = models["anthropic:first"].call_count
    result = await gateway.generate_structured("response", MESSAGES, Reply)

    assert models["anthropic:first"].call_count == calls_before
    assert result.model_spec == "openai:third"


async def test_every_model_failing_raises_with_the_failures_listed():
    models = {name: FakeChatModel(fails=True) for name in ROUTES["response"]}

    with pytest.raises(AllModelsFailedError) as caught_error:
        await make_gateway(models).generate_structured("response", MESSAGES, Reply)

    assert caught_error.value.failures == ROUTES["response"]


async def test_a_slow_model_times_out_and_the_next_model_answers():
    class SlowModel(FakeChatModel):
        async def ainvoke(self, messages):
            import asyncio

            await asyncio.sleep(5)

    models = healthy_models(anthropic_first=SlowModel())

    result = await make_gateway(models).generate_structured("response", MESSAGES, Reply)

    assert result.model_spec == "anthropic:second"


async def test_streaming_forwards_tokens_and_returns_the_full_text():
    collector = TokenCollector()

    result = await make_gateway(healthy_models()).stream_text(
        "response", MESSAGES, collector.on_text, collector.on_draft_reset
    )

    assert result.value == "first reply"
    assert collector.tokens == ["first reply"]
    assert collector.reset_count == 0


async def test_failure_after_text_clears_the_draft_before_the_next_model():
    # A draft from a failed model must never join the next model's answer.
    models = healthy_models(anthropic_first=FakeChatModel("partial", fails_after_text=True))
    collector = TokenCollector()

    result = await make_gateway(models).stream_text(
        "response", MESSAGES, collector.on_text, collector.on_draft_reset
    )

    assert collector.reset_count == 1
    assert result.value == "second reply"


async def test_failure_before_any_text_needs_no_draft_reset():
    models = healthy_models(anthropic_first=FakeChatModel(fails=True))
    collector = TokenCollector()

    await make_gateway(models).stream_text(
        "response", MESSAGES, collector.on_text, collector.on_draft_reset
    )

    assert collector.reset_count == 0


async def test_a_refusal_moves_on_to_the_next_model():
    models = healthy_models(anthropic_first=FakeChatModel("partial", refuses=True))
    collector = TokenCollector()

    result = await make_gateway(models).stream_text(
        "response", MESSAGES, collector.on_text, collector.on_draft_reset
    )

    assert result.model_spec == "anthropic:second"


async def test_streamed_answers_ask_anthropic_for_the_configured_effort():
    collector = TokenCollector()

    await make_gateway(healthy_models()).stream_text(
        "response", MESSAGES, collector.on_text, collector.on_draft_reset
    )

    assert created_model_options["anthropic:first"] == {"output_config": {"effort": "low"}}


@pytest.mark.faults
async def test_provider_fault_fails_only_that_provider():
    models = healthy_models()

    result = await make_gateway(models, faults="anthropic").generate_structured(
        "response", MESSAGES, Reply
    )

    assert result.model_spec == "openai:third"
    assert models["anthropic:first"].call_count == 0


@pytest.mark.faults
async def test_llm_fault_fails_every_provider():
    with pytest.raises(AllModelsFailedError):
        await make_gateway(healthy_models(), faults="llm").generate_structured(
            "response", MESSAGES, Reply
        )

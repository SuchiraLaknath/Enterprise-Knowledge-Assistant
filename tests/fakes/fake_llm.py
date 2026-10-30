"""In-memory stand-ins for chat models and the gateway, so that tests make no LLM calls."""

from types import SimpleNamespace

from agent.llm_gateway import AllModelsFailedError, GatewayResult
from agent.schemas import Plan


def make_plan(
    needs_documents: bool = True, search_query: str = "payment outage", **overrides
) -> Plan:
    values = {
        "intent": "Find out about payment outages.",
        "steps": ["Search the documents", "Write the answer"],
        "needs_documents": needs_documents,
        "search_query": search_query,
        "departments": [],
        "document_types": [],
        "recent_days": None,
    }
    return Plan(**(values | overrides))


class FakeLlm:
    """Replays preset plans and answers, and records what it was asked."""

    def __init__(self, plans=None, answers=None, fails=False, stream_tokens=None) -> None:
        self.plans = list(plans or [make_plan()])
        self.answers = list(answers or ["Payments failed because a certificate expired [1]."])
        self.fails = fails
        self.stream_tokens = stream_tokens
        self.received_answer_messages: list[list] = []

    async def generate_structured(self, route_name, messages, output_schema):
        if self.fails:
            raise AllModelsFailedError(route_name, ["fake:model"])
        return GatewayResult(self.plans.pop(0), "fake:model")

    async def stream_text(self, route_name, messages, on_text, on_draft_reset):
        self.received_answer_messages.append(list(messages))
        if self.fails:
            raise AllModelsFailedError(route_name, ["fake:model"])
        answer = self.answers.pop(0)
        for token in self.stream_tokens or [answer]:
            await on_text(token)
        return GatewayResult(answer, "fake:model")


class FakeChatModel:
    """Stands in for a LangChain chat model, for testing the gateway's fallback loop."""

    def __init__(self, reply="A reply", fails=False, fails_after_text=False, refuses=False):
        self.reply = reply
        self.fails = fails
        self.fails_after_text = fails_after_text
        self.refuses = refuses
        self.call_count = 0

    def with_structured_output(self, output_schema, method):
        self.structured_method = method
        return self

    async def ainvoke(self, messages):
        self.call_count += 1
        if self.fails:
            raise ConnectionError("Simulated model failure")
        return self.reply

    async def astream(self, messages):
        self.call_count += 1
        if self.fails:
            raise ConnectionError("Simulated model failure")
        yield SimpleNamespace(text=self.reply, response_metadata={})
        if self.fails_after_text:
            raise ConnectionError("Simulated failure after streaming text")
        if self.refuses:
            yield SimpleNamespace(text="", response_metadata={"stop_reason": "refusal"})

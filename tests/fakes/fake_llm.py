"""In-memory stand-ins for chat models and the gateway, so that tests make no LLM calls."""

from types import SimpleNamespace

from agent.llm_gateway import AllModelsFailedError, GatewayResult
from agent.schemas import AnswerJudgement, InputJudgement, Plan


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
        "needs_research": False,
        "needs_tools": False,
    }
    return Plan(**(values | overrides))


# The guards run on every turn, so tests that are not about them get a harmless reply.
DEFAULT_GUARD_REPLIES = {
    InputJudgement: InputJudgement(is_attack=False, category=""),
    AnswerJudgement: AnswerJudgement(supported_by_evidence=True, on_brand=True, problem=""),
}


class FakeLlm:
    """Replays preset plans and answers, and records what it was asked."""

    def __init__(
        self,
        plans=None,
        answers=None,
        fails=False,
        stream_tokens=None,
        replies=None,
        tool_call_rounds=None,
    ) -> None:
        self.plans = list(plans or [make_plan()])
        self.answers = list(answers or ["Payments failed because a certificate expired [1]."])
        self.fails = fails
        self.stream_tokens = stream_tokens
        self.received_answer_messages: list[list] = []
        # Replies by output schema, for research. An exception in the list is raised instead.
        # A function instead of a list replies from the messages, for batches sent at once.
        self.replies = replies or {}
        self.received_structured_calls: list[tuple] = []
        # Each round is a list of `ToolCallRequest`, or an exception to raise.
        self.tool_call_rounds = list(tool_call_rounds or [])
        self.received_tool_schemas: list[list[dict]] = []

    async def generate_structured(self, route_name, messages, output_schema):
        self.received_structured_calls.append((output_schema, list(messages)))
        if self.fails:
            raise AllModelsFailedError(route_name, ["fake:model"])
        if output_schema in DEFAULT_GUARD_REPLIES and output_schema not in self.replies:
            return GatewayResult(DEFAULT_GUARD_REPLIES[output_schema], "fake:model")
        if output_schema in self.replies:
            reply_source = self.replies[output_schema]
            reply = reply_source(messages) if callable(reply_source) else reply_source.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return GatewayResult(reply, "fake:model")
        return GatewayResult(self.plans.pop(0), "fake:model")

    async def generate_tool_calls(self, route_name, messages, tool_schemas):
        self.received_tool_schemas.append(list(tool_schemas))
        if self.fails:
            raise AllModelsFailedError(route_name, ["fake:model"])
        round_reply = self.tool_call_rounds.pop(0) if self.tool_call_rounds else []
        if isinstance(round_reply, Exception):
            raise round_reply
        return GatewayResult(round_reply, "fake:model")

    async def stream_text(self, route_name, messages, on_text, on_draft_reset):
        self.received_answer_messages.append(list(messages))
        if self.fails:
            raise AllModelsFailedError(route_name, ["fake:model"])
        # The last answer repeats, so that a test can reach the validator's retry limit.
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
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

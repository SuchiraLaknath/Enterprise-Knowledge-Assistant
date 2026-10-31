"""The state shared by all graph nodes, and which of it lasts beyond one turn.

The checkpointer keeps every field between turns, so `load_context` resets the per-turn fields
with `Overwrite`. Without that, the additive reducers below would carry old evidence into a new
turn, and an answer could cite chunks that were not retrieved for it (rule 4).
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.schemas import Citation, Finding, Plan, ResearchReport, ValidationReport
from retrieval.models import RetrievedChunk, UserContext
from tools.models import ToolResult


class AgentState(TypedDict, total=False):
    """Fields are grouped by how long they last."""

    # Set by the API from the verified JWT. Rule 1: no node may write this field.
    user: UserContext
    # Set by the API, for a fallback used before the graph started.
    request_degraded: list[str]

    # The conversation. Only finished answers are added, never a streaming draft (rule 9).
    messages: Annotated[list[BaseMessage], add_messages]

    # Per-turn fields, reset by `load_context`.
    plan: Plan | None
    retrieval_finished: bool
    research_allowed: bool
    research_finished: bool
    research_report: ResearchReport | None
    tools_finished: bool
    tool_results: Annotated[list[ToolResult], operator.add]
    # Set when the input guard's judge is down, so that only read tools may run.
    read_only_tools: bool
    input_blocked: bool
    validation_report: ValidationReport | None
    # What a retry must fix. Empty on the first attempt.
    validation_feedback: str
    # Stored in the conversation instead of the answer, for excerpts that are raw document text.
    conversation_note: str | None
    # A fixed sentence added to the answer, such as why deep research was not used.
    turn_notice: str | None
    evidence: Annotated[list[RetrievedChunk], operator.add]
    degraded: Annotated[list[str], operator.add]
    # The validator's two loops, which are bounded by these counters.
    response_retries: int
    evidence_retries: int
    # The answer shown this turn. It can differ from the message stored in the conversation.
    final_answer: str
    final_citations: list[Citation]


class ResearchInput(TypedDict, total=False):
    """What the research subgraph takes from the main graph."""

    user: UserContext
    plan: Plan | None
    messages: Annotated[list[BaseMessage], add_messages]


class ResearchOutput(TypedDict, total=False):
    """What the research subgraph gives back. Only new evidence is returned, never old."""

    evidence: Annotated[list[RetrievedChunk], operator.add]
    degraded: Annotated[list[str], operator.add]
    research_report: ResearchReport | None
    research_finished: bool


class ResearchState(ResearchInput, ResearchOutput, total=False):
    """Everything the subgraph tracks during one research run."""

    access_confirmed: bool
    # The plan ladder: attempts so far, the code tried, how it failed, and if the template ran.
    plan_attempt: int
    plan_code: str
    plan_error: str
    template_used: bool
    findings: list[Finding]
    # These carry across attempts, because the limits are per run (rule 7).
    sub_agent_calls_used: int
    run_deadline: float
    chunks_read: int
    depth_reached: int

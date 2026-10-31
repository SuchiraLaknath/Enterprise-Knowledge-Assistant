"""The state shared by all graph nodes, and which of it lasts beyond one turn.

The checkpointer keeps every field between turns, so `load_context` resets the per-turn fields
with `Overwrite`. Without that, the additive reducers below would carry old evidence into a new
turn, and an answer could cite chunks that were not retrieved for it (rule 4).
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.schemas import Citation, Plan, ResearchReport
from retrieval.models import RetrievedChunk, UserContext


class AgentState(TypedDict, total=False):
    """Fields are grouped by how long they last."""

    # Set by the API from the verified JWT. Rule 1: no node may write this field.
    user: UserContext

    # The conversation. Only finished answers are added, never a streaming draft (rule 9).
    messages: Annotated[list[BaseMessage], add_messages]

    # Per-turn fields, reset by `load_context`.
    plan: Plan | None
    retrieval_finished: bool
    research_finished: bool
    research_report: ResearchReport | None
    # A fixed sentence added to the answer, such as why deep research was not used.
    turn_notice: str | None
    evidence: Annotated[list[RetrievedChunk], operator.add]
    degraded: Annotated[list[str], operator.add]
    # Reserved for the validator loop in build step 8.
    response_retries: int
    evidence_retries: int
    # The answer shown this turn. It can differ from the message stored in the conversation.
    final_answer: str
    final_citations: list[Citation]

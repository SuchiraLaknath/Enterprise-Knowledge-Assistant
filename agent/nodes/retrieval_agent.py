"""`retrieval_agent`: find the evidence for the supervisor's plan.

The plan can narrow a search, but access always comes from the verified user. This node passes
`state["user"]` through untouched, and `DocumentSearchService` builds the access filter from it.
"""

from typing import Any

from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.schemas import Plan
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from retrieval.filters import timestamp_days_ago
from retrieval.models import DocumentFilters

NODE_NAME = "retrieval_agent"


def build_document_filters(plan: Plan) -> DocumentFilters:
    """Turn the plan's narrowing into search filters. These can only reduce what is searched."""
    created_after_ts = None
    if plan.recent_days is not None:
        created_after_ts = timestamp_days_ago(plan.recent_days)
    return DocumentFilters(
        document_types=tuple(plan.document_types), created_after_ts=created_after_ts
    )


async def retrieval_agent(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Search for the planned query and return attributed evidence.

    There is no error handling here, because the search service returns its failures as data
    in `degraded`, and falls back to local search by itself.
    """
    emit_node_started(NODE_NAME)
    plan = state["plan"]
    # The supervisor routes here only after making a plan, so one always exists.
    assert plan is not None

    search_result = await runtime.context.document_search.search(
        plan.search_query,
        state["user"],
        build_document_filters(plan),
        # An empty list means that the plan did not narrow the search to any department.
        namespaces=tuple(plan.departments) or None,
    )

    emit_event(
        EventType.RETRIEVAL,
        query=plan.search_query,
        departments=plan.departments or "all",
        result_count=len(search_result.chunks),
        dense_weight=search_result.dense_weight,
        sources=[
            {"chunk_id": chunk.chunk_id, "title": chunk.title, "score": round(chunk.score, 3)}
            for chunk in search_result.chunks
        ],
        degraded=search_result.degraded,
    )
    # `retrieval` lists the evidence, and `state` updates the running totals in the panel.
    emit_event(
        EventType.STATE,
        evidence_count=len(search_result.chunks),
        degraded=search_result.degraded,
    )
    emit_node_finished(NODE_NAME)
    return {
        "evidence": search_result.chunks,
        "degraded": search_result.degraded,
        "retrieval_finished": True,
    }

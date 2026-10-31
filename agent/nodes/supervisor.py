"""`supervisor`: understand the request, plan the work and route it to a worker.

The first visit creates the plan with an LLM. Workers then return here, and later visits only
choose the next step, which needs no LLM call.
"""

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError
from agent.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from agent.schemas import Plan
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started

NODE_NAME = "supervisor"
# Planning needs the same strong models as answering, so it shares their route.
PLANNING_ROUTE = "response"

DEGRADED_MODEL_FALLBACK = "supervisor_used_fallback_model"
DEGRADED_NO_PLAN = "supervisor_plan_unavailable"


def latest_user_text(state: AgentState) -> str:
    """Return the employee's newest message."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _fallback_plan(user_text: str) -> Plan:
    """Search with the employee's own words when no model can plan.

    A search still finds evidence, and the extractive answer can use it without an LLM.
    """
    return Plan(
        intent="Answer the employee's question.",
        steps=["Search the documents", "Show what was found"],
        needs_documents=True,
        search_query=user_text,
        departments=[],
        document_types=[],
        recent_days=None,
        # With no model to plan, there is none to write research code or sub-agent replies.
        needs_research=False,
    )


async def _create_plan(
    state: AgentState, dependencies: AgentDependencies
) -> tuple[Plan, list[str]]:
    """Ask the LLM for a plan, and return it with any degradation that occurred."""
    conversation: list[BaseMessage] = state["messages"][-dependencies.history_message_count :]
    planning_messages = [SystemMessage(SUPERVISOR_SYSTEM_PROMPT), *conversation]
    try:
        planning_result = await dependencies.llm.generate_structured(
            PLANNING_ROUTE, planning_messages, Plan
        )
    except AllModelsFailedError:
        return _fallback_plan(latest_user_text(state)), [DEGRADED_NO_PLAN]

    degraded = [
        f"{DEGRADED_MODEL_FALLBACK}:{model_spec}"
        for model_spec in planning_result.failed_model_specs
    ]
    return planning_result.value, degraded


async def supervisor(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Create the plan on the first visit. Routing happens in `choose_next_node`."""
    emit_node_started(NODE_NAME)

    # A worker has returned, so the plan already exists and there is nothing to generate.
    if state.get("plan") is not None:
        emit_node_finished(NODE_NAME)
        return {}

    plan, degraded = await _create_plan(state, runtime.context)
    emit_event(
        EventType.STATE,
        intent=plan.intent,
        plan_steps=plan.steps,
        degraded=degraded,
    )
    emit_node_finished(NODE_NAME)
    return {"plan": plan, "degraded": degraded}


def choose_next_node(state: AgentState) -> Literal["retrieval_agent", "response_agent"]:
    """Decide where the graph goes next, and report the decision to the activity panel.

    This is plain code rather than an LLM decision, so that every loop stays bounded.
    """
    plan = state["plan"]
    needs_retrieval = (
        plan is not None and plan.needs_documents and not state.get("retrieval_finished", False)
    )
    next_node: Literal["retrieval_agent", "response_agent"] = (
        "retrieval_agent" if needs_retrieval else "response_agent"
    )
    reason = "documents are needed" if needs_retrieval else "ready to answer"
    emit_event(EventType.ROUTE, source=NODE_NAME, target=next_node, reason=reason)
    return next_node

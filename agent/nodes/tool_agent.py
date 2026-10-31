"""`tool_agent`: let a model choose tools, and run what it chose through the registry.

Rule 2 checks RBAC three times. This node filters what the model is offered, the registry checks
the role again before running anything, and each tool filters its own data. The model's reply is
only a request: its arguments are validated, and a name it was never offered is refused.

The loop is bounded by rounds and by total calls, and a failed tool is recorded rather than
raised, so the response agent can still answer from what remains.

Rule 3: a write tool never runs in the loop. The first one a model asks for is stored, the graph
pauses at `human_approval`, and this node runs it on the way back, with the stored arguments.
"""

import asyncio
import copy
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import format_evidence_documents, format_tool_results
from agent.llm_gateway import AllModelsFailedError, ToolCallRequest
from agent.nodes.supervisor import latest_user_text
from agent.prompts.tool_agent import TOOL_AGENT_RESULTS_TEMPLATE, TOOL_AGENT_SYSTEM_PROMPT
from agent.schemas import PendingToolCall
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from retrieval.models import RetrievedChunk
from tools.knowledge_search import TOOL_NAME as KNOWLEDGE_SEARCH_TOOL
from tools.models import ToolContext, ToolResult
from tools.registry import ToolDefinition

NODE_NAME = "tool_agent"
# Choosing a tool is a small decision, so it uses the cheaper route.
TOOL_ROUTE = "fast"

DEGRADED_NO_TOOL_MODEL = "tool_selection_unavailable"
DEGRADED_MODEL_FALLBACK = "tool_agent_used_fallback_model"
DEGRADED_TOOL_FAILED = "tool_failed"
NOT_APPROVED_MESSAGE = "This action was not approved, so nothing was changed."
FAILED_STATUSES = ("error", "timeout")
# Evidence fields that `python_analysis` may read. This list leaves out the text (rule 7).
ANALYSIS_RECORD_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "section",
    "department",
    "document_type",
    "created_date",
)


def build_analysis_records(
    evidence: list[RetrievedChunk], tool_results: list[ToolResult]
) -> list[dict[str, Any]]:
    """Collect what `python_analysis` may work on: metadata and earlier tool data, never text."""
    records = [
        {field: getattr(chunk, field) for field in ANALYSIS_RECORD_FIELDS} for chunk in evidence
    ]
    for tool_result in tool_results:
        if isinstance(tool_result.data, list):
            # Copies, because generated code could otherwise change a result before it is read.
            records.extend(
                copy.deepcopy(item) for item in tool_result.data if isinstance(item, dict)
            )
    return records


def _build_tool_context(
    state: AgentState, dependencies: AgentDependencies, tool_results: list[ToolResult]
) -> ToolContext:
    """Build what tools may use. Rule 1: the user comes from the state, never from the model."""
    evidence = [*state.get("evidence", [])]
    for tool_result in tool_results:
        evidence.extend(tool_result.evidence)
    return ToolContext(
        user=state["user"],
        document_search=dependencies.document_search,
        audit_log=dependencies.audit_log,
        mcp_client=dependencies.mcp_client,
        sandbox_limits=dependencies.sandbox_limits,
        tool_limits=dependencies.tool_limits,
        analysis_records=build_analysis_records(evidence, tool_results),
    )


def _build_messages(
    state: AgentState, tool_results: list[ToolResult], evidence: list[RetrievedChunk]
) -> list[BaseMessage]:
    """Assemble the prompt: the request, plus anything found so far, marked as data (rule 5)."""
    messages: list[BaseMessage] = [
        SystemMessage(TOOL_AGENT_SYSTEM_PROMPT),
        HumanMessage(latest_user_text(state)),
    ]
    if evidence:
        messages.append(HumanMessage(format_evidence_documents(evidence)))
    if tool_results:
        results_text = TOOL_AGENT_RESULTS_TEMPLATE.format(
            tool_results=format_tool_results(tool_results)
        )
        messages.append(HumanMessage(results_text))
    return messages


async def _run_tool_call(
    tool_call: ToolCallRequest, dependencies: AgentDependencies, context: ToolContext
) -> ToolResult:
    """Run one requested tool through the registry, and report it to the activity panel."""
    tool_name = dependencies.tool_registry.resolve_model_name(tool_call.tool_name)
    tool_result = await dependencies.tool_registry.execute(tool_name, tool_call.arguments, context)
    emit_event(
        EventType.TOOL,
        tool_name=tool_name,
        arguments=tool_call.arguments,
        status=tool_result.status,
        message=tool_result.message,
    )
    return tool_result


def _select_offered_tools(
    state: AgentState, dependencies: AgentDependencies
) -> list[ToolDefinition]:
    """Return what the model may be offered. Rule 2's first check: only what this role may use."""
    offered_tools = dependencies.tool_registry.tools_for_model(state["user"].role)

    # Documents were already searched, and offering search again only invites repeat calls.
    plan = state["plan"]
    if plan is not None and plan.needs_documents:
        offered_tools = [tool for tool in offered_tools if tool.name != KNOWLEDGE_SEARCH_TOOL]

    # `input_guard` sets this when its judge is down. With only deterministic checks standing,
    # nothing may be changed.
    if state.get("read_only_tools", False):
        offered_tools = [tool for tool in offered_tools if tool.risk == "read"]
    return offered_tools


def _find_write_call(
    tool_calls: list[ToolCallRequest],
    dependencies: AgentDependencies,
    offered_tool_names: set[str],
) -> PendingToolCall | None:
    """Return the first requested write tool, which must wait for a person (rule 3).

    Only a tool that was offered can pause the turn. Anything else goes to the registry, which
    denies and audits it, so that a person is never asked to approve what the role cannot do.
    """
    for tool_call in tool_calls:
        tool_name = dependencies.tool_registry.resolve_model_name(tool_call.tool_name)
        definition = dependencies.tool_registry.get_definition(tool_name)
        if (
            definition is not None
            and definition.risk == "write"
            and definition.name in offered_tool_names
        ):
            return PendingToolCall(
                tool_call_id=tool_call.call_id, tool_name=tool_name, arguments=tool_call.arguments
            )
    return None


async def _run_tool_rounds(
    state: AgentState, dependencies: AgentDependencies, offered_tools: list[ToolDefinition]
) -> tuple[list[ToolResult], list[str], PendingToolCall | None]:
    """Let the model call tools and see the results, bounded by rounds and by total calls."""
    limits = dependencies.tool_limits
    tool_schemas = [definition.to_model_schema() for definition in offered_tools]
    offered_tool_names = {definition.name for definition in offered_tools}
    tool_results: list[ToolResult] = []
    degraded: list[str] = []
    calls_made = 0
    for _ in range(limits.max_rounds):
        messages = _build_messages(state, tool_results, state.get("evidence", []))

        # With no model, the turn carries on without tools.
        try:
            selection = await dependencies.llm.generate_tool_calls(
                TOOL_ROUTE, messages, tool_schemas
            )
        except AllModelsFailedError:
            degraded.append(DEGRADED_NO_TOOL_MODEL)
            break
        degraded.extend(selection.describe_fallbacks(DEGRADED_MODEL_FALLBACK))
        if not selection.value:
            break

        # A write tool stops the round, and the turn pauses before anything changes.
        pending_call = _find_write_call(selection.value, dependencies, offered_tool_names)
        if pending_call is not None:
            return tool_results, degraded, pending_call

        # Calls run at once, up to the turn's limit. Extra requests are ignored.
        allowed_calls = selection.value[: limits.max_calls_per_turn - calls_made]
        calls_made += len(allowed_calls)
        context = _build_tool_context(state, dependencies, tool_results)
        tool_results.extend(
            await asyncio.gather(
                *[_run_tool_call(call, dependencies, context) for call in allowed_calls]
            )
        )
        if calls_made >= limits.max_calls_per_turn:
            break
    return tool_results, degraded, None


async def _run_approved_call(state: AgentState, dependencies: AgentDependencies) -> dict[str, Any]:
    """Finish a paused write tool, now that a person has decided."""
    pending_call = state["pending_approval"]
    decision = state["approval_decision"]
    assert pending_call is not None and decision is not None

    if decision.approved and decision.tool_call_id == pending_call.tool_call_id:
        # Rule 2: `execute` checks the verified role again, and the arguments are the stored ones.
        context = _build_tool_context(state, dependencies, [])
        tool_result = await dependencies.tool_registry.execute(
            pending_call.tool_name, pending_call.arguments, context, approved=True
        )
    else:
        await dependencies.audit_log.record(
            state["user"], pending_call.tool_name, "rejected", "not approved by a person"
        )
        tool_result = ToolResult(
            tool_name=pending_call.tool_name, status="denied", message=NOT_APPROVED_MESSAGE
        )

    emit_event(
        EventType.TOOL,
        tool_name=pending_call.tool_name,
        arguments=pending_call.arguments,
        status=tool_result.status,
        message=tool_result.message,
    )
    return {
        "tool_results": [tool_result],
        "pending_approval": None,
        "approval_decision": None,
        "tools_finished": True,
    }


async def tool_agent(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Offer the tools this role may use, and run what the model asks for."""
    emit_node_started(NODE_NAME)

    # A person has decided on a paused write tool, so finish that and nothing else.
    if state.get("approval_decision") is not None:
        approved_update = await _run_approved_call(state, runtime.context)
        emit_node_finished(NODE_NAME)
        return approved_update

    # Step 1: choose what to offer, and skip the model if there is nothing.
    offered_tools = _select_offered_tools(state, runtime.context)
    tool_results: list[ToolResult] = []
    degraded: list[str] = []
    pending_call: PendingToolCall | None = None
    if offered_tools:
        tool_results, degraded, pending_call = await _run_tool_rounds(
            state, runtime.context, offered_tools
        )

    # Step 2: one failed tool must not fail the turn, but the employee should know about it.
    if any(tool_result.status in FAILED_STATUSES for tool_result in tool_results):
        degraded.append(DEGRADED_TOOL_FAILED)

    tool_evidence = [chunk for tool_result in tool_results for chunk in tool_result.evidence]
    emit_event(EventType.STATE, evidence_count=len(tool_evidence), degraded=degraded)
    emit_node_finished(NODE_NAME)
    return {
        "tool_results": tool_results,
        "evidence": tool_evidence,
        "pending_approval": pending_call,
        # The supervisor reads this, so that it does not route here a second time.
        "tools_finished": pending_call is None,
        "degraded": degraded,
    }


def choose_after_tool_agent(state: AgentState) -> Literal["human_approval", "supervisor"]:
    """Pause for a person if a write tool is waiting. Otherwise return to the supervisor."""
    return "human_approval" if state.get("pending_approval") is not None else "supervisor"

"""`tool_agent`: let a model choose tools, and run what it chose through the registry.

Rule 2 checks RBAC three times. This node filters what the model is offered, the registry checks
the role again before running anything, and each tool filters its own data. The model's reply is
only a request: its arguments are validated, and a name it was never offered is refused.

The loop is bounded by rounds and by total calls, and a failed tool is recorded rather than
raised, so the response agent can still answer from what remains.
"""

import asyncio
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import format_evidence_documents, format_tool_results
from agent.llm_gateway import AllModelsFailedError, ToolCallRequest
from agent.nodes.supervisor import latest_user_text
from agent.prompts.tool_agent import TOOL_AGENT_RESULTS_TEMPLATE, TOOL_AGENT_SYSTEM_PROMPT
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from retrieval.models import RetrievedChunk
from tools.models import ToolContext, ToolResult

NODE_NAME = "tool_agent"
# Choosing a tool is a small decision, so it uses the cheaper route.
TOOL_ROUTE = "fast"

DEGRADED_NO_TOOL_MODEL = "tool_selection_unavailable"
DEGRADED_MODEL_FALLBACK = "tool_agent_used_fallback_model"
DEGRADED_TOOL_FAILED = "tool_failed"
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
            records.extend(item for item in tool_result.data if isinstance(item, dict))
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


async def tool_agent(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Offer the tools this role may use, and run what the model asks for, in bounded rounds."""
    emit_node_started(NODE_NAME)
    dependencies = runtime.context
    limits = dependencies.tool_limits

    # Step 1: rule 2's first check, the model is only ever offered what this role may use.
    offered_tools = dependencies.tool_registry.tools_for_model(state["user"].role)
    tool_schemas = [definition.to_model_schema() for definition in offered_tools]

    tool_results: list[ToolResult] = []
    degraded: list[str] = []
    calls_made = 0
    for _ in range(limits.max_rounds):
        messages = _build_messages(state, tool_results, state.get("evidence", []))

        # Step 2: ask which tools to run. With no model, the turn carries on without them.
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

        # Step 3: run them at once, up to the turn's limit. Extra requests are ignored.
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

    # Step 4: one failed tool must not fail the turn, but the employee should know about it.
    if any(tool_result.status in FAILED_STATUSES for tool_result in tool_results):
        degraded.append(DEGRADED_TOOL_FAILED)

    tool_evidence = [chunk for tool_result in tool_results for chunk in tool_result.evidence]
    emit_event(EventType.STATE, evidence_count=len(tool_evidence), degraded=degraded)
    emit_node_finished(NODE_NAME)
    return {
        "tool_results": tool_results,
        "evidence": tool_evidence,
        "tools_finished": True,
        "degraded": degraded,
    }

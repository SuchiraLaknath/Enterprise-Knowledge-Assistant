"""Turn activity events into the lines shown in the activity panel.

This is plain Python with no Streamlit, so that it can be tested. Rule 9: the output is shown as
plain text, because it can include text written by the model.
"""

from dataclasses import dataclass
from typing import Any

NODE_STARTED = "started"
NODE_MEANINGS = {
    "load_context": "preparing this turn",
    "input_guard": "checking the message",
    "refuse": "refusing politely",
    "supervisor": "planning the work",
    "retrieval_agent": "searching the documents",
    "research_agent": "running deep research",
    "research_planner": "writing a research plan",
    "research_runner": "reading documents with sub-agents",
    "research_aggregator": "grouping the findings",
    "tool_agent": "running tools",
    "human_approval": "waiting for a person",
    "response_agent": "writing the answer",
    "validator": "checking the answer",
    "memory_writer": "saving what is useful",
}

BACKUP_MODEL_MARKER = "_used_fallback_model"
BACKUP_MODEL_SENTENCE = "A backup AI model answered some steps."
LOCAL_SEARCH_SENTENCE = "Document search was unavailable, so a local keyword search was used."
PARTIAL_RESEARCH_SENTENCE = "Part of the research failed, so some documents may be missing."
# These keys are the values the API sends in `degraded`. The UI talks to the API only through its
# events, so they are written out here. An id missing from this list is shown as it is.
DEGRADED_SENTENCES = {
    "guard_judge_unavailable_read_tools_only": (
        "The safety judge was unavailable, so only read tools were allowed."
    ),
    "long_term_memory_unavailable": "Long-term memory was off for this turn.",
    "conversation_summary_skipped": "The conversation summary was skipped.",
    "research_denied_for_role": "Your role does not include deep research.",
    "research_incomplete": PARTIAL_RESEARCH_SENTENCE,
    "research_failed_used_plain_retrieval": "Research failed, so a normal search was used.",
    "extractive_answer_without_llm": (
        "No AI model was available, so these are excerpts from the documents."
    ),
    "supervisor_plan_unavailable": "No planning model was available, so a default plan was used.",
    "tool_selection_unavailable": "Tools could not be chosen, so no tool was used.",
    "tool_failed": "A tool failed, so the answer uses the remaining evidence.",
    "answer_judge_unavailable": "Only code checked this answer, as the AI judge was unavailable.",
    "answer_failed_validation_excerpts_shown": (
        "The written answer failed its checks, so excerpts are shown instead."
    ),
    "research_grouped_by_label": "Findings were grouped by their own labels.",
    "research_planner_unavailable": "A fixed research plan was used.",
    "research_batch_failed": PARTIAL_RESEARCH_SENTENCE,
    "research_batch_truncated": "Some sections were left unread, as the batch was too large.",
    "research_budget_reached": "The research limit was reached, so some documents may be missing.",
    "research_documents_unresolved": "Some documents could not be fully covered.",
    "postgres_unavailable_in_memory_checkpoints": (
        "Saved storage was unavailable, so this conversation is kept in memory only."
    ),
    "rate_limit_in_process": "A local rate limiter was used.",
    "pinecone_unavailable": LOCAL_SEARCH_SENTENCE,
    "pinecone_circuit_open": LOCAL_SEARCH_SENTENCE,
    "local_keyword_search": LOCAL_SEARCH_SENTENCE,
    "namespace_unavailable": "One department could not be searched.",
    "rerank_unavailable": "Results were not reranked, so they stay in search order.",
}


def _describe_node(event: dict[str, Any]) -> str:
    """Show which node is active, with its name and what it does in plain words."""
    marker = "▶" if event["status"] == NODE_STARTED else "✓"
    meaning = NODE_MEANINGS.get(event["node"])
    return f"{marker} {event['node']}: {meaning}" if meaning else f"{marker} {event['node']}"


def _describe_state(event: dict[str, Any]) -> str:
    """Show the plan, the evidence count and any degradation."""
    lines = []
    if event.get("intent"):
        lines.append(f"  Intent: {event['intent']}")
    for step_number, step in enumerate(event.get("plan_steps", []), start=1):
        lines.append(f"  Step {step_number}: {step}")
    if "evidence_count" in event:
        lines.append(f"  Evidence: {event['evidence_count']} chunks")
    if "citation_count" in event:
        lines.append(f"  Citations: {event['citation_count']}")
    if event.get("degraded"):
        lines.append(f"  Degraded: {', '.join(event['degraded'])}")
    return "\n".join(lines)


def _describe_route(event: dict[str, Any]) -> str:
    """Show a routing decision."""
    return f"  Route: {event['source']} → {event['target']} ({event['reason']})"


def _describe_retrieval(event: dict[str, Any]) -> str:
    """Show what was searched for, how, and what came back."""
    lines = [
        f"  Search: {event['query']}",
        f"  Departments: {event['departments']}",
        f"  Dense weight: {event['dense_weight']}, results: {event['result_count']}",
    ]
    for source in event.get("sources", [])[:5]:
        lines.append(f"    {source['score']:.3f} {source['chunk_id']} {source['title']}")
    return "\n".join(lines)


def _describe_plan_ready(event: dict[str, Any]) -> str:
    """Show the generated plan in full, as it is the heart of the RLM flow."""
    heading = f"  Research plan (attempt {event['attempt']}, from {event['source']}):"
    code_lines = [f"    {line}" for line in event["code"].splitlines()]
    return "\n".join([heading, *code_lines])


def _describe_explored(event: dict[str, Any]) -> str:
    """Show the size of the collection the plan surveyed."""
    return f"  Explored catalog: {event['document_count']} documents"


def _describe_searched(event: dict[str, Any]) -> str:
    """Show one search that the plan ran."""
    return f"  Research search: {event['query']} ({event['result_count']} results)"


def _describe_sections_fetched(event: dict[str, Any]) -> str:
    """Show which sections the plan chose to read."""
    keyword = event["section_keyword"] or "all sections"
    return (
        f"  Fetched sections ({keyword}): {event['section_count']} "
        f"from {event['document_count']} documents"
    )


def _describe_batches_planned(event: dict[str, Any]) -> str:
    """Show how the work was divided."""
    return f"  Batches at depth {event['depth']}: sizes {event['batch_sizes']}"


def _describe_batch_finished(event: dict[str, Any]) -> str:
    """Show one sub-agent's result."""
    return (
        f"  Sub-agent at depth {event['depth']}: {event['finding_count']} findings "
        f"from {event['batch_size']} chunks ({event['chunks_read']} chunks read so far)"
    )


def _describe_plan_failed(event: dict[str, Any]) -> str:
    """Show why a plan stopped, and whether its work so far was kept."""
    if event["findings_kept"]:
        return (
            f"  Research plan stopped early, keeping {event['findings_kept']} findings: "
            f"{event['error']}"
        )
    return f"  Research plan failed: {event['error']}"


def _describe_report_ready(event: dict[str, Any]) -> str:
    """Show the themes and how much of the collection was read."""
    lines = [
        f"  Research: {event['documents_analysed']} of {event['documents_in_scope']} documents "
        "analysed, "
        f"{event['chunks_read']} of {event['chunks_available']} chunks read",
        f"  Depth reached: {event['depth_reached']}, sub-agents: {event['sub_agent_calls']}",
    ]
    for theme in event["themes"]:
        lines.append(f"    {theme['document_count']} documents: {theme['name']}")
    return "\n".join(lines)


# These keys are the values of `ResearchStage` in `api/events.py`. They are written out because
# the UI talks to the API only through its events, and a test checks that the two match.
RESEARCH_STAGE_DESCRIPTIONS = {
    "plan_ready": _describe_plan_ready,
    "explored": _describe_explored,
    "searched": _describe_searched,
    "sections_fetched": _describe_sections_fetched,
    "batches_planned": _describe_batches_planned,
    "batch_finished": _describe_batch_finished,
    "plan_failed": _describe_plan_failed,
    "report_ready": _describe_report_ready,
}


def _describe_research(event: dict[str, Any]) -> str:
    """Show one step of a research run, which reports its stages as they happen."""
    describe_stage = RESEARCH_STAGE_DESCRIPTIONS.get(event["stage"])
    return describe_stage(event) if describe_stage else f"  Research: {event['stage']}"


def _describe_tool(event: dict[str, Any]) -> str:
    """Show a tool call, its arguments and how it ended."""
    return f"  Tool: {event['tool_name']}({event['arguments']}) -> {event['status']}"


def _describe_validation(event: dict[str, Any]) -> str:
    """Show the result of a guard or validator check, and why."""
    reasons = f": {'; '.join(event['reasons'])}" if event.get("reasons") else ""
    return f"  Check {event['check']}: {event['status']}{reasons}"


def _describe_memory(event: dict[str, Any]) -> str:
    """Show a memory read or write."""
    return f"  Memory {event['operation']} ({event['tier']}): {event['item_count']} items"


def _describe_approval(event: dict[str, Any]) -> str:
    """Show that a write tool is waiting for a person."""
    return f"  Waiting for approval: {event['tool_name']}({event['arguments']})"


EVENT_DESCRIPTIONS = {
    "node": _describe_node,
    "state": _describe_state,
    "route": _describe_route,
    "retrieval": _describe_retrieval,
    "research": _describe_research,
    "tool": _describe_tool,
    "memory": _describe_memory,
    "approval": _describe_approval,
    "validation": _describe_validation,
}


def describe_event(event: dict[str, Any]) -> str | None:
    """Return the panel text for an event, or None if it belongs in the chat instead."""
    describe = EVENT_DESCRIPTIONS.get(event.get("type", ""))
    return describe(event) if describe else None


def add_activity_event(lines: list[str], event: dict[str, Any]) -> None:
    """Add an event to the panel. A finished node ticks its own line, so it takes only one."""
    is_finished_node = event.get("type") == "node" and event["status"] != NODE_STARTED
    if is_finished_node:
        started_line = _describe_node({**event, "status": NODE_STARTED})
        # Search from the end, because a node such as the supervisor runs several times.
        for position in range(len(lines) - 1, -1, -1):
            if lines[position] == started_line:
                lines[position] = _describe_node(event)
                return

    line = describe_event(event)
    if line:
        lines.append(line)


@dataclass
class TurnStatus:
    """What the line above the activity panel shows, built up from the events of one turn."""

    active_node: str = ""
    evidence_count: int = 0
    retries: int = 0
    is_finished: bool = False


def update_turn_status(status: TurnStatus, event: dict[str, Any]) -> None:
    """Keep the status line up to date with one event."""
    event_type = event.get("type")
    if event_type == "node" and event["status"] == NODE_STARTED:
        status.active_node = event["node"]
    elif event_type == "state":
        status.evidence_count = event.get("evidence_count", status.evidence_count)
        status.retries = event.get("response_retries", status.retries)
    elif event_type == "final":
        status.is_finished = True


def describe_turn_status(status: TurnStatus) -> str:
    """Return the status line, or a prompt if no turn has started."""
    if not status.active_node:
        return "Waiting for a question."
    step = "Finished" if status.is_finished else f"Now: {status.active_node}"
    return f"{step} | Evidence: {status.evidence_count} | Retries: {status.retries}"


def describe_waiting_step(status: TurnStatus) -> str:
    """Return what the chat shows before the first word of an answer arrives."""
    meaning = NODE_MEANINGS.get(status.active_node, "starting")
    return f"{meaning.capitalize()}. See Agent activity for each step."


def describe_degraded(reasons: list[str]) -> list[str]:
    """Turn `degraded` ids into plain sentences, each one once, in the order they happened."""
    sentences: list[str] = []
    for reason in reasons:
        sentence = DEGRADED_SENTENCES.get(reason, reason)
        if BACKUP_MODEL_MARKER in reason:
            sentence = BACKUP_MODEL_SENTENCE
        if sentence not in sentences:
            sentences.append(sentence)
    return sentences


def apply_token_event(draft: str, event: dict[str, Any]) -> str:
    """Return the draft after one token event. A reset means that a model failed mid-answer."""
    if event.get("reset"):
        return ""
    return draft + event.get("text", "")


def parse_sse_line(line: str) -> str | None:
    """Return the data of an SSE line, or None for keep-alive pings and other lines."""
    if line.startswith("data:"):
        return line.removeprefix("data:").strip()
    return None

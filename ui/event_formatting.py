"""Turn activity events into the lines shown in the activity panel.

This is plain Python with no Streamlit, so that it can be tested. Rule 9: the output is shown as
plain text, because it can include text written by the model.
"""

from typing import Any


def _describe_node(event: dict[str, Any]) -> str:
    """Show which node is active."""
    marker = "▶" if event["status"] == "started" else "✓"
    return f"{marker} {event['node']}"


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
        f"  Research: {event['documents_analysed']} documents analysed, "
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


EVENT_DESCRIPTIONS = {
    "node": _describe_node,
    "state": _describe_state,
    "route": _describe_route,
    "retrieval": _describe_retrieval,
    "research": _describe_research,
}


def describe_event(event: dict[str, Any]) -> str | None:
    """Return the panel text for an event, or None if it belongs in the chat instead."""
    describe = EVENT_DESCRIPTIONS.get(event.get("type", ""))
    return describe(event) if describe else None


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

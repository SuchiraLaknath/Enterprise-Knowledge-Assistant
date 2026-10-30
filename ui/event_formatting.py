"""Turn activity events into the lines shown in the activity panel.

This is plain Python with no Streamlit, so that it can be tested. The output is shown as plain
text, because it can include text written by the model.
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
    lines.append(f"  Evidence: {event.get('evidence_count', 0)} chunks")
    if event.get("degraded"):
        lines.append(f"  Degraded: {', '.join(event['degraded'])}")
    return "\n".join(lines)


def _describe_route(event: dict[str, Any]) -> str:
    """Show a routing decision."""
    return f"  Route: {event['source']} → {event['target']}"


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


EVENT_DESCRIPTIONS = {
    "node": _describe_node,
    "state": _describe_state,
    "route": _describe_route,
    "retrieval": _describe_retrieval,
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

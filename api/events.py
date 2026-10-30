"""The activity events that graph nodes stream to the UI.

Every internal step is reported as one of these, so an evaluator can follow what the agent is
doing. Nodes emit them, the API forwards them over SSE, and the activity panel displays them.
"""

from enum import StrEnum
from typing import Any

from langgraph.config import get_stream_writer


class EventType(StrEnum):
    """Every kind of event. The UI decides how to display each one."""

    # A node started or finished.
    NODE = "node"
    # The current plan step, evidence count, retries and degraded list.
    STATE = "state"
    # A routing decision between nodes.
    ROUTE = "route"
    # What a search looked for and found.
    RETRIEVAL = "retrieval"
    # The generated plan, batches and depth of recursive research.
    RESEARCH = "research"
    # A tool call and its result.
    TOOL = "tool"
    # A memory read or write.
    MEMORY = "memory"
    # The result of a validation check.
    VALIDATION = "validation"
    # A pause waiting for human approval.
    APPROVAL = "approval"
    # A fragment of the provisional draft, or a signal to clear it.
    TOKEN = "token"
    # The authoritative answer, which confirms or replaces the draft.
    FINAL = "final"
    # A failure the user should know about.
    ERROR = "error"


def build_event(event_type: EventType, **details: Any) -> dict[str, Any]:
    """Return an event in the shape that is sent to the UI."""
    return {"type": event_type.value, **details}


def emit_event(event_type: EventType, **details: Any) -> None:
    """Send an event to the UI from inside a graph node."""
    write_to_stream = get_stream_writer()
    write_to_stream(build_event(event_type, **details))


def emit_node_started(node_name: str) -> None:
    """Report that a node has started."""
    emit_event(EventType.NODE, node=node_name, status="started")


def emit_node_finished(node_name: str) -> None:
    """Report that a node has finished."""
    emit_event(EventType.NODE, node=node_name, status="finished")

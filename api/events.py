"""The activity events that graph nodes stream to the UI.

Every internal step is reported as one of these, so an evaluator can follow what the agent is
doing. Nodes emit them, the API forwards them over SSE, and the activity panel displays them.
"""

from enum import StrEnum
from typing import Any

from langgraph.config import get_stream_writer

from security.stream_masker import mask_sensitive_text


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
    # The link to this turn's trace, sent after the answer so that it can never delay it.
    TRACE = "trace"
    # A failure the user should know about.
    ERROR = "error"


class ResearchStage(StrEnum):
    """The steps of a research run, reported in the `stage` field of `research` events."""

    # A plan is ready, from the LLM or the template.
    PLAN_READY = "plan_ready"
    # The catalog was surveyed.
    EXPLORED = "explored"
    # A research search finished.
    SEARCHED = "searched"
    # Targeted sections were read.
    SECTIONS_FETCHED = "sections_fetched"
    # The plan's batches passed the budget check.
    BATCHES_PLANNED = "batches_planned"
    # One sub-agent finished.
    BATCH_FINISHED = "batch_finished"
    # A plan failed, or stopped early and left some findings.
    PLAN_FAILED = "plan_failed"
    # The themes and coverage are ready.
    REPORT_READY = "report_ready"


def _mask_event_values(value: Any) -> Any:
    """Mask every string in an event, however deeply it is nested."""
    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, dict):
        return {key: _mask_event_values(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_mask_event_values(item) for item in value]
    return value


def build_event(event_type: EventType, **details: Any) -> dict[str, Any]:
    """Return an event in the shape that is sent to the UI.

    Rule 9: every event passes through here, so a query, plan or title that quotes a sensitive
    number is masked as well as the answer, without each node having to remember.
    """
    return {"type": event_type.value, **_mask_event_values(details)}


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

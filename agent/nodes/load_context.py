"""`load_context`: prepare the state for a new turn.

The checkpointer restores everything from the previous turn, so this node clears what belonged
to that turn. It is the first node, so that nothing can read stale evidence. It also loads what
is remembered about this employee, which is the only long-term memory read in a turn.
"""

import asyncio
from typing import Any

import structlog
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import format_memory_context
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started

logger = structlog.get_logger()

NODE_NAME = "load_context"
DEGRADED_MEMORY_OFF = "long_term_memory_unavailable"


async def _read_long_term_facts(
    state: AgentState, dependencies: AgentDependencies
) -> tuple[list[str], list[str]]:
    """Return this employee's facts. Memory is a bonus, so a failure only switches it off."""
    limits = dependencies.conversation_limits
    # Rule 8: the namespace is the verified user's id, so one user cannot read another's memory.
    user_id = state["user"].user_id
    try:
        async with asyncio.timeout(limits.memory_timeout_s):
            memories = await dependencies.long_term_memory.read(user_id, limits.max_memories_read)
    except Exception as error:  # Failures are data, and the turn carries on without memory.
        logger.warning("memory_read_failed", error=type(error).__name__)
        return [], [DEGRADED_MEMORY_OFF]
    return [memory["fact"] for memory in memories], []


async def load_context(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Clear the per-turn fields, and load memory. `Overwrite` bypasses the additive reducers."""
    emit_node_started(NODE_NAME)
    facts, memory_degraded = await _read_long_term_facts(state, runtime.context)
    emit_event(EventType.MEMORY, operation="read", tier="long_term", item_count=len(facts))

    summary = state.get("conversation_summary", "")
    fresh_turn_state: dict[str, Any] = {
        "plan": None,
        "retrieval_finished": False,
        "research_allowed": False,
        "research_finished": False,
        "research_report": None,
        "tools_finished": False,
        "pending_approval": None,
        "approval_decision": None,
        "tool_results": Overwrite([]),
        "turn_notice": None,
        "read_only_tools": False,
        "input_blocked": False,
        "validation_report": None,
        "validation_feedback": "",
        "conversation_note": None,
        "answer_is_excerpts": False,
        "memory_context": format_memory_context(summary, facts),
        "evidence": Overwrite([]),
        "degraded": Overwrite([*state.get("request_degraded", []), *memory_degraded]),
        "response_retries": 0,
        "evidence_retries": 0,
        "final_answer": "",
        "final_citations": [],
    }
    emit_node_finished(NODE_NAME)
    return fresh_turn_state

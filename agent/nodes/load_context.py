"""`load_context`: prepare the state for a new turn.

The checkpointer restores everything from the previous turn, so this node clears what belonged
to that turn. It is the first node, so that nothing can read stale evidence.
"""

from typing import Any

from langgraph.types import Overwrite

from agent.state import AgentState
from api.events import emit_node_finished, emit_node_started

NODE_NAME = "load_context"


async def load_context(state: AgentState) -> dict[str, Any]:
    """Clear the per-turn fields. `Overwrite` bypasses the additive reducers."""
    emit_node_started(NODE_NAME)
    fresh_turn_state: dict[str, Any] = {
        "plan": None,
        "retrieval_finished": False,
        "research_finished": False,
        "research_report": None,
        "turn_notice": None,
        "evidence": Overwrite([]),
        "degraded": Overwrite([]),
        "response_retries": 0,
        "evidence_retries": 0,
        "final_answer": "",
        "final_citations": [],
    }
    emit_node_finished(NODE_NAME)
    return fresh_turn_state

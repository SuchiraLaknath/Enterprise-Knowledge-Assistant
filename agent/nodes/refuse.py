"""`refuse`: answer a blocked message courteously, in the bank's voice.

The reply is fixed text, so that a blocked message reaches no model that writes answers. It does
not say which check fired, because that would help an attacker to reword the attempt.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from agent.state import AgentState
from api.events import emit_node_finished, emit_node_started

NODE_NAME = "refuse"
REFUSAL_MESSAGE = (
    "I'm sorry, but I can't help with that request. I can answer questions about Commercial "
    "Bank's policies, runbooks, incidents and products. Please let me know how I can help."
)


async def refuse(state: AgentState) -> dict[str, Any]:
    """End the turn with the fixed refusal.

    `input_guard` has already reported why, and the chat route sends this as the `final` event.
    """
    emit_node_started(NODE_NAME)
    # Rule 5: the attack must not stay in the conversation, or "do what I said earlier" would
    # bring it back past the guard, which reads only the newest message.
    blocked_messages = [
        RemoveMessage(id=message.id)
        for message in state["messages"][-1:]
        if isinstance(message, HumanMessage) and message.id
    ]
    emit_node_finished(NODE_NAME)
    return {
        "messages": [*blocked_messages, AIMessage(REFUSAL_MESSAGE)],
        "final_answer": REFUSAL_MESSAGE,
        "final_citations": [],
    }

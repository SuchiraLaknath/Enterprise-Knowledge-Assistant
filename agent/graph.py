"""Wire the nodes into the agent graph.

One turn follows this path:

    api/routes/chat.py    verifies the user and starts the graph
      load_context        clears what belonged to the previous turn
      supervisor          understands the question and makes a plan
      retrieval_agent     finds evidence, if the plan needs documents
      supervisor          sees the worker has returned, and chooses the next step
      response_agent      streams a masked draft, then stores the finished answer
    api/routes/chat.py    sends the `final` event and the trace link

Each node reports its progress with events from `api/events.py`, and the chat route forwards
them to the activity panel. Later build steps add the guards, research, tools, approval,
validation and memory nodes shown in `docs/ARCHITECTURE.md`.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.dependencies import AgentDependencies
from agent.nodes.load_context import load_context
from agent.nodes.response_agent import response_agent
from agent.nodes.retrieval_agent import retrieval_agent
from agent.nodes.supervisor import choose_next_node, supervisor
from agent.state import AgentState

AgentGraph = CompiledStateGraph[AgentState, AgentDependencies, AgentState, AgentState]

# The only application types a checkpoint may rebuild. Loading a checkpoint can create objects,
# so naming them stops a tampered checkpoint from creating anything else.
CHECKPOINT_TYPES = (
    ("agent.schemas", "Citation"),
    ("agent.schemas", "Finding"),
    ("agent.schemas", "Plan"),
    ("agent.schemas", "ResearchReport"),
    ("agent.schemas", "ResearchTheme"),
    ("retrieval.models", "RetrievedChunk"),
    ("retrieval.models", "UserContext"),
)


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Return the serializer for checkpoints, limited to the application's own types."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def build_agent_graph(checkpointer: BaseCheckpointSaver[Any]) -> AgentGraph:
    """Return the compiled graph. The checkpointer gives conversations their memory."""
    graph = StateGraph(AgentState, context_schema=AgentDependencies)

    graph.add_node("load_context", load_context)
    graph.add_node("supervisor", supervisor)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("response_agent", response_agent)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "supervisor")
    # Workers return to the supervisor, which decides what happens next.
    graph.add_conditional_edges("supervisor", choose_next_node)
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("response_agent", END)

    return graph.compile(checkpointer=checkpointer)

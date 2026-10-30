"""Wire the nodes into the agent graph.

This skeleton covers planning, retrieval and the response. Later build steps add the guards,
research, tools, approval, validation and memory nodes shown in `docs/ARCHITECTURE.md`.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.dependencies import AgentDependencies
from agent.nodes.load_context import load_context
from agent.nodes.response_agent import response_agent
from agent.nodes.retrieval_agent import retrieval_agent
from agent.nodes.supervisor import choose_next_node, supervisor
from agent.state import AgentState

AgentGraph = CompiledStateGraph[AgentState, AgentDependencies, AgentState, AgentState]


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

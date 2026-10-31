"""`human_approval`: pause the graph until a person decides on a write tool.

LangGraph re-runs a node from its start on resume, so this node does nothing except call
`interrupt()`. Anything placed before that call would run twice. The tool itself runs afterwards
in `tool_agent`, which checks the role again and uses only the arguments stored with the approved
`tool_call_id`. The person's answer can approve those arguments, but it cannot change them.
"""

from typing import Any

from langgraph.types import interrupt

from agent.state import AgentState


async def human_approval(state: AgentState) -> dict[str, Any]:
    """Wait for the decision that `api/routes/approve.py` resumes the graph with."""
    return {"approval_decision": interrupt("A write tool needs approval.")}

"""The approval route: a person decides on a paused write tool, and the graph resumes.

The request says only which call and whether it is approved. It holds no arguments, so what runs
is exactly what was shown. The thread id is scoped to the verified user, so nobody can reach
another person's paused call, and the role is checked again here and when the tool runs.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.schemas import ApprovalDecision
from api.auth import get_current_user, get_services
from api.routes.chat import (
    THREAD_ID_PATTERN,
    build_checkpoint_thread_id,
    choose_agent_graph,
    stream_graph_run,
)
from api.services import ApplicationServices
from retrieval.models import UserContext

router = APIRouter()

NO_PENDING_CALL_MESSAGE = "There is no paused action with this id."
ROLE_DENIED_MESSAGE = "Your role does not allow this action."


class ApprovalRequest(BaseModel):
    """A person's decision on one paused tool call."""

    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    tool_call_id: str = Field(min_length=1, max_length=200)
    approved: bool


@router.post("/approve")
async def approve(
    approval: ApprovalRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    services: Annotated[ApplicationServices, Depends(get_services)],
) -> EventSourceResponse:
    """Resume a paused turn with a person's decision, streaming the rest of it."""
    # Step 1: the thread id holds the verified user, so this can only find their own turn.
    agent_graph, _ = await choose_agent_graph(services)
    run_config: RunnableConfig = {
        "configurable": {"thread_id": build_checkpoint_thread_id(user, approval.thread_id)}
    }
    paused_turn = await agent_graph.aget_state(run_config)
    pending_call = paused_turn.values.get("pending_approval")

    # Step 2: the decision must match the call that was shown.
    if (
        not paused_turn.next
        or pending_call is None
        or pending_call.tool_call_id != approval.tool_call_id
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, detail=NO_PENDING_CALL_MESSAGE)

    # Step 3: rule 2, check the role again. It may have changed since the turn paused.
    tool_registry = services.agent_dependencies.tool_registry
    if not tool_registry.is_role_allowed(pending_call.tool_name, user.role):
        await services.agent_dependencies.audit_log.record(
            user, pending_call.tool_name, "denied", "approval by a role below min_role"
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=ROLE_DENIED_MESSAGE)

    # Step 4: resume. The tool runs later in the graph, with the stored arguments.
    decision = ApprovalDecision(tool_call_id=approval.tool_call_id, approved=approval.approved)
    return EventSourceResponse(
        stream_graph_run(agent_graph, Command(resume=decision), approval.thread_id, user, services)
    )

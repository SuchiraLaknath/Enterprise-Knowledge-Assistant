"""The chat route: run one graph turn and stream its activity as server-sent events."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.state import AgentState
from api.auth import get_current_user, get_services
from api.events import EventType, build_event
from api.services import ApplicationServices
from retrieval.models import UserContext

logger = structlog.get_logger()
router = APIRouter()

# Only the provisional draft is sent by nodes, so the route confirms it in the final event.
DRAFT_CONFIRMED = "confirmed"
TURN_FAILED_MESSAGE = "Something went wrong while answering. Please try again."


class ChatRequest(BaseModel):
    """One message from the employee.

    The length limit comes from config, so the route checks it. The thread id format stops a
    client from placing odd characters in the checkpoint key.
    """

    message: str = Field(min_length=1)
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


def build_checkpoint_thread_id(user: UserContext, thread_id: str) -> str:
    """Scope a conversation to its owner, so that guessing an id cannot open someone else's."""
    return f"{user.user_id}:{thread_id}"


def as_sse_message(event: dict[str, Any]) -> dict[str, str]:
    """Wrap an activity event in the shape that the SSE response sends."""
    return {"data": json.dumps(event, default=str)}


async def find_trace_url(run_id: uuid.UUID, project_name: str) -> str | None:
    """Return a link to this turn's LangSmith trace, or None if it cannot be found.

    A missing link should never cost the employee their answer, so failures are only logged.
    """
    client = LangSmithClient()
    try:
        project = await asyncio.to_thread(client.read_project, project_name=project_name)
        run_link = await client.runs.get_url(
            str(run_id), project_id=str(project.id), trace_id=str(run_id)
        )
        return run_link.url
    except Exception as error:  # Tracing problems must not affect the answer.
        logger.warning("trace_url_unavailable", error=type(error).__name__)
        return None


def build_final_event(state_values: dict[str, Any], trace_url: str | None) -> dict[str, Any]:
    """Return the authoritative answer, which confirms or replaces the draft (rule 9)."""
    return build_event(
        EventType.FINAL,
        answer=state_values["messages"][-1].text,
        citations=[citation.model_dump() for citation in state_values["final_citations"]],
        degraded=state_values.get("degraded", []),
        draft_status=DRAFT_CONFIRMED,
        trace_url=trace_url,
    )


async def stream_chat_turn(
    request: ChatRequest, user: UserContext, services: ApplicationServices
) -> AsyncIterator[dict[str, str]]:
    """Run the graph once, forwarding each activity event as it happens."""
    run_id = uuid.uuid4()
    checkpoint_thread_id = build_checkpoint_thread_id(user, request.thread_id)
    structlog.contextvars.bind_contextvars(
        run_id=str(run_id), thread_id=request.thread_id, user_id=user.user_id
    )
    run_config: RunnableConfig = {
        "configurable": {"thread_id": checkpoint_thread_id},
        "run_id": run_id,
        "run_name": "chat_turn",
        "recursion_limit": services.api_limits.graph_recursion_limit,
        "metadata": {
            "user_id": user.user_id,
            "role": user.role,
            "thread_id": request.thread_id,
        },
    }
    # Rule 1: the user comes from the verified token, and the graph receives it as data.
    graph_input: AgentState = {"messages": [HumanMessage(request.message)], "user": user}

    try:
        async for activity_event in services.agent_graph.astream(
            graph_input,
            run_config,
            context=services.agent_dependencies,
            stream_mode="custom",
        ):
            yield as_sse_message(activity_event)

        final_state = await services.agent_graph.aget_state(run_config)
        trace_url = await find_trace_url(run_id, services.langsmith_project)
        yield as_sse_message(build_final_event(final_state.values, trace_url))
    except Exception:  # Show a safe message, and keep the details in the logs.
        logger.exception("chat_turn_failed")
        yield as_sse_message(build_event(EventType.ERROR, message=TURN_FAILED_MESSAGE))
    finally:
        structlog.contextvars.clear_contextvars()


@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    services: Annotated[ApplicationServices, Depends(get_services)],
) -> EventSourceResponse:
    """Answer one message, streaming the agent's activity and then the final answer."""
    if len(chat_request.message) > services.api_limits.max_message_length:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Message is too long.")
    return EventSourceResponse(stream_chat_turn(chat_request, user, services))

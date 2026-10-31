"""The chat route: run one graph turn and stream its activity as server-sent events.

Nodes create the activity events, using `api/events.py`, and this route only forwards them. A
turn ends with the `final` event, and then a `trace` event, so that a slow trace lookup can
never hold back an answer.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.state import AgentState
from api.auth import get_current_user, get_services
from api.checkpoint_store import DEGRADED_IN_MEMORY_CHECKPOINTS
from api.events import EventType, build_event
from api.services import ApplicationServices
from retrieval.models import UserContext

logger = structlog.get_logger()
router = APIRouter()

# Rule 9: the final event says whether the answer matches the draft the employee watched.
DRAFT_CONFIRMED = "confirmed"
DRAFT_REPLACED = "replaced"
TURN_FAILED_MESSAGE = "Something went wrong while answering. Please try again."

# With `subgraphs=True`, each event comes with the path of the subgraph that produced it.
SubgraphEvent = tuple[tuple[str, ...], dict[str, Any]]


class ChatRequest(BaseModel):
    """One message from the employee.

    The length limit comes from config, so the route checks it. The thread id format stops a
    client from placing odd characters in the checkpoint key.
    """

    message: str = Field(min_length=1)
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


def build_checkpoint_thread_id(user: UserContext, thread_id: str) -> str:
    """Scope a conversation to its owner and role.

    Guessing an id cannot open someone else's thread. The role is included so that a demoted
    user starts afresh, rather than keeping answers from their earlier access level.
    """
    return f"{user.user_id}:{user.role}:{thread_id}"


def as_sse_message(event: dict[str, Any]) -> dict[str, str]:
    """Wrap an activity event in the shape that the SSE response sends."""
    return {"data": json.dumps(event, default=str)}


async def find_trace_url(
    run_id: uuid.UUID, project_name: str, timeout_seconds: float
) -> str | None:
    """Return a link to this turn's LangSmith trace, or None if it cannot be found.

    A missing link should never affect the employee, so failures are only logged.
    """

    async def look_up_url() -> str:
        client = LangSmithClient()
        project = await asyncio.to_thread(client.read_project, project_name=project_name)
        # Creating this resource makes a blocking network call, so it runs in a thread.
        runs_resource = await asyncio.to_thread(lambda: client.runs)
        run_link = await runs_resource.get_url(
            str(run_id), project_id=str(project.id), trace_id=str(run_id)
        )
        return str(run_link.url)

    try:
        return await asyncio.wait_for(look_up_url(), timeout=timeout_seconds)
    except Exception as error:  # Tracing problems must not affect the answer.
        logger.warning("trace_url_unavailable", error=type(error).__name__)
        return None


def build_final_event(state_values: dict[str, Any], draft: str) -> dict[str, Any]:
    """Return the authoritative answer, which confirms or replaces the draft (rule 9)."""
    answer = state_values["final_answer"]
    return build_event(
        EventType.FINAL,
        answer=answer,
        citations=[citation.model_dump() for citation in state_values["final_citations"]],
        degraded=state_values.get("degraded", []),
        draft_status=DRAFT_CONFIRMED if answer == draft else DRAFT_REPLACED,
    )


async def stream_chat_turn(
    request: ChatRequest,
    user: UserContext,
    services: ApplicationServices,
    request_degraded: list[str],
) -> AsyncIterator[dict[str, str]]:
    """Run the graph once, forwarding each activity event as it happens."""
    # Step 1: scope the turn to this user, and bind its ids to every log line.
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
    graph_input: AgentState = {
        "messages": [HumanMessage(request.message)],
        "user": user,
        "request_degraded": request_degraded,
    }

    # Step 2: use Postgres if it can answer. A turn must not fail because a database did.
    agent_graph = services.agent_graph
    if not await services.checkpoint_store.is_available():
        agent_graph = services.in_memory_graph
        graph_input["request_degraded"] = [*request_degraded, DEGRADED_IN_MEMORY_CHECKPOINTS]

    # Step 3: forward each event as the nodes produce it, tracking the draft the employee sees.
    draft = ""
    try:
        # `subgraphs=True` is what forwards events from the research subgraph to the panel.
        async for stream_item in agent_graph.astream(
            graph_input,
            run_config,
            context=services.agent_dependencies,
            stream_mode="custom",
            subgraphs=True,
        ):
            _subgraph_path, activity_event = cast(SubgraphEvent, stream_item)
            if activity_event["type"] == EventType.TOKEN.value:
                draft = "" if activity_event.get("reset") else draft + activity_event["text"]
            yield as_sse_message(activity_event)

        # Step 4: send the authoritative answer, and then the trace link.
        final_state = await agent_graph.aget_state(run_config)
        yield as_sse_message(build_final_event(final_state.values, draft))
        trace_url = await find_trace_url(
            run_id, services.langsmith_project, services.api_limits.trace_lookup_timeout_s
        )
        yield as_sse_message(build_event(EventType.TRACE, url=trace_url))
    except Exception:  # Show a safe message, and keep the details in the logs.
        logger.exception("chat_turn_failed")
        yield as_sse_message(build_event(EventType.ERROR, message=TURN_FAILED_MESSAGE))
    finally:
        # The ids are cleared so that they cannot leak into the next request's logs.
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

    # This runs before the stream opens, so a refused request costs no LLM call.
    decision = await services.rate_limiter.check(user)
    if not decision.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a moment and try again.",
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    return EventSourceResponse(stream_chat_turn(chat_request, user, services, decision.degraded))

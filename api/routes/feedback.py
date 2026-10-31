"""The feedback route: an employee rates an answer.

A rating goes to Postgres, for the evals set, and to LangSmith, beside the trace of that turn.
Neither failing is the employee's problem, so the reply says what was recorded instead of raising.
"""

import asyncio
import uuid
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field

from api.auth import get_current_user, get_services
from api.services import ApplicationServices
from retrieval.models import UserContext

logger = structlog.get_logger()
router = APIRouter()

LANGSMITH_FEEDBACK_KEY = "user_rating"
SCORE_BY_RATING = {"up": 1, "down": 0}


class FeedbackRequest(BaseModel):
    """One employee's rating of one answer."""

    run_id: uuid.UUID
    rating: Literal["up", "down"]
    question: str = Field(min_length=1, max_length=4000)


class FeedbackReceipt(BaseModel):
    """Where the rating was stored."""

    stored: bool
    sent_to_langsmith: bool


async def send_to_langsmith(run_id: uuid.UUID, rating: str, timeout_s: float) -> bool:
    """Attach the rating to the turn's trace. A tracing problem must not affect the employee."""

    def create_feedback() -> None:
        LangSmithClient().create_feedback(
            run_id=run_id, key=LANGSMITH_FEEDBACK_KEY, score=SCORE_BY_RATING[rating]
        )

    try:
        await asyncio.wait_for(asyncio.to_thread(create_feedback), timeout=timeout_s)
    except Exception as error:  # Tracing problems must not affect the employee.
        logger.warning("feedback_not_sent_to_langsmith", error=type(error).__name__)
        return False
    return True


@router.post("/feedback")
async def record_feedback(
    feedback: FeedbackRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    services: Annotated[ApplicationServices, Depends(get_services)],
) -> FeedbackReceipt:
    """Store a rating. Thumbs-down questions become candidates for the evals set."""
    stored = await services.feedback_store.record(
        user, feedback.run_id, feedback.rating, feedback.question
    )
    sent_to_langsmith = await send_to_langsmith(
        feedback.run_id, feedback.rating, services.api_limits.trace_lookup_timeout_s
    )
    return FeedbackReceipt(stored=stored, sent_to_langsmith=sent_to_langsmith)

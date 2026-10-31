"""Store answer ratings, and read back the ones that became eval candidates.

Each rating holds the question the employee typed, and no answer or document text. Thumbs-down
questions feed `evals/export_candidates.py`, which is how a poor answer becomes a test.
"""

import uuid
from typing import Any, Protocol

import structlog
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from retrieval.models import UserContext

logger = structlog.get_logger()

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS answer_feedback (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    rating TEXT NOT NULL,
    question TEXT NOT NULL
)
"""
INSERT_RATING_STATEMENT = """
INSERT INTO answer_feedback (run_id, user_id, role, rating, question) VALUES (%s, %s, %s, %s, %s)
"""


class FeedbackStore(Protocol):
    """What the feedback route needs, so that tests can run with no Postgres."""

    async def record(
        self, user: UserContext, run_id: uuid.UUID, rating: str, question: str
    ) -> bool:
        """Store one rating, and return whether it was stored."""
        ...


class PostgresFeedbackStore:
    """Stores ratings in Postgres."""

    def __init__(self, connection_pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]]):
        """Share the application's pool."""
        self._connection_pool = connection_pool

    async def setup(self) -> None:
        """Create the table if this is the first run."""
        async with self._connection_pool.connection() as connection:
            await connection.execute(CREATE_TABLE_STATEMENT)

    async def record(
        self, user: UserContext, run_id: uuid.UUID, rating: str, question: str
    ) -> bool:
        """Store one rating. A failure is reported, and never raised to the employee."""
        try:
            async with self._connection_pool.connection() as connection:
                await connection.execute(
                    INSERT_RATING_STATEMENT, (run_id, user.user_id, user.role, rating, question)
                )
        except Exception as error:  # A lost rating must not become an error for the employee.
            logger.warning("feedback_write_failed", error=type(error).__name__)
            return False
        return True

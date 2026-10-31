"""Decide, for each turn, whether Postgres can hold its checkpoints.

If Postgres is down, a turn runs on an in-memory checkpointer instead of failing. That turn still
gets a full answer, and only loses its conversation history once the process restarts (failure
table). The check runs per turn, so the service recovers by itself as soon as Postgres returns.
"""

import asyncio
from collections.abc import Awaitable, Callable, Collection
from typing import Any, Protocol

import structlog
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger()

POSTGRES_FAULT = "postgres"
DEGRADED_IN_MEMORY_CHECKPOINTS = "postgres_unavailable_in_memory_checkpoints"
HEALTH_QUERY = "SELECT 1"

SetupStep = Callable[[], Awaitable[None]]


class CheckpointStore(Protocol):
    """What the chat route needs to know, so that tests can run with no Postgres."""

    async def is_available(self) -> bool:
        """Return whether Postgres can be used for this turn."""
        ...


class PostgresCheckpointStore:
    """Checks Postgres, and creates its tables the first time that it can."""

    def __init__(
        self,
        connection_pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]],
        setup_steps: list[SetupStep],
        timeout_s: float,
        enabled_faults: Collection[str] = (),
    ) -> None:
        """Keep the steps, because Postgres may be down when the API starts."""
        self._connection_pool = connection_pool
        self._setup_steps = setup_steps
        self._timeout_s = timeout_s
        self._enabled_faults = enabled_faults
        self._is_set_up = False

    async def is_available(self) -> bool:
        """Run a quick query. The timeout matters, as this delays turns while Postgres is down."""
        if POSTGRES_FAULT in self._enabled_faults:
            return False
        try:
            async with asyncio.timeout(self._timeout_s):
                async with self._connection_pool.connection() as connection:
                    await connection.execute(HEALTH_QUERY)
                if not self._is_set_up:
                    for setup_step in self._setup_steps:
                        await setup_step()
                    self._is_set_up = True
        except Exception as error:  # Any failure means using the fallback for this turn.
            logger.warning("postgres_unavailable", error=type(error).__name__)
            return False
        return True

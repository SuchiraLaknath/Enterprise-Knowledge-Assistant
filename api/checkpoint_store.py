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


class CheckpointAvailability(Protocol):
    """What the chat route needs to know, so that tests can run with no Postgres."""

    async def is_available(self) -> bool:
        """Return whether Postgres can be used for this turn."""
        ...


class PostgresAvailabilityCheck:
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
        self._setup_lock = asyncio.Lock()

    async def is_available(self) -> bool:
        """Answer quickly, because this runs at the start of every turn."""
        if POSTGRES_FAULT in self._enabled_faults:
            return False
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._ping_postgres()
                await self._ensure_tables_exist()
        except Exception as error:  # Any failure means using the fallback for this turn.
            logger.warning("postgres_unavailable", error=type(error).__name__)
            return False
        return True

    async def _ping_postgres(self) -> None:
        """Run the smallest possible query, which fails if Postgres cannot answer."""
        async with self._connection_pool.connection() as connection:
            await connection.execute(HEALTH_QUERY)

    async def _ensure_tables_exist(self) -> None:
        """Create the tables once. This waits until here, as Postgres may be down at startup."""
        # The lock stops two turns that arrive together from both running the setup.
        async with self._setup_lock:
            if self._is_set_up:
                return
            for setup_step in self._setup_steps:
                await setup_step()
            self._is_set_up = True

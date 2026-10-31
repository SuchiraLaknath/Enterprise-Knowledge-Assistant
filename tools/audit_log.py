"""The audit log: who used which tool, and what happened.

Rule 2 requires an entry for every denied call. Allowed calls are recorded as well, so that an
administrator can see the whole picture. Entries hold tool names and outcomes, and never tool
arguments, which could hold personal data. A failed write is kept in memory rather than raised,
because an audit outage must not take the assistant down, and the caller is told, so that a change
which cannot be recorded does not happen.
"""

from collections import deque
from datetime import UTC, datetime
from typing import Any

import structlog
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from retrieval.models import UserContext

logger = structlog.get_logger()

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    outcome TEXT NOT NULL,
    detail TEXT NOT NULL
)
"""
INSERT_ENTRY_STATEMENT = """
INSERT INTO audit_log (user_id, role, tool_name, outcome, detail) VALUES (%s, %s, %s, %s, %s)
"""
SELECT_RECENT_STATEMENT = """
SELECT created_at, user_id, role, tool_name, outcome, detail
FROM audit_log ORDER BY id DESC LIMIT %s
"""


class InMemoryAuditLog:
    """Keeps recent entries in memory, for tests and for when Postgres is unavailable."""

    def __init__(self, max_entries: int = 1000) -> None:
        """Bound the entries, so that a long outage cannot grow memory without limit."""
        self.entries: deque[dict[str, Any]] = deque(maxlen=max_entries)

    async def record(self, user: UserContext, tool_name: str, outcome: str, detail: str) -> bool:
        """Store one entry. Memory is what tests and an outage have, so this reports success."""
        self.entries.appendleft(
            {
                "created_at": datetime.now(UTC),
                "user_id": user.user_id,
                "role": user.role,
                "tool_name": tool_name,
                "outcome": outcome,
                "detail": detail,
            }
        )
        return True

    async def read_recent(self, entry_count: int) -> list[dict[str, Any]]:
        """Return the newest entries, newest first."""
        return list(self.entries)[:entry_count]


class PostgresAuditLog:
    """Stores entries in Postgres, beside the checkpoints."""

    def __init__(self, connection_pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]]):
        """Share the application's pool, so that a restart of Postgres is handled in one place."""
        self._connection_pool = connection_pool
        # Entries from an outage stay here, so that they can still be read.
        self._unsaved_entries = InMemoryAuditLog()

    async def setup(self) -> None:
        """Create the table if this is the first run."""
        async with self._connection_pool.connection() as connection:
            await connection.execute(CREATE_TABLE_STATEMENT)

    async def record(self, user: UserContext, tool_name: str, outcome: str, detail: str) -> bool:
        """Store one entry, keeping it in memory if Postgres cannot take it."""
        try:
            async with self._connection_pool.connection() as connection:
                await connection.execute(
                    INSERT_ENTRY_STATEMENT, (user.user_id, user.role, tool_name, outcome, detail)
                )
        except Exception as error:  # An audit outage must not take the assistant down.
            logger.warning("audit_write_failed", error=type(error).__name__, tool=tool_name)
            await self._unsaved_entries.record(user, tool_name, outcome, detail)
            return False
        return True

    async def read_recent(self, entry_count: int) -> list[dict[str, Any]]:
        """Return the newest entries, including any held in memory from an outage."""
        unsaved_entries = await self._unsaved_entries.read_recent(entry_count)
        async with self._connection_pool.connection() as connection:
            cursor = await connection.execute(SELECT_RECENT_STATEMENT, (entry_count,))
            rows = await cursor.fetchall()
        return [*unsaved_entries, *[dict(row) for row in rows]][:entry_count]

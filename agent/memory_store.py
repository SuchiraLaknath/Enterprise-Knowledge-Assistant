"""Long-term memory: facts an employee shared, kept between conversations.

Rule 8: this stores doc ids, never document text, and is namespaced by `user_id`, so that no user
can read another's memory. Facts come only from what the employee said, and pass the injection
checks, so a poisoned document cannot plant an instruction for a later conversation.

Facts are read by recency. Semantic search would need an embedding call on every turn, and a
person only ever has a handful of facts, so reading them all is both simpler and cheaper.
"""

import hashlib
from typing import Any, Protocol

import structlog
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger()

NAMESPACE_ROOT = "user_memory"
FACT_KEY_LENGTH = 16


class LongTermMemory(Protocol):
    """What the memory nodes need, so that tests can run with no Postgres."""

    async def read(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        """Return this user's newest memories."""
        ...

    async def write(self, user_id: str, fact: str, doc_ids: list[str]) -> None:
        """Store one fact for this user, with the documents it relates to."""
        ...


class PostgresLongTermMemory:
    """Stores memories in `AsyncPostgresStore`, beside the checkpoints."""

    def __init__(self, connection_pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]]):
        """Share the application's pool, as Postgres outages are handled in one place."""
        self._store = AsyncPostgresStore(connection_pool)

    async def setup(self) -> None:
        """Create the tables if this is the first run."""
        await self._store.setup()

    async def read(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        """Return this user's newest memories. Rule 8: the namespace holds only their entries."""
        items = await self._store.asearch((NAMESPACE_ROOT, user_id), limit=limit)
        return [item.value for item in items]

    async def write(self, user_id: str, fact: str, doc_ids: list[str]) -> None:
        """Store one fact. Its key comes from its text, so repeating it creates no duplicate."""
        fact_key = hashlib.sha256(fact.encode()).hexdigest()[:FACT_KEY_LENGTH]
        await self._store.aput(
            (NAMESPACE_ROOT, user_id),
            fact_key,
            {"fact": fact, "doc_ids": doc_ids},
            index=False,
        )

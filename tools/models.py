"""Typed records shared by the tool registry, the tools and the `tool_agent` node.

Tool arguments never hold a role, a user or an access level. Those travel in `ToolContext`, which
the node builds from the verified user, so that a model cannot influence them (rule 1).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from config.settings import SandboxLimits, ToolLimits
from retrieval.models import DocumentFilters, RetrievedChunk, SearchResult, UserContext

ToolStatus = Literal["ok", "denied", "error", "timeout"]
ToolRisk = Literal["read", "write"]


class ToolArguments(BaseModel):
    """Base for every args schema. Unknown fields fail, so an extra `role` argument is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolResult(BaseModel):
    """What came of one tool call. Failures are data, so the turn can carry on without it."""

    tool_name: str
    status: ToolStatus
    # The tool's output, for the response agent to read as data.
    data: Any = None
    # A short, safe explanation. It never holds an exception's own text.
    message: str = ""
    # Chunks a tool retrieved, which become citable evidence for this turn.
    evidence: list[RetrievedChunk] = []


class ToolSearcher(Protocol):
    """The search operation `knowledge_search` needs."""

    async def search(
        self,
        query_text: str,
        user: UserContext,
        document_filters: DocumentFilters | None = None,
        namespaces: tuple[str, ...] | None = None,
        use_reranker: bool = True,
        result_count: int | None = None,
    ) -> SearchResult:
        """Return the best chunks this user may read."""
        ...


class AuditRecorder(Protocol):
    """The audit operations the tools need, so that tests can supply a fake."""

    async def record(self, user: UserContext, tool_name: str, outcome: str, detail: str) -> None:
        """Store one audit entry."""
        ...

    async def read_recent(self, entry_count: int) -> list[dict[str, Any]]:
        """Return the newest audit entries."""
        ...


class McpCaller(Protocol):
    """The MCP operation the tools need, so that tests can supply a fake."""

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], user: UserContext, timeout_s: float
    ) -> Any:
        """Call one MCP tool as this user."""
        ...


@dataclass(frozen=True)
class ToolContext:
    """What a tool may use. `user` comes from the verified token, never from tool arguments."""

    user: UserContext
    document_search: ToolSearcher
    audit_log: AuditRecorder
    mcp_client: McpCaller
    sandbox_limits: SandboxLimits
    tool_limits: ToolLimits
    # Data from this turn that `python_analysis` may work on. It holds no document text.
    analysis_records: list[dict[str, Any]] = field(default_factory=list)


ToolHandler = Callable[[Any, ToolContext], Awaitable[ToolResult]]

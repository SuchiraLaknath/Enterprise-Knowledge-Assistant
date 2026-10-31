"""The services that graph nodes use, passed in at run time.

Keeping these out of `AgentState` means they are never written to a checkpoint, and that tests
can run the real graph with fakes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agent.llm_gateway import DraftResetHandler, GatewayResult, TextHandler, ToolCallRequest
from config.settings import ResearchLimits, SandboxLimits, ToolLimits
from retrieval.catalog import DocumentCatalog
from retrieval.models import DocumentFilters, SearchResult, UserContext
from tools.models import AuditRecorder, McpCaller
from tools.registry import ToolRegistry


class LlmClient(Protocol):
    """The LLM operations nodes use, implemented by `LlmGateway`."""

    async def generate_structured[OutputSchema: BaseModel](
        self,
        route_name: str,
        messages: list[BaseMessage],
        output_schema: type[OutputSchema],
    ) -> GatewayResult[OutputSchema]:
        """Return a validated object from the first usable model."""
        ...

    async def generate_tool_calls(
        self,
        route_name: str,
        messages: list[BaseMessage],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> GatewayResult[list[ToolCallRequest]]:
        """Return the tools a model chose, from those it was offered."""
        ...

    async def stream_text(
        self,
        route_name: str,
        messages: list[BaseMessage],
        on_text: TextHandler,
        on_draft_reset: DraftResetHandler,
    ) -> GatewayResult[str]:
        """Stream text from the first usable model."""
        ...


class DocumentSearcher(Protocol):
    """The search operation nodes use, implemented by `DocumentSearchService`."""

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

    async def fetch_chunks(
        self, chunk_ids_by_namespace: dict[str, list[str]], user: UserContext
    ) -> SearchResult:
        """Return the named chunks this user may read."""
        ...


@dataclass(frozen=True)
class AgentDependencies:
    """Everything the nodes need, created once at startup."""

    llm: LlmClient
    document_search: DocumentSearcher
    document_catalog: DocumentCatalog
    history_message_count: int
    research_limits: ResearchLimits
    sandbox_limits: SandboxLimits
    tool_limits: ToolLimits
    tool_registry: ToolRegistry
    audit_log: AuditRecorder
    mcp_client: McpCaller

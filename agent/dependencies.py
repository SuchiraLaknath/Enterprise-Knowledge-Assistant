"""The services that graph nodes use, passed in at run time.

Keeping these out of `AgentState` means they are never written to a checkpoint, and that tests
can run the real graph with fakes.
"""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agent.llm_gateway import DraftResetHandler, GatewayResult, TextHandler
from retrieval.models import DocumentFilters, SearchResult, UserContext


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
    ) -> SearchResult:
        """Return the best chunks this user may read."""
        ...


@dataclass(frozen=True)
class AgentDependencies:
    """Everything the nodes need, created once at startup."""

    llm: LlmClient
    document_search: DocumentSearcher
    history_message_count: int

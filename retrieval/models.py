"""Typed records and interfaces shared by the retrieval modules and the agents."""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class UserContext(BaseModel):
    """The verified identity of the caller.

    Only the API creates this, from a verified JWT. It is frozen so that no later step can
    change it, because the LLM must never influence what a user may access.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    role: str
    # At least one level is required, so that a malformed user cannot produce an empty filter.
    allowed_access_levels: tuple[str, ...] = Field(min_length=1)


class DocumentFilters(BaseModel):
    """Optional narrowing that a search may request.

    Access level is deliberately absent, and unknown fields are rejected, so that a tool
    argument trying to set one fails loudly rather than being ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_types: tuple[str, ...] = ()
    created_after_ts: int | None = None
    created_before_ts: int | None = None


class RetrievedChunk(BaseModel):
    """One search result, with everything needed to cite it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    section: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    created_ts: int
    source_uri: str
    text: str
    score: float


class RerankedPosition(BaseModel):
    """Where one input chunk belongs after reranking."""

    model_config = ConfigDict(frozen=True)

    input_index: int
    score: float


class SearchResult(BaseModel):
    """The outcome of a search, including how it was produced."""

    chunks: list[RetrievedChunk]
    # Names of the fallbacks used, so that the UI can show when an answer is degraded.
    degraded: list[str] = Field(default_factory=list)
    dense_weight: float | None = None


class SearchBackend(Protocol):
    """The vector search operations retrieval needs, so that tests can supply a fake."""

    async def embed_query(self, query_text: str) -> list[float]:
        """Turn a query into a dense vector."""
        ...

    async def query_namespace(
        self,
        namespace: str,
        dense_vector: list[float],
        sparse_vector: dict[str, list[Any]] | None,
        metadata_filter: dict[str, Any],
        result_count: int,
    ) -> list[RetrievedChunk]:
        """Return the best matches from one namespace."""
        ...

    async def rerank(
        self, query_text: str, document_texts: list[str], result_count: int
    ) -> list[RerankedPosition]:
        """Return the most relevant inputs, best first."""
        ...


class SparseEncoder(Protocol):
    """The BM25 operations retrieval needs, so that tests avoid downloading tokenizer data."""

    def encode_queries(self, texts: str) -> dict[str, list[Any]]:
        """Turn a query into a sparse vector."""
        ...

    def encode_documents(self, texts: str) -> dict[str, list[Any]]:
        """Turn a document into a sparse vector."""
        ...

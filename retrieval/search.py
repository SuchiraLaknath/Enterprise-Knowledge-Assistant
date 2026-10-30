"""The single entry point for document search.

This reads as the story of one search: try hybrid search unless Pinecone is known to be down,
fall back to local keyword search if needed, rerank when asked to, and confirm access last.
Each fallback that was used is named in the result, so that the UI can show it.
"""

import structlog
from langsmith import traceable

from config.settings import RetrievalLimits
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.filters import build_metadata_filter, keep_permitted_chunks
from retrieval.hybrid import hybrid_search
from retrieval.models import (
    DocumentFilters,
    RetrievedChunk,
    SearchBackend,
    SearchResult,
    SparseEncoder,
    UserContext,
)
from retrieval.rerank import rerank_chunks
from retrieval.tracing import keep_query_inputs

logger = structlog.get_logger()


class DocumentSearchService:
    """Combines the search paths. Created once and shared by all requests."""

    def __init__(
        self,
        backend: SearchBackend,
        sparse_encoder: SparseEncoder,
        local_keyword_index: LocalKeywordIndex,
        circuit_breaker: CircuitBreaker,
        limits: RetrievalLimits,
    ) -> None:
        """Keep the collaborators, so that tests can supply fakes."""
        self._backend = backend
        self._sparse_encoder = sparse_encoder
        self._local_keyword_index = local_keyword_index
        self._circuit_breaker = circuit_breaker
        self._limits = limits

    @traceable(run_type="retriever", name="document_search", process_inputs=keep_query_inputs)
    async def search(
        self,
        query_text: str,
        user: UserContext,
        document_filters: DocumentFilters | None = None,
        namespaces: tuple[str, ...] | None = None,
        use_reranker: bool = True,
    ) -> SearchResult:
        """Return the best chunks this user may read.

        Research passes `use_reranker=False`, because its many searches would otherwise use
        up the monthly rerank quota.
        """
        active_filters = document_filters or DocumentFilters()
        active_namespaces = namespaces or self._limits.namespaces
        degraded: list[str] = []
        dense_weight: float | None = None
        candidate_chunks: list[RetrievedChunk] | None = None

        # Step 1: try hybrid search, unless recent failures show that Pinecone is down.
        if self._circuit_breaker.allows_request():
            metadata_filter = build_metadata_filter(user, active_filters)
            try:
                outcome = await hybrid_search(
                    query_text,
                    active_namespaces,
                    metadata_filter,
                    self._backend,
                    self._sparse_encoder,
                    self._limits,
                )
            except Exception as error:  # Any Pinecone failure leads to the same fallback.
                self._circuit_breaker.record_failure()
                degraded.append("pinecone_unavailable")
                logger.warning("hybrid_search_failed", error=str(error))
            else:
                self._circuit_breaker.record_success()
                candidate_chunks = outcome.chunks
                dense_weight = outcome.dense_weight
                degraded.extend(
                    f"namespace_unavailable:{namespace}" for namespace in outcome.failed_namespaces
                )
        else:
            degraded.append("pinecone_circuit_open")

        # Step 2: fall back to local keyword search, which needs no network.
        if candidate_chunks is None:
            local_chunks = self._local_keyword_index.search(
                query_text,
                user,
                active_namespaces,
                active_filters,
                self._limits.final_result_count,
            )
            return SearchResult(
                chunks=keep_permitted_chunks(local_chunks, user),
                degraded=[*degraded, "local_keyword_search"],
            )

        # Step 3: reorder the short list, keeping hybrid order if reranking fails.
        final_chunks = candidate_chunks[: self._limits.final_result_count]
        if use_reranker:
            try:
                final_chunks = await rerank_chunks(
                    query_text,
                    candidate_chunks,
                    self._backend,
                    self._limits.final_result_count,
                    self._limits.rerank_timeout_s,
                )
            except Exception as error:  # Hybrid order is a usable answer, so continue with it.
                degraded.append("rerank_unavailable")
                logger.warning("rerank_failed", error=str(error))

        # Step 4: Rule 2, confirm access on the final results, whichever path produced them.
        return SearchResult(
            chunks=keep_permitted_chunks(final_chunks, user),
            degraded=degraded,
            dense_weight=dense_weight,
        )

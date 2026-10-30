"""The single entry point for document search.

`DocumentSearchService.search` reads as the story of one search: try hybrid search unless
Pinecone is known to be down, fall back to local keyword search if needed, rerank when asked
to, and confirm access last. Each fallback used is named in the result, so the UI can show it.
"""

from dataclasses import dataclass, field

import structlog
from langsmith import traceable

from config.settings import RetrievalLimits
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.filters import build_metadata_filter, keep_permitted_chunks
from retrieval.hybrid import HybridSearchOutcome, hybrid_search
from retrieval.models import (
    DegradedReason,
    DocumentFilters,
    RetrievedChunk,
    SearchBackend,
    SearchResult,
    SparseEncoder,
    UserContext,
)
from retrieval.rerank import rerank_chunks
from retrieval.tracing import RETRIEVER_TRACE_OPTIONS

logger = structlog.get_logger()


@dataclass(frozen=True)
class _HybridAttempt:
    """What happened when hybrid search was tried. `outcome` is None if it was unusable."""

    outcome: HybridSearchOutcome | None
    degraded: list[str] = field(default_factory=list)


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

    @traceable(name="document_search", **RETRIEVER_TRACE_OPTIONS)
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

        # Step 1: try hybrid search, unless recent failures show that Pinecone is down.
        hybrid_attempt = await self._try_hybrid_search(
            query_text, user, active_filters, active_namespaces
        )

        # Step 2: fall back to local keyword search, which needs no network.
        if hybrid_attempt.outcome is None:
            local_chunks = self._local_keyword_index.search(
                query_text,
                user,
                active_namespaces,
                active_filters,
                self._limits.final_result_count,
            )
            return SearchResult(
                chunks=keep_permitted_chunks(local_chunks, user),
                degraded=[*hybrid_attempt.degraded, DegradedReason.LOCAL_KEYWORD_SEARCH],
            )

        # Step 3: choose the final short list.
        final_chunks, rank_degraded = await self._rank_final_chunks(
            query_text, hybrid_attempt.outcome.chunks, use_reranker
        )

        # Step 4: Rule 2, confirm access on the final results, whichever path produced them.
        return SearchResult(
            chunks=keep_permitted_chunks(final_chunks, user),
            degraded=[*hybrid_attempt.degraded, *rank_degraded],
            dense_weight=hybrid_attempt.outcome.dense_weight,
        )

    async def _try_hybrid_search(
        self,
        query_text: str,
        user: UserContext,
        document_filters: DocumentFilters,
        namespaces: tuple[str, ...],
    ) -> _HybridAttempt:
        """Run hybrid search if the circuit breaker allows it, and report what happened."""
        if not self._circuit_breaker.allows_request():
            return _HybridAttempt(None, [DegradedReason.PINECONE_CIRCUIT_OPEN])

        metadata_filter = build_metadata_filter(user, document_filters)
        try:
            outcome = await hybrid_search(
                query_text,
                namespaces,
                metadata_filter,
                self._backend,
                self._sparse_encoder,
                self._limits,
            )
        except Exception as error:  # Any Pinecone failure leads to the same fallback.
            self._circuit_breaker.record_failure()
            logger.warning("hybrid_search_failed", error=str(error))
            return _HybridAttempt(None, [DegradedReason.PINECONE_UNAVAILABLE])

        self._circuit_breaker.record_success()
        unavailable_namespaces = [
            f"{DegradedReason.NAMESPACE_UNAVAILABLE}:{namespace}"
            for namespace in outcome.failed_namespaces
        ]
        return _HybridAttempt(outcome, unavailable_namespaces)

    async def _rank_final_chunks(
        self, query_text: str, candidate_chunks: list[RetrievedChunk], use_reranker: bool
    ) -> tuple[list[RetrievedChunk], list[str]]:
        """Return the final chunks, keeping hybrid order if reranking is off or fails."""
        result_count = self._limits.final_result_count
        hybrid_order_chunks = candidate_chunks[:result_count]
        if not use_reranker:
            return hybrid_order_chunks, []

        try:
            reranked_chunks = await rerank_chunks(
                query_text,
                candidate_chunks[: self._limits.rerank_candidate_count],
                self._backend,
                result_count,
                self._limits.rerank_timeout_s,
            )
        except Exception as error:  # Hybrid order is a usable answer, so continue with it.
            logger.warning("rerank_failed", error=str(error))
            return hybrid_order_chunks, [DegradedReason.RERANK_UNAVAILABLE]
        relevant_chunks = [
            chunk for chunk in reranked_chunks if chunk.score >= self._limits.minimum_rerank_score
        ]
        return relevant_chunks, []

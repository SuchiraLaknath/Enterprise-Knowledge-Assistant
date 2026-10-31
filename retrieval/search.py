"""The single entry point for document search.

`DocumentSearchService.search` reads as the story of one search: try hybrid search unless
Pinecone is known to be down, fall back to local keyword search if needed, rerank when asked
to, and confirm access last. Each fallback used is named in the result, so the UI can show it.
"""

import asyncio
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
        result_count: int | None = None,
    ) -> SearchResult:
        """Return the best chunks this user may read.

        Research passes `use_reranker=False`, because its many searches would otherwise use
        up the monthly rerank quota. It also asks for a larger `result_count`, because it
        surveys a topic rather than looking for the few best passages.
        """
        active_filters = document_filters or DocumentFilters()
        active_namespaces = namespaces or self._limits.namespaces
        final_result_count = result_count or self._limits.final_result_count

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
                final_result_count,
            )
            return SearchResult(
                chunks=keep_permitted_chunks(local_chunks, user),
                degraded=[*hybrid_attempt.degraded, DegradedReason.LOCAL_KEYWORD_SEARCH],
            )

        # Step 3: choose the final short list.
        final_chunks, rank_degraded = await self._rank_final_chunks(
            query_text, hybrid_attempt.outcome.chunks, use_reranker, final_result_count
        )

        # Step 4: Rule 2, confirm access on the final results, whichever path produced them.
        return SearchResult(
            chunks=keep_permitted_chunks(final_chunks, user),
            degraded=[*hybrid_attempt.degraded, *rank_degraded],
            dense_weight=hybrid_attempt.outcome.dense_weight,
        )

    @traceable(name="fetch_chunks", **RETRIEVER_TRACE_OPTIONS)
    async def fetch_chunks(
        self, chunk_ids_by_namespace: dict[str, list[str]], user: UserContext
    ) -> SearchResult:
        """Return the named chunks this user may read, falling back to the local copy.

        Rule 2: fetching by id bypasses the query filter, so access is checked on what comes
        back, whichever path produced it.
        """
        degraded: list[str] = []
        if self._circuit_breaker.allows_request():
            try:
                fetched_chunks = await self._fetch_from_pinecone(chunk_ids_by_namespace)
            except Exception as error:  # Any Pinecone failure leads to the same fallback.
                self._circuit_breaker.record_failure()
                logger.warning("fetch_chunks_failed", error=type(error).__name__)
                degraded.append(DegradedReason.PINECONE_UNAVAILABLE)
            else:
                self._circuit_breaker.record_success()
                return SearchResult(chunks=keep_permitted_chunks(fetched_chunks, user))
        else:
            degraded.append(DegradedReason.PINECONE_CIRCUIT_OPEN)

        all_chunk_ids = [
            chunk_id for chunk_ids in chunk_ids_by_namespace.values() for chunk_id in chunk_ids
        ]
        local_chunks = self._local_keyword_index.fetch_chunks(all_chunk_ids, user)
        return SearchResult(
            chunks=keep_permitted_chunks(local_chunks, user),
            degraded=[*degraded, DegradedReason.LOCAL_KEYWORD_SEARCH],
        )

    async def _fetch_from_pinecone(
        self, chunk_ids_by_namespace: dict[str, list[str]]
    ) -> list[RetrievedChunk]:
        """Fetch from each namespace at once, because one document set can span departments."""
        fetches = [
            asyncio.wait_for(
                self._backend.fetch_chunks(namespace, chunk_ids),
                timeout=self._limits.pinecone_timeout_s,
            )
            for namespace, chunk_ids in chunk_ids_by_namespace.items()
        ]
        chunks_by_namespace = await asyncio.gather(*fetches)
        return [chunk for namespace_chunks in chunks_by_namespace for chunk in namespace_chunks]

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
            logger.warning("hybrid_search_failed", error=type(error).__name__)
            return _HybridAttempt(None, [DegradedReason.PINECONE_UNAVAILABLE])

        self._circuit_breaker.record_success()
        unavailable_namespaces = [
            f"{DegradedReason.NAMESPACE_UNAVAILABLE}:{namespace}"
            for namespace in outcome.failed_namespaces
        ]
        return _HybridAttempt(outcome, unavailable_namespaces)

    async def _rank_final_chunks(
        self,
        query_text: str,
        candidate_chunks: list[RetrievedChunk],
        use_reranker: bool,
        result_count: int,
    ) -> tuple[list[RetrievedChunk], list[str]]:
        """Return the final chunks, keeping hybrid order if reranking is off or fails."""
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
            logger.warning("rerank_failed", error=type(error).__name__)
            return hybrid_order_chunks, [DegradedReason.RERANK_UNAVAILABLE]
        relevant_chunks = [
            chunk for chunk in reranked_chunks if chunk.score >= self._limits.minimum_rerank_score
        ]
        return relevant_chunks, []

"""Local keyword search, used when Pinecone is unavailable.

It reads the same documents from disk and scores them with the fitted BM25 statistics, so the
assistant can still answer with citations during an outage, with lower recall.

Rule 2: this path has no server-side filter, so it applies the access check itself.
"""

from dataclasses import asdict
from pathlib import Path

from langsmith import traceable

from config.settings import ChunkingLimits
from ingestion.chunk import DocumentChunk, chunk_document, load_documents
from retrieval.filters import is_access_allowed, matches_document_filters
from retrieval.models import DocumentFilters, RetrievedChunk, SparseEncoder, UserContext
from retrieval.tracing import RETRIEVER_TRACE_OPTIONS


class LocalKeywordIndex:
    """All chunks in memory, each with its BM25 document vector.

    A department name is the same as its Pinecone namespace, so `namespaces` selects
    departments here.
    """

    def __init__(self, chunks: list[DocumentChunk], sparse_encoder: SparseEncoder) -> None:
        """Encode every chunk once, so that searching only needs a score calculation."""
        self._sparse_encoder = sparse_encoder
        self._chunks_with_term_weights = [
            (chunk, self._encode_term_weights(chunk)) for chunk in chunks
        ]

    @classmethod
    def from_documents(
        cls,
        documents_directory: Path,
        sparse_encoder: SparseEncoder,
        chunking_limits: ChunkingLimits,
    ) -> "LocalKeywordIndex":
        """Build the index from the Markdown corpus on disk."""
        chunks = [
            chunk
            for document in load_documents(documents_directory)
            for chunk in chunk_document(document, chunking_limits)
        ]
        return cls(chunks, sparse_encoder)

    def _encode_term_weights(self, chunk: DocumentChunk) -> dict[int, float]:
        """Return a chunk's sparse vector as a term-to-weight lookup."""
        sparse_vector = self._sparse_encoder.encode_documents(chunk.text_for_embedding())
        return dict(zip(sparse_vector["indices"], sparse_vector["values"], strict=True))

    @traceable(name="local_keyword_search", **RETRIEVER_TRACE_OPTIONS)
    def search(
        self,
        query_text: str,
        user: UserContext,
        namespaces: tuple[str, ...],
        document_filters: DocumentFilters,
        result_count: int,
    ) -> list[RetrievedChunk]:
        """Return the best chunks this user may read."""
        sparse_query = self._sparse_encoder.encode_queries(query_text)
        query_term_weights = dict(zip(sparse_query["indices"], sparse_query["values"], strict=True))

        scored_chunks: list[RetrievedChunk] = []
        for chunk, term_weights in self._chunks_with_term_weights:
            # Rule 2: the access check runs first, so a blocked chunk is never even scored.
            if not is_access_allowed(user, chunk.access_level):
                continue
            if chunk.department not in namespaces:
                continue
            if not matches_document_filters(chunk, document_filters):
                continue

            score = sum(
                query_weight * term_weights.get(term, 0.0)
                for term, query_weight in query_term_weights.items()
            )
            # A score of 0 means that no query term occurs in the chunk.
            if score > 0:
                scored_chunks.append(RetrievedChunk(score=score, **asdict(chunk)))

        scored_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored_chunks[:result_count]

    def fetch_chunks(self, chunk_ids: list[str], user: UserContext) -> list[RetrievedChunk]:
        """Return the named chunks this user may read, for when Pinecone cannot fetch them.

        A fetch has no query, so there is no relevance score, and each chunk scores zero.
        """
        requested_chunk_ids = set(chunk_ids)
        fetched_chunks = []
        for chunk, _ in self._chunks_with_term_weights:
            # Rule 2: fetching by id has no query filter, so access is checked here.
            if chunk.chunk_id in requested_chunk_ids and is_access_allowed(
                user, chunk.access_level
            ):
                fetched_chunks.append(RetrievedChunk(score=0.0, **asdict(chunk)))
        return fetched_chunks

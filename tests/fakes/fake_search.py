"""In-memory stand-ins for Pinecone and BM25, so that tests never touch the network."""

import zlib
from typing import Any

from retrieval.models import RerankedPosition, RetrievedChunk


def make_chunk(
    chunk_id: str = "DOC-1#c0",
    access_level: str = "internal",
    department: str = "payments",
    score: float = 0.5,
    text: str = "payment gateway timeout",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        title="Test title",
        section="Summary",
        department=department,
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=1768435200,
        source_uri=f"docs://{department}/{chunk_id}",
        text=text,
        score=score,
    )


class FakeSparseEncoder:
    """Hashes each word, so that shared words produce a positive score."""

    def _encode(self, text: str) -> dict[str, list[Any]]:
        unique_terms = sorted({zlib.crc32(word.encode()) for word in text.lower().split()})
        return {"indices": unique_terms, "values": [1.0] * len(unique_terms)}

    def encode_queries(self, texts: str) -> dict[str, list[Any]]:
        return self._encode(texts)

    def encode_documents(self, texts: str) -> dict[str, list[Any]]:
        return self._encode(texts)


class FakeSearchBackend:
    """Returns preset results per namespace, can be told to fail, and records its calls."""

    def __init__(
        self,
        results_by_namespace: dict[str, list[RetrievedChunk]] | None = None,
        failing_namespaces: tuple[str, ...] = (),
        embedding_fails: bool = False,
        rerank_fails: bool = False,
        fetch_fails: bool = False,
    ) -> None:
        self.results_by_namespace = results_by_namespace or {}
        self.failing_namespaces = failing_namespaces
        self.embedding_fails = embedding_fails
        self.rerank_fails = rerank_fails
        self.fetch_fails = fetch_fails
        self.queried_namespaces: list[str] = []
        self.received_filters: list[dict[str, Any]] = []
        self.received_sparse_vectors: list[dict[str, list[Any]] | None] = []
        self.received_dense_vectors: list[list[float]] = []
        self.rerank_call_count = 0
        self.fetched_chunk_ids: list[str] = []

    async def embed_query(self, query_text: str) -> list[float]:
        if self.embedding_fails:
            raise ConnectionError("Simulated embedding failure")
        return [1.0, 1.0]

    async def query_namespace(
        self,
        namespace: str,
        dense_vector: list[float],
        sparse_vector: dict[str, list[Any]] | None,
        metadata_filter: dict[str, Any],
        result_count: int,
    ) -> list[RetrievedChunk]:
        self.queried_namespaces.append(namespace)
        self.received_filters.append(metadata_filter)
        self.received_sparse_vectors.append(sparse_vector)
        self.received_dense_vectors.append(dense_vector)
        if namespace in self.failing_namespaces:
            raise ConnectionError(f"Simulated failure in {namespace}")
        return self.results_by_namespace.get(namespace, [])

    async def fetch_chunks(self, namespace: str, chunk_ids: list[str]) -> list[RetrievedChunk]:
        self.fetched_chunk_ids.extend(chunk_ids)
        if self.fetch_fails:
            raise ConnectionError("Simulated fetch failure")
        # Like Pinecone, an id that does not exist is simply missing, and there is no filter.
        return [
            chunk.model_copy(update={"score": 0.0})
            for chunk in self.results_by_namespace.get(namespace, [])
            if chunk.chunk_id in chunk_ids
        ]

    async def rerank(
        self, query_text: str, document_texts: list[str], result_count: int
    ) -> list[RerankedPosition]:
        self.rerank_call_count += 1
        if self.rerank_fails:
            raise ConnectionError("Simulated rerank failure")
        # Reverses the input, so that a test can see that reranking changed the order.
        reversed_indices = list(reversed(range(len(document_texts))))[:result_count]
        return [
            RerankedPosition(input_index=input_index, score=0.9 - position / 100)
            for position, input_index in enumerate(reversed_indices)
        ]

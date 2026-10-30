"""Hybrid search: combine dense and sparse relevance, across namespaces, in parallel."""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nltk
import structlog
from langsmith import traceable
from pinecone_text.sparse import BM25Encoder

from config.settings import RetrievalLimits
from retrieval.models import RetrievedChunk, SearchBackend, SparseEncoder
from retrieval.tracing import RETRIEVER_TRACE_OPTIONS

logger = structlog.get_logger()

# Tokenizer data that BM25Encoder would otherwise download when it is created.
REQUIRED_TOKENIZER_DATA = ("tokenizers/punkt_tab", "corpora/stopwords")

# Matches identifiers such as ERR-PAY-5041, INC-2026-014, RB-003 and ISO8583.
IDENTIFIER_PATTERN = re.compile(r"\b[a-z]{2,}(?:-[a-z0-9]+)*-?\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class HybridSearchOutcome:
    """Merged results, plus what the caller needs in order to report on them."""

    chunks: list[RetrievedChunk]
    dense_weight: float
    failed_namespaces: list[str]


def _confirm_tokenizer_data_is_installed() -> None:
    """Fail at startup if data is missing, because Rule 6 forbids a download at runtime."""
    for data_path in REQUIRED_TOKENIZER_DATA:
        try:
            nltk.data.find(data_path)
        except LookupError as error:
            raise RuntimeError(
                f"Missing tokenizer data {data_path}. Install it before starting: "
                "uv run python -m nltk.downloader punkt_tab stopwords"
            ) from error


def load_bm25_encoder(parameters_path: Path) -> BM25Encoder:
    """Load the word statistics fitted on this corpus during indexing."""
    _confirm_tokenizer_data_is_installed()
    encoder = BM25Encoder()
    encoder.load(str(parameters_path))
    return encoder


def choose_dense_weight(query_text: str, limits: RetrievalLimits) -> float:
    """Favour exact keyword matching when a query names an id or an error code.

    Embeddings capture meaning, so they blur the difference between ERR-PAY-5041 and
    ERR-PAY-5032. BM25 treats those as distinct terms.
    """
    if IDENTIFIER_PATTERN.search(query_text):
        return limits.keyword_query_dense_weight
    return limits.default_dense_weight


def scale_hybrid_vectors(
    dense_vector: list[float],
    sparse_vector: dict[str, list[Any]],
    dense_weight: float,
) -> tuple[list[float], dict[str, list[Any]]]:
    """Weight both vectors, because a dotproduct index sums their scores directly."""
    sparse_weight = 1 - dense_weight
    scaled_dense_vector = [value * dense_weight for value in dense_vector]
    scaled_sparse_vector = {
        "indices": sparse_vector["indices"],
        "values": [value * sparse_weight for value in sparse_vector["values"]],
    }
    return scaled_dense_vector, scaled_sparse_vector


@traceable(name="hybrid_search", **RETRIEVER_TRACE_OPTIONS)
async def hybrid_search(
    query_text: str,
    namespaces: tuple[str, ...],
    metadata_filter: dict[str, Any],
    backend: SearchBackend,
    sparse_encoder: SparseEncoder,
    limits: RetrievalLimits,
) -> HybridSearchOutcome:
    """Search each namespace in parallel, and merge the results by score.

    One failed namespace leaves the others usable, so this raises only if all of them fail.
    """
    # Step 1: encode the query both ways, and weight the vectors.
    dense_vector = await asyncio.wait_for(
        backend.embed_query(query_text), timeout=limits.pinecone_timeout_s
    )
    sparse_vector = sparse_encoder.encode_queries(query_text)
    dense_weight = choose_dense_weight(query_text, limits)
    scaled_dense_vector, scaled_sparse_vector = scale_hybrid_vectors(
        dense_vector, sparse_vector, dense_weight
    )
    # A query of only stop words has no sparse terms, and Pinecone rejects an empty vector.
    sparse_vector_to_send = scaled_sparse_vector if scaled_sparse_vector["indices"] else None

    # Step 2: query every namespace at once, keeping any failures as values.
    namespace_queries = [
        asyncio.wait_for(
            backend.query_namespace(
                namespace,
                scaled_dense_vector,
                sparse_vector_to_send,
                metadata_filter,
                limits.candidates_per_namespace,
            ),
            timeout=limits.pinecone_timeout_s,
        )
        for namespace in namespaces
    ]
    namespace_results = await asyncio.gather(*namespace_queries, return_exceptions=True)

    # Step 3: separate successes from failures.
    merged_chunks: list[RetrievedChunk] = []
    failed_namespaces: list[str] = []
    last_failure: BaseException | None = None
    for namespace, namespace_result in zip(namespaces, namespace_results, strict=True):
        # gather returns failures as values, typed BaseException because of cancellation.
        if isinstance(namespace_result, BaseException):
            failed_namespaces.append(namespace)
            last_failure = namespace_result
            logger.warning(
                "namespace_query_failed", namespace=namespace, error=str(namespace_result)
            )
        else:
            merged_chunks.extend(namespace_result)

    # With nothing to merge, re-raise so that the caller can use the local fallback.
    if last_failure is not None and len(failed_namespaces) == len(namespaces):
        raise last_failure

    # Step 4: rank across departments. Scores are comparable, as they share one scaling.
    merged_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    return HybridSearchOutcome(merged_chunks, dense_weight, failed_namespaces)

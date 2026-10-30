"""Second-stage ranking with a cross-encoder model.

Hybrid scores are quick but approximate. A reranker reads the query and each chunk together,
so it is more accurate but too slow to run over a whole index. It therefore reorders only the
short list that hybrid search returns.
"""

import asyncio

import structlog
from langsmith import traceable

from ingestion.chunk import build_text_with_context
from retrieval.models import RetrievedChunk, SearchBackend
from retrieval.tracing import keep_query_inputs

logger = structlog.get_logger()


@traceable(run_type="retriever", name="rerank_chunks", process_inputs=keep_query_inputs)
async def rerank_chunks(
    query_text: str,
    chunks: list[RetrievedChunk],
    backend: SearchBackend,
    result_count: int,
    timeout_seconds: float,
) -> list[RetrievedChunk]:
    """Return the most relevant chunks, best first, with their reranker scores."""
    if not chunks:
        return []

    document_texts = [
        build_text_with_context(chunk.title, chunk.section, chunk.text) for chunk in chunks
    ]
    reranked_positions = await asyncio.wait_for(
        backend.rerank(query_text, document_texts, result_count), timeout=timeout_seconds
    )
    return [
        chunks[position.input_index].model_copy(update={"score": position.score})
        for position in reranked_positions
    ]

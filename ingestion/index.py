"""Chunk the corpus, encode it, and load it into Pinecone.

This is an offline batch job, not part of the request path, so it uses the synchronous client.
It also writes the two files the API needs at query time: the fitted BM25 parameters and the
document catalog. Each run replaces the whole index, because a record left over from an
earlier run would keep the old access level of a document that has since been reclassified.

Run with: uv run python -m ingestion.index
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import structlog
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from config.settings import (
    BM25_PARAMETERS_PATH,
    DOCUMENT_CATALOG_PATH,
    DOCUMENTS_DIRECTORY,
    get_environment_settings,
    get_limits,
    get_model_names,
)
from ingestion.chunk import DocumentChunk, chunk_document, load_documents

logger = structlog.get_logger()


def split_into_batches[BatchItem](
    items: Sequence[BatchItem], batch_size: int
) -> Iterator[Sequence[BatchItem]]:
    """Yield consecutive groups of at most `batch_size` items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_document_catalog(chunks: Sequence[DocumentChunk]) -> list[dict[str, Any]]:
    """Summarise each document, so the research planner can survey without reading any text."""
    catalog_by_doc_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        catalog_entry = catalog_by_doc_id.setdefault(
            chunk.doc_id,
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "department": chunk.department,
                "document_type": chunk.document_type,
                "access_level": chunk.access_level,
                "created_date": chunk.created_date,
                "created_ts": chunk.created_ts,
                "source_uri": chunk.source_uri,
                "chunk_count": 0,
            },
        )
        catalog_entry["chunk_count"] += 1
    return list(catalog_by_doc_id.values())


def build_vector_record(
    chunk: DocumentChunk,
    dense_values: list[float],
    sparse_vector: dict[str, list[Any]],
) -> dict[str, Any]:
    """Combine a chunk's two vectors and its metadata into one Pinecone record."""
    metadata = asdict(chunk)
    # The chunk id is already the record id, so storing it again would only waste space.
    del metadata["chunk_id"]
    return {
        "id": chunk.chunk_id,
        "values": dense_values,
        "sparse_values": sparse_vector,
        "metadata": metadata,
    }


def fit_bm25_encoder(chunks: Sequence[DocumentChunk], parameters_path: Path) -> BM25Encoder:
    """Learn word statistics from this corpus, and save them for use at query time."""
    encoder = BM25Encoder()
    encoder.fit([chunk.text_for_embedding() for chunk in chunks])
    parameters_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(parameters_path))
    return encoder


def ensure_index_exists(pinecone_client: Pinecone, index_name: str) -> None:
    """Create the hybrid index when it is missing."""
    if pinecone_client.has_index(index_name):
        logger.info("index_already_exists", index_name=index_name)
        return

    indexing_limits = get_limits().indexing
    # Only the dotproduct metric supports a dense and a sparse vector on the same record.
    pinecone_client.create_index(
        name=index_name,
        dimension=get_model_names().embedding_dimension,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=indexing_limits.cloud, region=indexing_limits.region),
    )
    logger.info("index_created", index_name=index_name)


def embed_chunks(pinecone_client: Pinecone, chunks: Sequence[DocumentChunk]) -> list[list[float]]:
    """Create one dense vector per chunk, in batches the embedding API accepts."""
    model_name = get_model_names().embedding_model
    batch_size = get_limits().indexing.embedding_batch_size
    dense_vectors: list[list[float]] = []

    for chunk_batch in split_into_batches(chunks, batch_size):
        embeddings = pinecone_client.inference.embed(
            model=model_name,
            inputs=[chunk.text_for_embedding() for chunk in chunk_batch],
            parameters={"input_type": "passage", "truncate": "END"},
        )
        # The SDK types cover sparse models too, but this model always returns a dense list.
        dense_vectors.extend(cast(list[float], embedding.values) for embedding in embeddings.data)
        logger.info("chunks_embedded", total_so_far=len(dense_vectors))
    return dense_vectors


def clear_existing_records(index: Any) -> None:
    """Delete every record, in every namespace, including any no longer in the config.

    Rule 2: upserts only overwrite matching ids. If a reclassified document produces fewer
    chunks, or moves department, its old records would stay searchable at the old level.
    """
    for namespace in index.describe_index_stats().namespaces:
        index.delete(delete_all=True, namespace=namespace)
        logger.info("namespace_cleared", namespace=namespace)


def upsert_records(
    pinecone_client: Pinecone,
    index_name: str,
    chunks: Sequence[DocumentChunk],
    dense_vectors: Sequence[list[float]],
    encoder: BM25Encoder,
) -> None:
    """Write records to the namespace of each chunk's department."""
    # Step 1: build the records, grouped by department, which is also the namespace.
    records_by_namespace: dict[str, list[dict[str, Any]]] = {}
    for chunk, dense_values in zip(chunks, dense_vectors, strict=True):
        sparse_vector = encoder.encode_documents(chunk.text_for_embedding())
        record = build_vector_record(chunk, dense_values, sparse_vector)
        records_by_namespace.setdefault(chunk.department, []).append(record)

    # Step 2: clear the old records only now, so that an encoding error leaves them intact.
    index = pinecone_client.Index(index_name)
    clear_existing_records(index)

    # Step 3: send each namespace in batches.
    batch_size = get_limits().indexing.upsert_batch_size
    for namespace, records in records_by_namespace.items():
        for record_batch in split_into_batches(records, batch_size):
            index.upsert(vectors=list(record_batch), namespace=namespace)
        logger.info("namespace_upserted", namespace=namespace, record_count=len(records))


def run_indexing() -> None:
    """Run the whole pipeline, from Markdown files to a searchable index."""
    settings = get_environment_settings()
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    # Step 1: read and chunk the corpus.
    documents = load_documents(DOCUMENTS_DIRECTORY)
    chunking_limits = get_limits().chunking
    chunks = [
        chunk for document in documents for chunk in chunk_document(document, chunking_limits)
    ]
    logger.info("corpus_chunked", document_count=len(documents), chunk_count=len(chunks))

    # Step 2: write the query-time files first, so that a failed upload still leaves the
    # local fallback usable.
    encoder = fit_bm25_encoder(chunks, BM25_PARAMETERS_PATH)
    catalog = build_document_catalog(chunks)
    DOCUMENT_CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    # Step 3: encode the chunks and load them into Pinecone.
    ensure_index_exists(pinecone_client, settings.pinecone_index)
    dense_vectors = embed_chunks(pinecone_client, chunks)
    upsert_records(pinecone_client, settings.pinecone_index, chunks, dense_vectors, encoder)
    logger.info("indexing_finished", record_count=len(chunks))


if __name__ == "__main__":
    run_indexing()

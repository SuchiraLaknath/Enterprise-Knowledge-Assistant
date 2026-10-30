from types import SimpleNamespace

from ingestion.chunk import DocumentChunk
from ingestion.index import (
    build_document_catalog,
    build_vector_record,
    clear_existing_records,
    split_into_batches,
    upsert_records,
)
from tests.fakes.fake_search import FakeSparseEncoder


def make_chunk(doc_id: str = "DOC-1", chunk_number: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}#c{chunk_number}",
        doc_id=doc_id,
        title="Title",
        section="Summary",
        department="payments",
        document_type="incident",
        access_level="internal",
        created_date="2026-01-15",
        created_ts=1768435200,
        source_uri=f"docs://payments/{doc_id}",
        text="Some chunk text",
    )


def test_batches_never_exceed_the_batch_size_and_keep_every_item():
    batches = list(split_into_batches(list(range(205)), 96))

    assert [len(batch) for batch in batches] == [96, 96, 13]
    assert [item for batch in batches for item in batch] == list(range(205))


def test_catalog_has_one_entry_per_document_with_its_chunk_count():
    chunks = [make_chunk("DOC-1", 0), make_chunk("DOC-1", 1), make_chunk("DOC-2", 0)]

    catalog = build_document_catalog(chunks)

    assert [(entry["doc_id"], entry["chunk_count"]) for entry in catalog] == [
        ("DOC-1", 2),
        ("DOC-2", 1),
    ]


def test_catalog_holds_no_document_text():
    catalog_entry = build_document_catalog([make_chunk()])[0]

    assert "text" not in catalog_entry


def test_vector_record_holds_both_vectors_and_all_required_metadata():
    sparse_vector = {"indices": [7], "values": [0.5]}

    record = build_vector_record(make_chunk(), [0.1, 0.2], sparse_vector)

    assert record["id"] == "DOC-1#c0"
    assert record["values"] == [0.1, 0.2]
    assert record["sparse_values"] == sparse_vector
    assert set(record["metadata"]) == {
        "doc_id",
        "title",
        "section",
        "department",
        "document_type",
        "access_level",
        "created_date",
        "created_ts",
        "source_uri",
        "text",
    }


class RecordingIndex:
    def __init__(self, namespaces: list[str]) -> None:
        self.namespaces = namespaces
        self.deleted_namespaces: list[str] = []
        self.call_order: list[str] = []

    def describe_index_stats(self):
        return SimpleNamespace(namespaces={name: object() for name in self.namespaces})

    def delete(self, delete_all: bool, namespace: str) -> None:
        assert delete_all
        self.deleted_namespaces.append(namespace)
        self.call_order.append("delete")

    def upsert(self, vectors: list, namespace: str) -> None:
        self.call_order.append("upsert")


def test_clearing_covers_every_existing_namespace_even_one_no_longer_configured():
    # A document that moved department must not stay readable in its old namespace.
    index = RecordingIndex(["payments", "retired-department"])

    clear_existing_records(index)

    assert index.deleted_namespaces == ["payments", "retired-department"]


def test_upserting_clears_old_records_first():
    # Without this, a reclassified document with fewer chunks keeps its old chunks searchable.
    index = RecordingIndex(["payments"])
    pinecone_client = SimpleNamespace(Index=lambda index_name: index)

    upsert_records(pinecone_client, "test-index", [make_chunk()], [[0.1]], FakeSparseEncoder())

    assert index.call_order == ["delete", "upsert"]

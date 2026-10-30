from ingestion.chunk import DocumentChunk
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.models import DocumentFilters, UserContext
from tests.fakes.fake_search import FakeSparseEncoder

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ALL_NAMESPACES = ("payments", "platform")


def make_chunk(doc_id: str, access_level: str, department: str = "payments") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}#c0",
        doc_id=doc_id,
        title="Title",
        section="Summary",
        department=department,
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=1000,
        source_uri=f"docs://{department}/{doc_id}",
        text="payment outage report",
    )


def search(chunks, namespaces=ALL_NAMESPACES):
    index = LocalKeywordIndex(chunks, FakeSparseEncoder())
    return index.search("payment outage", VIEWER, namespaces, DocumentFilters(), 10)


def test_index_itself_excludes_chunks_above_the_users_level():
    # Rule 2: tested on the index alone, so the later check in search.py cannot hide a bug here.
    results = search(
        [
            make_chunk("OPEN", "internal"),
            make_chunk("CONFIDENTIAL", "confidential"),
            make_chunk("RESTRICTED", "restricted"),
        ]
    )

    assert [chunk.doc_id for chunk in results] == ["OPEN"]


def test_search_stays_inside_the_requested_namespaces():
    results = search(
        [make_chunk("PAY", "internal"), make_chunk("PLAT", "internal", department="platform")],
        namespaces=("platform",),
    )

    assert [chunk.doc_id for chunk in results] == ["PLAT"]


def test_a_chunk_sharing_no_terms_is_not_a_match():
    index = LocalKeywordIndex([make_chunk("DOC", "internal")], FakeSparseEncoder())

    assert index.search("zebra", VIEWER, ALL_NAMESPACES, DocumentFilters(), 10) == []

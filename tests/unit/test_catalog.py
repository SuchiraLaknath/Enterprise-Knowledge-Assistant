from config.settings import DOCUMENT_CATALOG_PATH
from retrieval.catalog import CatalogEntry, DocumentCatalog
from retrieval.models import DocumentFilters, UserContext

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)


def make_entry(doc_id, access_level="internal", department="payments", created_ts=1000):
    return CatalogEntry(
        doc_id=doc_id,
        title=f"Title of {doc_id}",
        department=department,
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=created_ts,
        source_uri=f"docs://{department}/{doc_id}",
        chunk_count=2,
    )


def test_a_document_above_the_users_level_is_not_listed():
    # Rule 2: a title can reveal an incident even when its text stays hidden.
    catalog = DocumentCatalog([make_entry("OPEN"), make_entry("SECRET", "confidential")])

    viewer_ids = [entry.doc_id for entry in catalog.list_documents(VIEWER)]
    analyst_ids = {entry.doc_id for entry in catalog.list_documents(ANALYST)}

    assert viewer_ids == ["OPEN"]
    assert analyst_ids == {"OPEN", "SECRET"}


def test_counts_never_reveal_a_hidden_document():
    # Rule 2: "3 documents" would tell a viewer that one more exists than they can see.
    catalog = DocumentCatalog([make_entry("OPEN"), make_entry("SECRET", "restricted")])

    summary = catalog.summarize(VIEWER)

    assert summary.document_count == 1
    assert summary.chunk_count == 2
    assert summary.documents_by_department == {"payments": 1}


def test_listing_narrows_by_department_and_date_newest_first():
    catalog = DocumentCatalog(
        [
            make_entry("OLD", created_ts=100),
            make_entry("NEW", created_ts=900),
            make_entry("MIDDLE", created_ts=500),
            make_entry("OTHER", department="security", created_ts=900),
        ]
    )

    entries = catalog.list_documents(
        VIEWER, DocumentFilters(created_after_ts=200), namespaces=("payments",)
    )

    assert [entry.doc_id for entry in entries] == ["NEW", "MIDDLE"]


def test_chunk_ids_follow_the_ingestion_format():
    assert make_entry("INC-1").chunk_ids() == ["INC-1#c0", "INC-1#c1"]


def test_the_real_catalog_loads():
    catalog = DocumentCatalog.from_file(DOCUMENT_CATALOG_PATH)

    assert catalog.summarize(VIEWER).document_count > 0

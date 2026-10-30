import pytest
from pydantic import ValidationError

from retrieval.filters import build_metadata_filter, is_access_allowed, keep_permitted_chunks
from retrieval.models import DocumentFilters, UserContext
from tests.fakes.fake_search import make_chunk

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))


def test_filter_always_contains_the_users_access_levels():
    metadata_filter = build_metadata_filter(VIEWER, DocumentFilters())

    assert metadata_filter == {"access_level": {"$in": ["public", "internal"]}}


def test_narrowing_is_added_without_changing_the_access_clause():
    document_filters = DocumentFilters(
        document_types=("incident",), created_after_ts=100, created_before_ts=200
    )

    metadata_filter = build_metadata_filter(VIEWER, document_filters)

    assert metadata_filter == {
        "access_level": {"$in": ["public", "internal"]},
        "document_type": {"$in": ["incident"]},
        "created_ts": {"$gte": 100, "$lte": 200},
    }


def test_document_filters_reject_an_access_level_argument():
    # Tool abuse: an LLM tool argument tries to widen access to restricted documents.
    with pytest.raises(ValidationError):
        DocumentFilters(access_level="restricted")


def test_user_context_cannot_be_changed_after_creation():
    with pytest.raises(ValidationError):
        VIEWER.allowed_access_levels = ("restricted",)


def test_user_context_requires_at_least_one_access_level():
    with pytest.raises(ValidationError):
        UserContext(user_id="broken", role="viewer", allowed_access_levels=())


def test_access_check_allows_and_denies_correctly():
    assert is_access_allowed(VIEWER, "internal")
    assert not is_access_allowed(VIEWER, "confidential")


def test_keep_permitted_chunks_drops_chunks_above_the_users_level():
    chunks = [
        make_chunk("OPEN#c0", access_level="internal"),
        make_chunk("SECRET#c0", access_level="restricted"),
    ]

    permitted_chunks = keep_permitted_chunks(chunks, VIEWER)

    assert [chunk.chunk_id for chunk in permitted_chunks] == ["OPEN#c0"]

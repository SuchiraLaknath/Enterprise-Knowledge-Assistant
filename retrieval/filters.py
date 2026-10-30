"""Build metadata filters. This is the one place that turns a user into data access rules.

Rule 1: access always comes from the verified `UserContext`. `DocumentFilters` has no
access-level field, so even a value chosen by the LLM cannot widen what a user can read.
"""

from typing import Any

from retrieval.models import DocumentFilters, RetrievedChunk, UserContext


def build_metadata_filter(user: UserContext, document_filters: DocumentFilters) -> dict[str, Any]:
    """Return the Pinecone filter for a search: the access rule plus any narrowing."""
    # Rule 1: the access clause is always present, and it comes only from the verified user.
    metadata_filter: dict[str, Any] = {"access_level": {"$in": list(user.allowed_access_levels)}}

    if document_filters.document_types:
        metadata_filter["document_type"] = {"$in": list(document_filters.document_types)}

    # Pinecone range filters need numbers, which is why documents store created_ts.
    created_ts_range: dict[str, int] = {}
    if document_filters.created_after_ts is not None:
        created_ts_range["$gte"] = document_filters.created_after_ts
    if document_filters.created_before_ts is not None:
        created_ts_range["$lte"] = document_filters.created_before_ts
    if created_ts_range:
        metadata_filter["created_ts"] = created_ts_range

    return metadata_filter


def is_access_allowed(user: UserContext, access_level: str) -> bool:
    """Return True if the user may read a document at this access level."""
    return access_level in user.allowed_access_levels


def keep_permitted_chunks(chunks: list[RetrievedChunk], user: UserContext) -> list[RetrievedChunk]:
    """Drop any chunk above the user's access level.

    Rule 2: this second check protects read paths that have no query filter, such as fetching
    by id and the local fallback, and it catches a mistake in any single path.
    """
    return [chunk for chunk in chunks if is_access_allowed(user, chunk.access_level)]

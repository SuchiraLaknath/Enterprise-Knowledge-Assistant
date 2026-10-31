"""The document catalog: what exists, without any document text.

Ingestion writes `data/doc_catalog.json`. RLM research reads it to survey a collection, such as
"all payment incidents from the last year", which a search limited to the best matches cannot
list. It holds metadata only, so an LLM can see the shape of a collection without loading it.

Rule 2: every read goes through `is_access_allowed`, because a document's title or its existence
can itself be sensitive.
"""

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from retrieval.filters import is_access_allowed, matches_document_filters
from retrieval.models import DocumentFilters, UserContext


class CatalogEntry(BaseModel):
    """One document, as ingestion recorded it."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    created_ts: int
    source_uri: str
    chunk_count: int

    def chunk_ids(self) -> list[str]:
        """Return the id of every chunk, which ingestion numbers from zero."""
        return [f"{self.doc_id}#c{chunk_number}" for chunk_number in range(self.chunk_count)]


class CatalogSummary(BaseModel):
    """Counts of the documents one user may read, which is all the research planner sees."""

    document_count: int
    chunk_count: int
    documents_by_department: dict[str, int]
    documents_by_type: dict[str, int]
    oldest_date: str | None
    newest_date: str | None


class DocumentCatalog:
    """All catalog entries, loaded once at startup."""

    def __init__(self, entries: list[CatalogEntry]) -> None:
        """Keep the entries, so that tests can supply their own."""
        self._entries = entries

    @classmethod
    def from_file(cls, catalog_path: Path) -> "DocumentCatalog":
        """Load the catalog that ingestion wrote."""
        with catalog_path.open(encoding="utf-8") as catalog_file:
            raw_entries = json.load(catalog_file)
        return cls([CatalogEntry.model_validate(raw_entry) for raw_entry in raw_entries])

    def list_documents(
        self,
        user: UserContext,
        document_filters: DocumentFilters | None = None,
        namespaces: tuple[str, ...] | None = None,
    ) -> list[CatalogEntry]:
        """Return the documents this user may read, newest first."""
        active_filters = document_filters or DocumentFilters()
        permitted_entries = []
        for entry in self._entries:
            # Rule 2: the access check runs first, so a blocked document is never considered.
            if not is_access_allowed(user, entry.access_level):
                continue
            if namespaces and entry.department not in namespaces:
                continue
            if matches_document_filters(entry, active_filters):
                permitted_entries.append(entry)
        permitted_entries.sort(key=lambda entry: entry.created_ts, reverse=True)
        return permitted_entries

    def summarize(self, user: UserContext) -> CatalogSummary:
        """Describe what this user may read, so that no count reveals a hidden document."""
        permitted_entries = self.list_documents(user)
        created_dates = sorted(entry.created_date for entry in permitted_entries)
        return CatalogSummary(
            document_count=len(permitted_entries),
            chunk_count=sum(entry.chunk_count for entry in permitted_entries),
            documents_by_department=dict(Counter(entry.department for entry in permitted_entries)),
            documents_by_type=dict(Counter(entry.document_type for entry in permitted_entries)),
            oldest_date=created_dates[0] if created_dates else None,
            newest_date=created_dates[-1] if created_dates else None,
        )

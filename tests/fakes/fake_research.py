"""Stand-ins for a research run: a small corpus, its catalog and scripted sub-agents."""

import re

from agent.schemas import BatchAnalysis, ExtractedFinding, Finding, ResearchCode
from retrieval.catalog import CatalogEntry, DocumentCatalog
from retrieval.models import SearchResult
from tests.fakes.fake_search import make_chunk

DOCUMENT_TAG_PATTERN = re.compile(r'<document id="(\d+)"')
EXPLORE_AND_DELEGATE_PLAN = """
documents = explore_documents(document_types=["incident"])
sections = fetch_sections([document["doc_id"] for document in documents], "Root cause")
delegate("State the root cause.", chunk(sections, 2))
"""


def make_research_code(python_code=EXPLORE_AND_DELEGATE_PLAN):
    return ResearchCode(reasoning="Read the root causes.", python_code=python_code)


def make_section(doc_id, chunk_number=0, access_level="internal", text=None):
    chunk = make_chunk(
        f"{doc_id}#c{chunk_number}",
        access_level=access_level,
        text=text or f"Source text of {doc_id}: certificate expired.",
    )
    section = "Summary; Root cause" if chunk_number == 0 else "Lessons learned"
    return chunk.model_copy(update={"section": section})


def make_catalog_entry(doc_id, access_level="internal", chunk_count=1):
    return CatalogEntry(
        doc_id=doc_id,
        title=f"Title of {doc_id}",
        department="payments",
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=1768435200,
        source_uri=f"docs://payments/{doc_id}",
        chunk_count=chunk_count,
    )


def make_catalog(*entries):
    return DocumentCatalog(list(entries))


def make_finding(finding_id, doc_id, theme_label="certificate expiry", date="2026-01-15"):
    return Finding(
        finding_id=finding_id,
        doc_id=doc_id,
        title=f"Title of {doc_id}",
        created_date=date,
        claim=f"{doc_id} failed.",
        theme_label=theme_label,
        evidence_chunk_ids=[f"{doc_id}#c0"],
    )


def analyse_every_document(messages):
    """Reply as a sub-agent would, with one finding per document tag in the prompt."""
    markers = [int(marker) for marker in DOCUMENT_TAG_PATTERN.findall(messages[-1].text)]
    return BatchAnalysis(
        findings=[
            ExtractedFinding(
                claim=f"Finding from document {marker}.",
                theme_label="certificate expiry",
                evidence_markers=[marker],
            )
            for marker in markers
        ],
        unresolved_markers=[],
    )


class FakeResearchSearch:
    """Holds chunks and, like the real service, has no idea what a plan intends."""

    def __init__(self, chunks=(), degraded=()):
        self.chunks = list(chunks)
        self.degraded = list(degraded)
        self.search_calls = []
        self.fetched_chunk_ids = []

    async def search(self, query_text, user, document_filters=None, namespaces=None, **options):
        self.search_calls.append({"query": query_text, "user": user, **options})
        return SearchResult(chunks=self.chunks, degraded=self.degraded)

    async def fetch_chunks(self, chunk_ids_by_namespace, user):
        requested_ids = [
            chunk_id for chunk_ids in chunk_ids_by_namespace.values() for chunk_id in chunk_ids
        ]
        self.fetched_chunk_ids.extend(requested_ids)
        found_chunks = [chunk for chunk in self.chunks if chunk.chunk_id in requested_ids]
        return SearchResult(chunks=found_chunks, degraded=self.degraded)

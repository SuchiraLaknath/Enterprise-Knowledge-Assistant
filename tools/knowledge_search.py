"""`knowledge_search`: let the tool agent search the documents.

Rule 1: the access filter comes from the verified user in `ToolContext`, and the args schema has
no access field. Rule 5: results become evidence, so they are cited like any other chunk. It
skips the reranker, which only `retrieval_agent` may use, to protect the monthly quota.
"""

from langsmith import traceable
from pydantic import Field

from agent.schemas import Department, DocumentType
from retrieval.models import DocumentFilters
from tools.models import ToolArguments, ToolContext, ToolResult
from tools.tracing import hide_tool_inputs

TOOL_NAME = "knowledge_search"


class KnowledgeSearchArguments(ToolArguments):
    """What a model may ask for. Access level is deliberately absent."""

    query: str = Field(min_length=1, max_length=500, description="What to search for.")
    departments: list[Department] = Field(
        default=[], description="Departments to search. Empty means all of them."
    )
    document_types: list[DocumentType] = Field(
        default=[], description="Document types to search. Empty means all of them."
    )


@traceable(name=TOOL_NAME, run_type="tool", process_inputs=hide_tool_inputs)
async def search_knowledge(arguments: KnowledgeSearchArguments, context: ToolContext) -> ToolResult:
    """Search and return attributed chunks, with no document text in the tool's own data."""
    search_result = await context.document_search.search(
        arguments.query,
        context.user,
        DocumentFilters(document_types=tuple(arguments.document_types)),
        namespaces=tuple(arguments.departments) or None,
        use_reranker=False,
    )
    # The text travels as evidence, which is wrapped in document tags before any model sees it.
    found_documents = [
        {"chunk_id": chunk.chunk_id, "title": chunk.title, "section": chunk.section}
        for chunk in search_result.chunks
    ]
    return ToolResult(
        tool_name=TOOL_NAME, status="ok", data=found_documents, evidence=search_result.chunks
    )

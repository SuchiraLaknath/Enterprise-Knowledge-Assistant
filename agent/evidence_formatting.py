"""Put retrieved text into prompts, wrapped so that it can only be read as data.

Rule 5: every document or finding goes inside `<document>` tags, with its text escaped so that
it cannot close its own tag. Sub-agents and the response agent share this, so there is one
implementation of the rule. `api/tracing.py` hides the body of these tags from LangSmith.
"""

import json
from html import escape

from agent.schemas import Finding, ResearchReport
from retrieval.models import RetrievedChunk
from tools.models import ToolResult


def format_document_tag(tag_id: int | str, title: str, source_label: str, body: str) -> str:
    """Wrap one piece of text in a document tag, escaping everything a document controls.

    `source_label` says where the text sits, such as a section heading.
    """
    safe_title = escape(title, quote=True)
    safe_label = escape(source_label, quote=True)
    return (
        f'<document id="{tag_id}" title="{safe_title}" section="{safe_label}">\n'
        f"{escape(body, quote=False)}\n"
        "</document>"
    )


def format_evidence_documents(evidence: list[RetrievedChunk]) -> str:
    """Wrap each chunk in document tags, numbered to match its citation marker."""
    documents = [
        format_document_tag(marker, chunk.title, chunk.section, chunk.text)
        for marker, chunk in enumerate(evidence, start=1)
    ]
    return "\n\n".join(documents)


def format_tool_results(tool_results: list[ToolResult]) -> str:
    """Wrap each tool result as data (rule 5). Records from a tool are untrusted like documents."""
    result_tags = []
    for tool_result in tool_results:
        body = json.dumps(tool_result.data, default=str) if tool_result.status == "ok" else ""
        safe_name = escape(tool_result.tool_name, quote=True)
        result_tags.append(
            f'<tool_result name="{safe_name}" status="{tool_result.status}">\n'
            f"{escape(body or tool_result.message, quote=False)}\n"
            "</tool_result>"
        )
    return "\n\n".join(result_tags)


def _format_finding_tags(finding: Finding, marker_by_chunk_id: dict[str, int]) -> list[str]:
    """Return one tag per evidence chunk, so that each citation marker has a tag to point to."""
    source_label = f"{finding.created_date}, {finding.doc_id}"
    tags = []
    for chunk_id in finding.evidence_chunk_ids:
        tags.append(
            format_document_tag(
                marker_by_chunk_id[chunk_id], finding.title, source_label, finding.claim
            )
        )
    return tags


def format_research_report(report: ResearchReport, evidence: list[RetrievedChunk]) -> str:
    """Show the report's themes, with each finding numbered to match `state["evidence"]`.

    Rule 4: markers are positions in `evidence`, exactly as in a normal answer, so citations
    work the same way and the validator needs no special case.
    """
    marker_by_chunk_id = {chunk.chunk_id: marker for marker, chunk in enumerate(evidence, start=1)}
    theme_blocks = []
    for theme in report.themes:
        theme_heading = (
            f'<theme name="{escape(theme.name, quote=True)}" '
            f'documents="{theme.document_count}" '
            f'first_date="{theme.first_date}" last_date="{theme.last_date}" '
            f'summary="{escape(theme.summary, quote=True)}">'
        )
        finding_tags: list[str] = []
        for finding in theme.findings:
            finding_tags.extend(_format_finding_tags(finding, marker_by_chunk_id))
        theme_blocks.append("\n".join([theme_heading, *finding_tags, "</theme>"]))
    return "\n\n".join(theme_blocks)

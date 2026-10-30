"""Choose what retrieval sends to LangSmith, which is an external service.

Both directions use an allow-list, so a new parameter or field stays out of traces until it
is named here. Traces show which chunks were found and why, but never their text. The agent
nodes add `user_id`, `role` and `thread_id` as run metadata.
"""

from typing import Any

TRACED_INPUT_NAMES = (
    "query_text",
    "namespaces",
    "document_filters",
    "use_reranker",
    "result_count",
)

TRACED_CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "section",
    "department",
    "access_level",
    "score",
)

TRACED_RESULT_FIELDS = ("degraded", "dense_weight", "failed_namespaces")


def keep_query_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the meaningful inputs of a retrieval call.

    Traces should show what was searched for, not client objects or the whole encoder.
    """
    return {name: inputs[name] for name in TRACED_INPUT_NAMES if name in inputs}


def keep_output_summary(outputs: Any) -> dict[str, Any]:
    """Describe a search result without its document text.

    `outputs` is either a list of chunks or a result object holding `chunks`.
    """
    chunks = outputs if isinstance(outputs, list) else outputs.chunks
    summary: dict[str, Any] = {
        "chunks": [
            {field: getattr(chunk, field) for field in TRACED_CHUNK_FIELDS} for chunk in chunks
        ]
    }
    for field in TRACED_RESULT_FIELDS:
        if hasattr(outputs, field):
            summary[field] = getattr(outputs, field)
    return summary


# Shared by every retrieval function, so that none can forget the allow-lists above.
RETRIEVER_TRACE_OPTIONS: dict[str, Any] = {
    "run_type": "retriever",
    "process_inputs": keep_query_inputs,
    "process_outputs": keep_output_summary,
}

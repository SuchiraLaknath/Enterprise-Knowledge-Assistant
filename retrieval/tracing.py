"""Helpers that keep LangSmith traces readable."""

from typing import Any

TRACED_INPUT_NAMES = (
    "query_text",
    "namespaces",
    "document_filters",
    "use_reranker",
    "result_count",
)


def keep_query_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the meaningful inputs of a retrieval call.

    Traces should show what was searched for, not client objects or the whole encoder.
    """
    return {name: inputs[name] for name in TRACED_INPUT_NAMES if name in inputs}

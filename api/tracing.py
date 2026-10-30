"""Keep document text out of LangSmith, which is an external service.

Retrieval traces already use an allow-list, but LangChain also traces every graph node and LLM
call, and those carry the evidence and the prompts. This module redacts them in one place, so
that no node can forget. Traces still show the structure of a turn: which chunks were used, the
system prompt, the question and the answer.
"""

import re
from typing import Any

import langsmith
from langsmith import Client
from pydantic import BaseModel

# The body of a document in a prompt, between its opening and closing tags.
DOCUMENT_BODY_PATTERN = re.compile(r"(<document [^>]*>).*?(</document>)", re.DOTALL)
HIDDEN_DOCUMENT_TEXT = r"\1[document text hidden]\2"


def redact_document_text(payload: Any) -> Any:
    """Return a copy of a trace payload with all document text removed.

    Payloads hold nested dicts, lists and pydantic objects, so this walks every value.
    """
    if isinstance(payload, str):
        return DOCUMENT_BODY_PATTERN.sub(HIDDEN_DOCUMENT_TEXT, payload)
    if isinstance(payload, BaseModel):
        return redact_document_text(payload.model_dump())
    if isinstance(payload, dict):
        # A record with both of these fields is a retrieved chunk, whose text must not leave.
        is_retrieved_chunk = "chunk_id" in payload and "text" in payload
        return {
            key: redact_document_text(value)
            for key, value in payload.items()
            if not (is_retrieved_chunk and key == "text")
        }
    if isinstance(payload, list | tuple):
        return [redact_document_text(value) for value in payload]
    return payload


def configure_redacted_tracing() -> None:
    """Make every trace in this process pass through the redaction."""
    langsmith.configure(
        client=Client(hide_inputs=redact_document_text, hide_outputs=redact_document_text)
    )

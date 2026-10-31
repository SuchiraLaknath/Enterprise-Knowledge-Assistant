"""Keep document text out of LangSmith, which is an external service.

Retrieval traces already use an allow-list, but LangChain also traces every graph node and LLM
call, and those carry the evidence and the prompts. This module redacts them in one place, so
that no node can forget. Traces still show the structure of a turn: which chunks were used, the
system prompt, the question and the answer.

Research findings are hidden as well, because a sub-agent's claim often quotes its document
closely. Their ids, labels and evidence chunk ids stay, so a trace still shows what was found.
"""

import re
from typing import Any

import langsmith
from langsmith import Client
from pydantic import BaseModel

# The body of a document in a prompt, between its opening and closing tags.
DOCUMENT_BODY_PATTERN = re.compile(r"(<document [^>]*>).*?(</document>)", re.DOTALL)
HIDDEN_DOCUMENT_TEXT = r"\1[document text hidden]\2"

# A sub-agent's raw reply is JSON inside a string, so its claims need a pattern as well.
CLAIM_FIELD_NAME = "claim"
CLAIM_IN_JSON_PATTERN = re.compile(r'("claim"\s*:\s*")(?:[^"\\]|\\.)*(")')
HIDDEN_FINDING_TEXT = "[finding text hidden]"


def redact_document_text(payload: Any) -> Any:
    """Return a copy of a trace payload with all document text removed.

    Payloads hold nested dicts, lists and pydantic objects, so this walks every value.
    """
    if isinstance(payload, str):
        without_documents = DOCUMENT_BODY_PATTERN.sub(HIDDEN_DOCUMENT_TEXT, payload)
        return CLAIM_IN_JSON_PATTERN.sub(rf"\1{HIDDEN_FINDING_TEXT}\2", without_documents)
    if isinstance(payload, BaseModel):
        return redact_document_text(payload.model_dump())
    if isinstance(payload, dict):
        # A record with both of these fields is a retrieved chunk, whose text must not leave.
        is_retrieved_chunk = "chunk_id" in payload and "text" in payload
        return {
            key: HIDDEN_FINDING_TEXT if key == CLAIM_FIELD_NAME else redact_document_text(value)
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

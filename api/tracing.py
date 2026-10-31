"""Keep document text out of LangSmith, which is an external service.

Retrieval traces already use an allow-list, but LangChain also traces every graph node and LLM
call, and those carry the evidence and the prompts. This module redacts them in one place, so
that no node can forget. Traces still show the structure of a turn: which chunks were used, the
system prompt, the question and the answer.

Research findings are hidden as well, because a sub-agent's claim often quotes its document
closely, and a theme summary paraphrases those claims. Their ids, labels and evidence chunk ids
stay, so a trace still shows what was found. Tool data, tool arguments and memory are hidden for
the same reason. An error keeps only its type, because a parser's error quotes the whole reply
that it could not read.
"""

import re
from typing import Any

import langsmith
from langsmith import Client
from pydantic import BaseModel

# The body of a document in a prompt, between its opening and closing tags.
DOCUMENT_BODY_PATTERN = re.compile(r"(<document [^>]*>).*?(</document>)", re.DOTALL)
HIDDEN_DOCUMENT_TEXT = r"\1[document text hidden]\2"

# Text in a prompt that came from a tool, or from what was remembered about the employee.
TOOL_RESULT_BODY_PATTERN = re.compile(r"(<tool_result [^>]*>).*?(</tool_result>)", re.DOTALL)
MEMORY_BODY_PATTERN = re.compile(r"(<memory>).*?(</memory>)", re.DOTALL)
HIDDEN_DATA_TEXT = r"\1[data hidden]\2"

# Model text that comes from documents, and fields that hold tool arguments or memory. In parsed
# form these are ordinary fields. `args` is how a model's tool call carries its arguments.
HIDDEN_FIELD_NAMES = ("claim", "summary", "args", "memory_context", "conversation_summary")
HIDDEN_FINDING_TEXT = "[finding text hidden]"

# Before parsing, a model's reply is JSON inside a string. The closing quote is optional, because
# a reply can be cut off partway through a claim.
FINDING_IN_JSON_PATTERN = re.compile(r'("(?:claim|summary)"\s*:\s*")(?:[^"\\]|\\.)*("|\Z)')
# The response prompt carries each theme's summary as a tag attribute.
THEME_SUMMARY_PATTERN = re.compile(r'(<theme [^>]*summary=")[^"]*(")')
HIDDEN_FINDING_SUBSTITUTION = rf"\1{HIDDEN_FINDING_TEXT}\2"

# The name at the start of an error, such as `OutputParserException`.
ERROR_TYPE_PATTERN = re.compile(r"[\w.]+")
UNKNOWN_ERROR_TYPE = "UnknownError"


def redact_document_text(payload: Any) -> Any:
    """Return a copy of a trace payload with all document text removed.

    Payloads hold nested dicts, lists and pydantic objects, so this walks every value.
    """
    if isinstance(payload, str):
        without_documents = DOCUMENT_BODY_PATTERN.sub(HIDDEN_DOCUMENT_TEXT, payload)
        without_documents = TOOL_RESULT_BODY_PATTERN.sub(HIDDEN_DATA_TEXT, without_documents)
        without_documents = MEMORY_BODY_PATTERN.sub(HIDDEN_DATA_TEXT, without_documents)
        without_summaries = THEME_SUMMARY_PATTERN.sub(
            HIDDEN_FINDING_SUBSTITUTION, without_documents
        )
        return FINDING_IN_JSON_PATTERN.sub(HIDDEN_FINDING_SUBSTITUTION, without_summaries)
    if isinstance(payload, BaseModel):
        return redact_document_text(payload.model_dump())
    if isinstance(payload, dict):
        # A record with both of these fields is a retrieved chunk, whose text must not leave.
        is_retrieved_chunk = "chunk_id" in payload and "text" in payload
        # A record with `tool_name` and `data` is a tool result, which can hold internal records.
        is_tool_result = "tool_name" in payload and "data" in payload
        # Excerpts are raw document text, so such an answer is hidden, as an answer normally isn't.
        hidden_field_names: tuple[str, ...] = HIDDEN_FIELD_NAMES
        if payload.get("answer_is_excerpts"):
            hidden_field_names = (*HIDDEN_FIELD_NAMES, "final_answer")
        return {
            key: HIDDEN_FINDING_TEXT if key in hidden_field_names else redact_document_text(value)
            for key, value in payload.items()
            if not (is_retrieved_chunk and key == "text") and not (is_tool_result and key == "data")
        }
    if isinstance(payload, list | tuple):
        return [redact_document_text(value) for value in payload]
    return payload


def anonymize_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Clean one payload on its way to LangSmith.

    LangSmith wraps a run's error as `{"error": text}`. Only the error type is kept, because the
    text can quote a model's reply, in a form that no pattern can clean reliably.
    """
    if set(payload) == {"error"}:
        error_type = ERROR_TYPE_PATTERN.match(str(payload["error"]))
        return {"error": error_type.group() if error_type else UNKNOWN_ERROR_TYPE}
    redacted_payload: dict[str, Any] = redact_document_text(payload)
    return redacted_payload


def configure_redacted_tracing() -> None:
    """Make every trace in this process pass through the redaction.

    This uses `anonymizer` rather than `hide_inputs` and `hide_outputs`, because it is the only
    hook that also covers a run's `error` field.
    """
    langsmith.configure(client=Client(anonymizer=anonymize_trace_payload))

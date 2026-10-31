"""`response_agent`: write the cited answer, streaming it as a provisional draft.

Rule 9: tokens are masked and shown as a draft. Only the finished answer is added to the
conversation and sent as the `final` event. If no model can answer, the employee still gets the
evidence, as cited excerpts that need no LLM.
"""

import re
from html import escape
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError, GatewayResult
from agent.prompts.response import (
    EVIDENCE_ACKNOWLEDGEMENT,
    EVIDENCE_MESSAGE_TEMPLATE,
    NO_DOCUMENTS_SYSTEM_PROMPT,
    NO_EVIDENCE_ACKNOWLEDGEMENT,
    NO_EVIDENCE_MESSAGE,
    RESPONSE_SYSTEM_PROMPT,
)
from agent.schemas import Citation
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from retrieval.models import RetrievedChunk
from security.output_guard import clean_final_answer
from security.stream_masker import StreamMasker, mask_sensitive_text

NODE_NAME = "response_agent"
ANSWER_ROUTE = "response"

CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")
EXCERPT_WORD_COUNT = 60

DEGRADED_MODEL_FALLBACK = "response_used_fallback_model"
DEGRADED_EXTRACTIVE = "extractive_answer_without_llm"
CONVERSATION_NOTE_FOR_EXCERPTS = (
    "I could not write an answer, so I showed excerpts from these documents: {doc_ids}."
)
NO_MODEL_NO_EVIDENCE_REPLY = (
    "I'm sorry, the assistant is temporarily unavailable. Please try again shortly."
)


def format_evidence_documents(evidence: list[RetrievedChunk]) -> str:
    """Wrap each chunk in document tags, numbered to match its citation marker.

    Rule 5: escaping stops a document from closing its own tag and posing as an instruction.
    """
    documents = []
    for marker, chunk in enumerate(evidence, start=1):
        title = escape(chunk.title, quote=True)
        section = escape(chunk.section, quote=True)
        documents.append(
            f'<document id="{marker}" title="{title}" section="{section}">\n'
            f"{escape(chunk.text, quote=False)}\n"
            "</document>"
        )
    return "\n\n".join(documents)


def extract_citations(answer: str, evidence: list[RetrievedChunk]) -> list[Citation]:
    """Return the citation for each `[n]` used.

    Rule 4: only markers that match this turn's evidence become citations.
    """
    used_markers = sorted({int(marker) for marker in CITATION_MARKER_PATTERN.findall(answer)})
    citations = []
    for marker in used_markers:
        if 1 <= marker <= len(evidence):
            chunk = evidence[marker - 1]
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    section=chunk.section,
                    source_uri=chunk.source_uri,
                )
            )
    return citations


def build_extractive_answer(evidence: list[RetrievedChunk]) -> str:
    """Show the evidence directly, for when no model can write an answer."""
    if not evidence:
        return NO_MODEL_NO_EVIDENCE_REPLY

    excerpts = []
    for marker, chunk in enumerate(evidence, start=1):
        # Masking comes before the cut, which could otherwise split a number and hide it.
        excerpt = " ".join(mask_sensitive_text(chunk.text).split()[:EXCERPT_WORD_COUNT])
        excerpts.append(f"[{marker}] **{chunk.title}** ({chunk.section}): {excerpt}...")
    introduction = (
        "The assistant cannot write a full answer at the moment, so here are the most "
        "relevant excerpts from the documents:"
    )
    return "\n\n".join([introduction, *excerpts])


def remove_citation_markers(message: BaseMessage) -> BaseMessage:
    """Remove `[n]` from an old answer, because each turn numbers its evidence afresh.

    Rule 4: a repeated `[1]` from an earlier answer would point at a different document.
    """
    if isinstance(message, AIMessage):
        return AIMessage(CITATION_MARKER_PATTERN.sub("", message.text))
    return message


def _build_answer_messages(state: AgentState, dependencies: AgentDependencies) -> list[BaseMessage]:
    """Assemble the system prompt, the evidence and the recent conversation."""
    plan = state.get("plan")
    evidence = state.get("evidence", [])
    recent_messages = state["messages"][-dependencies.history_message_count :]
    conversation = [remove_citation_markers(message) for message in recent_messages]

    if plan is not None and not plan.needs_documents:
        return [SystemMessage(NO_DOCUMENTS_SYSTEM_PROMPT), *conversation]

    if evidence:
        evidence_text = EVIDENCE_MESSAGE_TEMPLATE.format(
            documents=format_evidence_documents(evidence)
        )
        acknowledgement = EVIDENCE_ACKNOWLEDGEMENT
    else:
        evidence_text = NO_EVIDENCE_MESSAGE
        acknowledgement = NO_EVIDENCE_ACKNOWLEDGEMENT

    # Rule 5: the documents sit in a user turn, followed by a fixed reply that labels them as
    # data, so they never read as instructions. The employee's question comes last.
    return [
        SystemMessage(RESPONSE_SYSTEM_PROMPT),
        HumanMessage(evidence_text),
        AIMessage(acknowledgement),
        *conversation,
    ]


async def _stream_masked_answer(
    messages: list[BaseMessage], dependencies: AgentDependencies
) -> GatewayResult[str]:
    """Stream the answer through the masker, and restart the draft if a model fails."""
    masker = StreamMasker()

    async def send_masked_text(text: str) -> None:
        safe_text = masker.feed(text)
        if safe_text:
            emit_event(EventType.TOKEN, text=safe_text)

    async def clear_draft() -> None:
        # Text held by the masker belonged to the failed model, so it is dropped here.
        masker.flush()
        emit_event(EventType.TOKEN, reset=True)

    answer_result = await dependencies.llm.stream_text(
        ANSWER_ROUTE, messages, send_masked_text, clear_draft
    )

    remaining_text = masker.flush()
    if remaining_text:
        emit_event(EventType.TOKEN, text=remaining_text)
    return answer_result


async def response_agent(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Write the answer, or fall back to excerpts if no model is available."""
    emit_node_started(NODE_NAME)
    evidence = state.get("evidence", [])
    degraded: list[str] = []

    # Step 1: stream an answer from the first model that can finish one.
    answer_messages = _build_answer_messages(state, runtime.context)
    try:
        answer_result = await _stream_masked_answer(answer_messages, runtime.context)
        answer = answer_result.value
        conversation_note = None
        degraded.extend(
            f"{DEGRADED_MODEL_FALLBACK}:{model_spec}"
            for model_spec in answer_result.failed_model_specs
        )
    except AllModelsFailedError:
        answer = build_extractive_answer(evidence)
        degraded.append(DEGRADED_EXTRACTIVE)
        # Rule 5: excerpts are raw document text, so they must not join the conversation,
        # where later turns would read them as the assistant's own words.
        cited_doc_ids = ", ".join(sorted({chunk.doc_id for chunk in evidence})) or "none"
        conversation_note = CONVERSATION_NOTE_FOR_EXCERPTS.format(doc_ids=cited_doc_ids)

    # Step 2: clean the complete text, because the masker protected only the draft, and the
    # answer may be excerpts that never passed through it. Then link its citations.
    final_answer = clean_final_answer(mask_sensitive_text(answer))
    citations = extract_citations(final_answer, evidence)

    # Step 3: only now does the answer join the conversation (rule 9).
    emit_event(EventType.STATE, citation_count=len(citations), degraded=degraded)
    emit_node_finished(NODE_NAME)
    return {
        "messages": [AIMessage(conversation_note or final_answer)],
        "final_answer": final_answer,
        "final_citations": citations,
        "degraded": degraded,
    }

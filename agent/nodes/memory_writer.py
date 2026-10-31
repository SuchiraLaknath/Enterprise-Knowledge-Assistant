"""`memory_writer`: keep what is worth keeping, after the answer has passed validation.

Two tiers are written here. Older turns move into a rolling summary, so that a long conversation
stays within the prompt. Lasting facts about the employee go to long-term memory (rule 8): they
come only from what the employee typed, they pass the injection checks, and alongside them are
doc ids, never document text.

Memory is a bonus, so a failure here is recorded and the answer still goes out.
"""

import asyncio
from html import escape
from typing import Any

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError
from agent.nodes.load_context import DEGRADED_MEMORY_OFF
from agent.nodes.supervisor import latest_user_text
from agent.prompts.memory_writer import (
    FACT_EXTRACTION_REQUEST_TEMPLATE,
    FACT_EXTRACTION_SYSTEM_PROMPT,
    SUMMARY_REQUEST_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
)
from agent.schemas import ConversationSummary, ExtractedFacts
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from security.content_guard import contains_injection
from security.input_guard import find_input_attack
from security.stream_masker import mask_sensitive_text

logger = structlog.get_logger()

NODE_NAME = "memory_writer"
MEMORY_ROUTE = "fast"
DEGRADED_SUMMARY_SKIPPED = "conversation_summary_skipped"
# Facts are about the speaker, so a message with no first person has none, and needs no LLM call.
FIRST_PERSON_WORDS = (" i ", " i'm ", " my ", " we ", " our ", " i am ")


def is_fact_safe_to_store(fact: str, max_characters: int) -> bool:
    """Return whether a fact may enter memory. Rule 8: injected text must never get this far."""
    if not fact.strip() or len(fact) > max_characters:
        return False
    # Masking changed the text, so it held a NIC or card number.
    if mask_sensitive_text(fact) != fact:
        return False
    return not contains_injection(fact) and find_input_attack(fact) is None


def _might_state_a_fact(message: str) -> bool:
    """Return whether a message could hold a fact about the speaker."""
    padded_message = f" {message.lower()} "
    return any(word in padded_message for word in FIRST_PERSON_WORDS)


async def _extract_facts(message: str, dependencies: AgentDependencies) -> list[str]:
    """Ask for lasting facts in the employee's own message, and keep the safe ones."""
    limits = dependencies.conversation_limits
    # Rule 8: only the employee's words go in, so no document or answer can become a memory.
    messages: list[BaseMessage] = [
        SystemMessage(FACT_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(FACT_EXTRACTION_REQUEST_TEMPLATE.format(message=escape(message, quote=False))),
    ]
    extraction = await dependencies.llm.generate_structured(MEMORY_ROUTE, messages, ExtractedFacts)
    safe_facts = [
        fact
        for fact in extraction.value.facts
        if is_fact_safe_to_store(fact, limits.max_fact_characters)
    ]
    return safe_facts[: limits.max_facts_per_turn]


async def _write_long_term_memory(state: AgentState, dependencies: AgentDependencies) -> list[str]:
    """Store this turn's facts, and return any degradation."""
    message = latest_user_text(state)
    if not _might_state_a_fact(message):
        return []

    limits = dependencies.conversation_limits
    # Rule 8: doc ids from citations, which code wrote, and never any text from a document.
    cited_doc_ids = sorted(
        {citation.chunk_id.split("#")[0] for citation in state.get("final_citations", [])}
    )
    try:
        facts = await _extract_facts(message, dependencies)
        async with asyncio.timeout(limits.memory_timeout_s):
            for fact in facts:
                await dependencies.long_term_memory.write(
                    state["user"].user_id, fact, cited_doc_ids
                )
    except Exception as error:  # Failures are data, and the answer still goes out.
        logger.warning("memory_write_failed", error=type(error).__name__)
        return [DEGRADED_MEMORY_OFF]

    emit_event(EventType.MEMORY, operation="write", tier="long_term", item_count=len(facts))
    return []


def _render_conversation(messages: list[BaseMessage]) -> str:
    """Write messages out for the summary, escaped because they can quote anything."""
    lines = []
    for message in messages:
        speaker = "Assistant" if isinstance(message, AIMessage) else "Employee"
        lines.append(f"{speaker}: {escape(message.text, quote=False)}")
    return "\n".join(lines)


async def _summarize_older_turns(
    state: AgentState, dependencies: AgentDependencies
) -> dict[str, Any]:
    """Move older messages into the summary, and remove them from the conversation."""
    limits = dependencies.conversation_limits
    all_messages = state["messages"]
    if len(all_messages) <= limits.summarize_after_messages:
        return {}

    older_messages = all_messages[: -limits.history_message_count]
    request_text = SUMMARY_REQUEST_TEMPLATE.format(
        earlier_summary=state.get("conversation_summary", "None."),
        conversation=_render_conversation(older_messages),
    )
    messages: list[BaseMessage] = [
        SystemMessage(SUMMARY_SYSTEM_PROMPT),
        HumanMessage(request_text),
    ]
    try:
        summary_result = await dependencies.llm.generate_structured(
            MEMORY_ROUTE, messages, ConversationSummary
        )
    except AllModelsFailedError:
        # Nothing is removed, so no history is lost, and the next turn can try again.
        return {"degraded": [DEGRADED_SUMMARY_SKIPPED]}

    emit_event(EventType.MEMORY, operation="write", tier="summary", item_count=len(older_messages))
    return {
        "conversation_summary": summary_result.value.summary,
        "messages": [RemoveMessage(id=message.id) for message in older_messages if message.id],
    }


async def memory_writer(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Update the rolling summary and long-term memory."""
    emit_node_started(NODE_NAME)
    update = await _summarize_older_turns(state, runtime.context)
    memory_degraded = await _write_long_term_memory(state, runtime.context)
    update["degraded"] = [*update.get("degraded", []), *memory_degraded]
    emit_event(EventType.STATE, degraded=update["degraded"])
    emit_node_finished(NODE_NAME)
    return update

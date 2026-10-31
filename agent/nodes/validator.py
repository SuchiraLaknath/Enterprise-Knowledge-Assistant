"""`validator`: check an answer before it counts, and send it back if it falls short.

Deterministic checks run first: citations must point at this turn's evidence (rule 4), and known
brand patterns must be absent. An LLM judge then checks support and tone. The outcome is one of:

    pass                    the answer joins the conversation, and the turn ends
    fail                    the response agent writes it again, up to the retry limit
    insufficient evidence   the search runs once more, without the plan's narrowing

After the last failed retry, the answer is replaced by cited excerpts of the documents, which
need no model. Rule 9: nothing reaches the conversation until this node lets it.
"""

from html import escape
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import make_prompt_safe
from agent.llm_gateway import AllModelsFailedError
from agent.nodes.response_agent import (
    CONVERSATION_NOTE_FOR_EXCERPTS,
    NO_RELIABLE_ANSWER_REPLY,
    build_evidence_text,
    build_extractive_answer,
    extract_citations,
    find_citation_markers,
)
from agent.nodes.supervisor import latest_user_text
from agent.prompts.validator import (
    ANSWER_JUDGE_REQUEST_TEMPLATE,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    RETRY_FEEDBACK_TEMPLATE,
)
from agent.schemas import AnswerJudgement, Plan, ValidationReport
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from security.output_guard import find_brand_problem

logger = structlog.get_logger()

NODE_NAME = "validator"
JUDGE_ROUTE = "fast"
# Enough of the conversation to check a follow-up, without paying for all of it.
JUDGE_CONVERSATION_MESSAGES = 6

DEGRADED_JUDGE_UNAVAILABLE = "answer_judge_unavailable"
DEGRADED_MODEL_FALLBACK = "validator_used_fallback_model"
DEGRADED_ANSWER_REPLACED = "answer_failed_validation_excerpts_shown"
EMPTY_ANSWER_PROBLEM = "The answer was empty."
BRAND_PROBLEM_TEMPLATE = "The answer breaks a brand rule: {category}."
DEAD_CITATION_TEMPLATE = "It cites [{marker}], which is not one of the documents provided."
# Fixed wording. The judge reads retrieved text, so its own words must not reach the next prompt.
UNSUPPORTED_PROBLEM = "It makes a claim that the evidence does not support."
OFF_BRAND_PROBLEM = "It breaks a brand rule: advice, a named customer, a competitor or its tone."


def find_dead_citations(answer: str, evidence_count: int) -> list[int]:
    """Return each `[n]` that points at nothing. Rule 4: these would be hallucinated sources."""
    markers = find_citation_markers(answer)
    return sorted(marker for marker in markers if not 1 <= marker <= evidence_count)


def check_answer_rules(answer: str, evidence_count: int) -> list[str]:
    """Return every rule the answer breaks that code can decide alone."""
    if not answer.strip():
        return [EMPTY_ANSWER_PROBLEM]

    problems = [
        DEAD_CITATION_TEMPLATE.format(marker=marker)
        for marker in find_dead_citations(answer, evidence_count)
    ]
    brand_category = find_brand_problem(answer)
    if brand_category:
        problems.append(BRAND_PROBLEM_TEMPLATE.format(category=brand_category))
    return problems


def _build_judge_evidence(state: AgentState) -> str:
    """Return what the answer may fairly draw on: evidence, memory and the conversation.

    Without the last two, the judge would reject a correct answer drawn from memory, or a reply
    to "what did I just ask?".
    """
    evidence_text, _ = build_evidence_text(state)
    recent_messages = state["messages"][-JUDGE_CONVERSATION_MESSAGES:]
    conversation = "\n".join(make_prompt_safe(message.text) for message in recent_messages)
    return f"{evidence_text}{state.get('memory_context', '')}\n\nConversation:\n{conversation}"


async def _ask_judge(
    state: AgentState, answer: str, dependencies: AgentDependencies
) -> tuple[list[str], list[str]]:
    """Return what the judge found, and any degradation. With no model, code's checks stand."""
    request_text = ANSWER_JUDGE_REQUEST_TEMPLATE.format(
        question=make_prompt_safe(latest_user_text(state)),
        evidence=_build_judge_evidence(state),
        answer=escape(answer, quote=False),
    )
    messages: list[BaseMessage] = [
        SystemMessage(ANSWER_JUDGE_SYSTEM_PROMPT),
        HumanMessage(request_text),
    ]
    try:
        judge_result = await dependencies.llm.generate_structured(
            JUDGE_ROUTE, messages, AnswerJudgement
        )
    except AllModelsFailedError:
        return [], [DEGRADED_JUDGE_UNAVAILABLE]

    judgement = judge_result.value
    degraded = judge_result.describe_fallbacks(DEGRADED_MODEL_FALLBACK)
    problems = []
    if not judgement.supported_by_evidence:
        problems.append(UNSUPPORTED_PROBLEM)
    if not judgement.on_brand:
        problems.append(OFF_BRAND_PROBLEM)
    return problems, degraded


def _needs_wider_search(state: AgentState, max_evidence_retries: int) -> bool:
    """Return whether nothing was found, and one wider search is still allowed."""
    plan = state.get("plan")
    return (
        plan is not None
        and plan.needs_documents
        and not state.get("evidence")
        and not state.get("tool_results")
        and state.get("research_report") is None
        and state.get("evidence_retries", 0) < max_evidence_retries
    )


def _widen_plan(plan: Plan, question: str) -> Plan:
    """Drop the plan's narrowing, and search with the employee's own words."""
    return plan.model_copy(
        update={
            "search_query": question,
            "departments": [],
            "document_types": [],
            "recent_days": None,
        }
    )


def _finish_turn(
    state: AgentState, report: ValidationReport, degraded: list[str]
) -> dict[str, Any]:
    """Let the answer join the conversation. Rule 9: this is the only place that does so."""
    stored_message = state.get("conversation_note") or state["final_answer"]
    return {
        "validation_report": report,
        "messages": [AIMessage(stored_message)],
        "degraded": degraded,
    }


def _replace_with_excerpts(
    state: AgentState, problems: list[str], degraded: list[str]
) -> dict[str, Any]:
    """Show cited excerpts instead, as the retries are used up. They cannot go off-brand."""
    evidence = state.get("evidence", [])
    excerpts = build_extractive_answer(evidence, NO_RELIABLE_ANSWER_REPLY)

    # A notice is owed whatever the answer turned out to be.
    turn_notice = state.get("turn_notice")
    if turn_notice:
        excerpts = f"{excerpts}\n\n{turn_notice}"
    doc_ids = ", ".join(sorted({chunk.doc_id for chunk in evidence})) or "none"
    # Rule 5: excerpts are raw document text, so the conversation keeps only a short note.
    return {
        "validation_report": ValidationReport(status="replaced_with_excerpts", reasons=problems),
        "final_answer": excerpts,
        "final_citations": extract_citations(excerpts, evidence),
        "messages": [AIMessage(CONVERSATION_NOTE_FOR_EXCERPTS.format(doc_ids=doc_ids))],
        "degraded": [*degraded, DEGRADED_ANSWER_REPLACED],
    }


async def validator(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Check the answer, and decide what happens to it."""
    emit_node_started(NODE_NAME)
    # The fixed notice is the code's own sentence, so it is not the model's claim to check.
    answer = state["final_answer"].removesuffix(f"\n\n{state.get('turn_notice') or ''}")
    evidence = state.get("evidence", [])

    # Step 1: nothing was found, so try once more before accepting that.
    limits = runtime.context.validation_limits
    if _needs_wider_search(state, limits.max_evidence_retries):
        plan = state["plan"]
        assert plan is not None
        report = ValidationReport(status="insufficient_evidence", reasons=["No evidence found."])
        evidence_retries = state.get("evidence_retries", 0) + 1
        emit_event(
            EventType.VALIDATION, check=NODE_NAME, status=report.status, reasons=report.reasons
        )
        emit_event(EventType.STATE, evidence_retries=evidence_retries)
        emit_node_finished(NODE_NAME)
        return {
            "validation_report": report,
            "plan": _widen_plan(plan, latest_user_text(state)),
            "retrieval_finished": False,
            "evidence_retries": evidence_retries,
        }

    # Step 2: code's checks come first, as they are instant and exact (rule 4).
    problems = check_answer_rules(answer, len(evidence))
    degraded: list[str] = []

    # Step 3: the judge reads for sense. Excerpts are skipped, as they quote the evidence itself.
    if not problems and not state.get("answer_is_excerpts", False):
        problems, degraded = await _ask_judge(state, answer, runtime.context)

    # Step 4: act on the result.
    report = ValidationReport(status="fail" if problems else "pass", reasons=problems)
    emit_event(EventType.VALIDATION, check=NODE_NAME, status=report.status, reasons=problems)
    emit_event(
        EventType.STATE, degraded=degraded, response_retries=state.get("response_retries", 0)
    )
    emit_node_finished(NODE_NAME)

    if not problems:
        return _finish_turn(state, report, degraded)

    retries_used = state.get("response_retries", 0)
    if retries_used >= limits.max_response_retries:
        logger.warning("answer_replaced_after_retries", retries=retries_used)
        return _replace_with_excerpts(state, problems, degraded)
    return {
        "validation_report": report,
        "validation_feedback": RETRY_FEEDBACK_TEMPLATE.format(problems=" ".join(problems)),
        "response_retries": retries_used + 1,
        "degraded": degraded,
    }


def choose_after_validation(
    state: AgentState,
) -> Literal["response_agent", "supervisor", "memory_writer"]:
    """Route on the report. Both loops are bounded by counters, and never by a model."""
    report = state["validation_report"]
    assert report is not None
    if report.status == "insufficient_evidence":
        return "supervisor"
    if report.status == "fail":
        return "response_agent"
    return "memory_writer"

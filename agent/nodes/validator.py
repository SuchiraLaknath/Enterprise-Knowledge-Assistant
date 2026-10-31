"""`validator`: check an answer before it counts, and send it back if it falls short.

Deterministic checks run first: citations must point at this turn's evidence (rule 4), and known
brand patterns must be absent. An LLM judge then checks support and tone. The outcome is one of:

    pass                    the answer joins the conversation, and the turn ends
    fail                    the response agent writes it again, up to the retry limit
    insufficient evidence   the search runs once more, without the plan's narrowing

After the last failed retry, the answer is replaced by cited excerpts, which need no model and
cannot go off-brand. Rule 9: nothing reaches the conversation until this node lets it.
"""

from html import escape
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError
from agent.nodes.response_agent import (
    CITATION_MARKER_PATTERN,
    CONVERSATION_NOTE_FOR_EXCERPTS,
    DEGRADED_EXTRACTIVE,
    build_evidence_text,
    build_extractive_answer,
    extract_citations,
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
CHECK_NAME = "validator"
MAX_RESPONSE_RETRIES = 2
MAX_EVIDENCE_RETRIES = 1

DEGRADED_JUDGE_UNAVAILABLE = "answer_judge_unavailable"
DEGRADED_MODEL_FALLBACK = "validator_used_fallback_model"
DEGRADED_ANSWER_REPLACED = "answer_failed_validation_excerpts_shown"
EMPTY_ANSWER_PROBLEM = "The answer was empty."
BRAND_PROBLEM_TEMPLATE = "The answer breaks a brand rule: {category}."
DEAD_CITATION_TEMPLATE = "It cites [{marker}], which is not one of the documents provided."


def find_dead_citations(answer: str, evidence_count: int) -> list[int]:
    """Return each `[n]` that points at nothing. Rule 4: these would be hallucinated sources."""
    markers = {int(marker) for marker in CITATION_MARKER_PATTERN.findall(answer)}
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


async def _ask_judge(
    state: AgentState, dependencies: AgentDependencies
) -> tuple[list[str], list[str]]:
    """Return what the judge found, and any degradation. With no model, code's checks stand."""
    evidence_text, _ = build_evidence_text(state)
    request_text = ANSWER_JUDGE_REQUEST_TEMPLATE.format(
        question=escape(latest_user_text(state), quote=False),
        evidence=evidence_text,
        answer=escape(state["final_answer"], quote=False),
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
    if judgement.supported_by_evidence and judgement.on_brand:
        return [], degraded
    return [judgement.problem or "The answer failed the judge's check."], degraded


def _needs_wider_search(state: AgentState) -> bool:
    """Return whether nothing was found, and one wider search is still allowed."""
    plan = state.get("plan")
    return (
        plan is not None
        and plan.needs_documents
        and not state.get("evidence")
        and state.get("research_report") is None
        and state.get("evidence_retries", 0) < MAX_EVIDENCE_RETRIES
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
    state: AgentState, report: ValidationReport, degraded: list[str]
) -> dict[str, Any]:
    """Show cited excerpts instead, as the retries are used up. They cannot go off-brand."""
    evidence = state.get("evidence", [])
    excerpts = build_extractive_answer(evidence)
    doc_ids = ", ".join(sorted({chunk.doc_id for chunk in evidence})) or "none"
    # Rule 5: excerpts are raw document text, so the conversation keeps only a short note.
    return {
        "validation_report": report,
        "final_answer": excerpts,
        "final_citations": extract_citations(excerpts, evidence),
        "messages": [AIMessage(CONVERSATION_NOTE_FOR_EXCERPTS.format(doc_ids=doc_ids))],
        "degraded": [*degraded, DEGRADED_ANSWER_REPLACED],
    }


async def validator(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Check the answer, and decide what happens to it."""
    emit_node_started(NODE_NAME)
    answer = state["final_answer"]
    evidence = state.get("evidence", [])

    # Step 1: nothing was found, so try once more before accepting that.
    if _needs_wider_search(state):
        plan = state["plan"]
        assert plan is not None
        report = ValidationReport(status="insufficient_evidence", reasons=["No evidence found."])
        emit_event(
            EventType.VALIDATION, check=CHECK_NAME, status=report.status, reasons=report.reasons
        )
        emit_node_finished(NODE_NAME)
        return {
            "validation_report": report,
            "plan": _widen_plan(plan, latest_user_text(state)),
            "retrieval_finished": False,
            "evidence_retries": state.get("evidence_retries", 0) + 1,
        }

    # Step 2: code's checks come first, as they are instant and exact (rule 4).
    problems = check_answer_rules(answer, len(evidence))
    degraded: list[str] = []

    # Step 3: the judge reads for sense. Excerpts are skipped, as they quote the evidence itself.
    is_excerpts = DEGRADED_EXTRACTIVE in state.get("degraded", [])
    if not problems and not is_excerpts:
        problems, degraded = await _ask_judge(state, runtime.context)

    # Step 4: act on the result.
    report = ValidationReport(status="fail" if problems else "pass", reasons=problems)
    emit_event(EventType.VALIDATION, check=CHECK_NAME, status=report.status, reasons=problems)
    emit_event(EventType.STATE, degraded=degraded)
    emit_node_finished(NODE_NAME)

    if not problems:
        return _finish_turn(state, report, degraded)

    retries_used = state.get("response_retries", 0)
    if retries_used >= MAX_RESPONSE_RETRIES:
        logger.warning("answer_replaced_after_retries", retries=retries_used)
        return _replace_with_excerpts(state, report, degraded)
    return {
        "validation_report": report,
        "validation_feedback": RETRY_FEEDBACK_TEMPLATE.format(problems=" ".join(problems)),
        "response_retries": retries_used + 1,
        "degraded": degraded,
    }


def choose_after_validation(
    state: AgentState,
) -> Literal["response_agent", "supervisor", "finish"]:
    """Route on the report. Both loops are bounded by counters, and never by a model."""
    report = state["validation_report"]
    assert report is not None
    if report.status == "insufficient_evidence":
        return "supervisor"
    if report.status == "fail" and state.get("validation_feedback") and not _was_replaced(state):
        return "response_agent"
    return "finish"


def _was_replaced(state: AgentState) -> bool:
    """Return whether the answer was swapped for excerpts, which ends the turn."""
    return DEGRADED_ANSWER_REPLACED in state.get("degraded", [])

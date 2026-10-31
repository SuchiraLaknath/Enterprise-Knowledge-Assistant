"""`research_agent`: the recursive RLM investigation, built as a subgraph.

One run follows this path:

    check_research_access   confirms the role again, because routing alone must not be trusted
    research_planner        gets a Python plan: from the LLM, or the template after failures
    research_runner         runs the plan in the sandbox, which delegates to sub-agents
    research_aggregator     groups the findings into a report with counted themes

If a plan fails, `research_runner` returns to `research_planner` for the next rung of the
ladder. If every rung fails, `research_gave_up` ends the run with no report, and the supervisor
falls back to plain retrieval, so one failed worker does not fail the turn.

The subgraph has its own input and output schemas. Without them, `evidence` would come back
holding what went in, and the additive reducer would count it twice.
"""

import time
from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.nodes.supervisor import latest_user_text
from agent.rlm.aggregate import build_themes, propose_themes
from agent.rlm.planner import TEMPLATE_PLAN_CODE, has_llm_attempts_left, write_plan_code
from agent.rlm.research_session import ResearchSession
from agent.schemas import ResearchReport, ResearchTheme
from agent.state import ResearchInput, ResearchOutput, ResearchState
from api.events import (
    EventType,
    ResearchStage,
    emit_event,
    emit_node_finished,
    emit_node_started,
)
from config.settings import SandboxLimits
from security.sandbox import run_sandboxed

logger = structlog.get_logger()

NODE_NAME = "research_agent"

PLAN_SOURCE_LLM = "llm"
PLAN_SOURCE_TEMPLATE = "template"

DEGRADED_ROLE_DENIED = "research_denied_for_role"
DEGRADED_INCOMPLETE = "research_incomplete"
DEGRADED_NO_FINDINGS = "research_failed_used_plain_retrieval"
NO_FINDINGS_ERROR = "The plan finished without any findings. Delegate batches to sub-agents."


async def check_research_access(
    state: ResearchState, runtime: Runtime[AgentDependencies]
) -> dict[str, Any]:
    """Confirm the role from the verified user, as a second check after routing (rule 2)."""
    # `research_agent` is the one node the main graph sees, so it opens here and closes at the
    # subgraph's last step. The nodes in between report as steps inside it.
    emit_node_started(NODE_NAME)
    if state["user"].role in runtime.context.research_roles:
        return {"access_confirmed": True}

    logger.warning("research_denied", role=state["user"].role)
    emit_event(EventType.STATE, degraded=[DEGRADED_ROLE_DENIED])
    emit_node_finished(NODE_NAME)
    return {
        "access_confirmed": False,
        "research_finished": True,
        "degraded": [DEGRADED_ROLE_DENIED],
    }


def _choose_after_access_check(state: ResearchState) -> Literal["research_planner", "stop"]:
    """Stop here unless the role was confirmed."""
    return "research_planner" if state["access_confirmed"] else "stop"


async def research_planner(
    state: ResearchState, runtime: Runtime[AgentDependencies]
) -> dict[str, Any]:
    """Get the next plan on the ladder: LLM, LLM with the error, then the template."""
    emit_node_started("research_planner")
    dependencies = runtime.context
    attempts_made = state.get("plan_attempt", 0)
    regeneration_limit = dependencies.research_limits.plan_regeneration_limit
    plan = state["plan"]
    # The supervisor routes here only after making a plan, so one always exists.
    assert plan is not None

    plan_code: str | None = None
    degraded: list[str] = []
    if has_llm_attempts_left(attempts_made, regeneration_limit):
        plan_code, degraded = await write_plan_code(
            question=latest_user_text(state),
            plan=plan,
            user=state["user"],
            dependencies=dependencies,
            previous_code=state.get("plan_code", ""),
            plan_error=state.get("plan_error", ""),
        )

    # If no model could write a plan, a retry would fail too, so this also skips the retries.
    template_used = plan_code is None
    if plan_code is None:
        plan_code = TEMPLATE_PLAN_CODE

    # The plan is shown in full, because generated code is what an evaluator wants to see.
    emit_event(
        EventType.RESEARCH,
        stage=ResearchStage.PLAN_READY,
        source=PLAN_SOURCE_TEMPLATE if template_used else PLAN_SOURCE_LLM,
        attempt=attempts_made + 1,
        code=plan_code,
    )
    emit_node_finished("research_planner")
    return {
        "plan_code": plan_code,
        "plan_attempt": attempts_made + 1,
        "template_used": template_used,
        "degraded": degraded,
    }


async def _run_plan(
    session: ResearchSession, state: ResearchState, sandbox_limits: SandboxLimits
) -> str:
    """Run the plan in the sandbox, and return what went wrong, or an empty string."""
    plan = state["plan"]
    assert plan is not None
    sandbox_names = session.build_sandbox_names(latest_user_text(state), plan)
    try:
        await run_sandboxed(state["plan_code"], sandbox_names, sandbox_limits)
    except Exception as error:  # Generated code can fail in any way, and each is a plan error.
        logger.warning("research_plan_failed", error=type(error).__name__)
        return f"{type(error).__name__}: {error}"
    return ""


async def research_runner(
    state: ResearchState, runtime: Runtime[AgentDependencies]
) -> dict[str, Any]:
    """Run the plan. Findings are kept even if the plan stopped partway through."""
    emit_node_started("research_runner")

    # Step 1: the deadline is set once per run, so a failed plan cannot earn a second budget.
    research_limits = runtime.context.research_limits
    run_deadline = state.get("run_deadline") or time.monotonic() + research_limits.deadline_s

    # Step 2: rule 1, the session holds the verified user, so the plan never handles identity.
    session = ResearchSession(
        state["user"], runtime.context, state.get("sub_agent_calls_used", 0), run_deadline
    )
    plan_error = await _run_plan(session, state, runtime.context.sandbox_limits)

    # Step 3: a plan that crashed may have found something first, so that work is kept.
    if plan_error and session.findings:
        session.degraded.append(DEGRADED_INCOMPLETE)
    # A plan can also run cleanly and find nothing, which the ladder treats as a failed plan.
    if not plan_error and not session.findings:
        plan_error = NO_FINDINGS_ERROR
    if plan_error:
        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.PLAN_FAILED,
            error=plan_error,
            findings_kept=len(session.findings),
        )
    emit_event(EventType.STATE, degraded=session.degraded)

    emit_node_finished("research_runner")
    return {
        "plan_error": plan_error,
        "findings": session.findings,
        "evidence": session.evidence_chunks(),
        "degraded": session.degraded,
        "sub_agent_calls_used": session.sub_agents.sub_agent_calls,
        "run_deadline": run_deadline,
        "chunks_read": session.sub_agents.chunks_read,
        "depth_reached": session.sub_agents.depth_reached,
    }


def _choose_after_runner(
    state: ResearchState,
) -> Literal["research_aggregator", "research_planner", "research_gave_up"]:
    """Aggregate any findings, otherwise take the next rung of the ladder."""
    if state["findings"]:
        return "research_aggregator"
    return "research_gave_up" if state["template_used"] else "research_planner"


def _build_report(
    themes: list[ResearchTheme], state: ResearchState, dependencies: AgentDependencies
) -> ResearchReport:
    """Assemble the report. Every number here is counted by code."""
    return ResearchReport(
        themes=themes,
        documents_analysed=len({finding.doc_id for finding in state["findings"]}),
        chunks_read=state["chunks_read"],
        chunks_available=dependencies.document_catalog.summarize(state["user"]).chunk_count,
        depth_reached=state["depth_reached"],
        sub_agent_calls=state["sub_agent_calls_used"],
    )


def _emit_report_ready(report: ResearchReport) -> None:
    """Show the themes and coverage, with doc ids so that a theme can be checked by hand."""
    emit_event(
        EventType.RESEARCH,
        stage=ResearchStage.REPORT_READY,
        themes=[
            {
                "name": theme.name,
                "document_count": theme.document_count,
                "doc_ids": sorted({finding.doc_id for finding in theme.findings}),
            }
            for theme in report.themes
        ],
        documents_analysed=report.documents_analysed,
        chunks_read=report.chunks_read,
        chunks_available=report.chunks_available,
        depth_reached=report.depth_reached,
        sub_agent_calls=report.sub_agent_calls,
    )


async def research_aggregator(
    state: ResearchState, runtime: Runtime[AgentDependencies]
) -> dict[str, Any]:
    """Group the findings into themes, and report the coverage of the run."""
    emit_node_started("research_aggregator")
    findings = state["findings"]

    # Step 1: the LLM proposes which findings belong together.
    theme_drafts, degraded = await propose_themes(
        latest_user_text(state), findings, runtime.context
    )

    # Step 2: code checks the groups, and does all the counting.
    themes = build_themes(theme_drafts, findings)
    report = _build_report(themes, state, runtime.context)

    # Step 3: report to the panel. `evidence` holds what the runner added in an earlier step.
    _emit_report_ready(report)
    emit_event(EventType.STATE, evidence_count=len(state.get("evidence", [])), degraded=degraded)
    emit_node_finished("research_aggregator")
    emit_node_finished(NODE_NAME)
    return {"research_report": report, "research_finished": True, "degraded": degraded}


async def research_gave_up(state: ResearchState) -> dict[str, Any]:
    """End with no report. The supervisor then runs plain retrieval instead."""
    emit_event(EventType.STATE, degraded=[DEGRADED_NO_FINDINGS])
    emit_node_finished(NODE_NAME)
    return {"research_finished": True, "degraded": [DEGRADED_NO_FINDINGS]}


def build_research_agent() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the compiled subgraph, which the main graph uses as one node."""
    graph = StateGraph(
        ResearchState,
        context_schema=AgentDependencies,
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    graph.add_node("check_research_access", check_research_access)
    graph.add_node("research_planner", research_planner)
    graph.add_node("research_runner", research_runner)
    graph.add_node("research_aggregator", research_aggregator)
    graph.add_node("research_gave_up", research_gave_up)

    graph.add_edge(START, "check_research_access")
    graph.add_conditional_edges(
        "check_research_access",
        _choose_after_access_check,
        {"research_planner": "research_planner", "stop": END},
    )
    graph.add_edge("research_planner", "research_runner")
    graph.add_conditional_edges("research_runner", _choose_after_runner)
    graph.add_edge("research_aggregator", END)
    graph.add_edge("research_gave_up", END)
    return graph.compile()

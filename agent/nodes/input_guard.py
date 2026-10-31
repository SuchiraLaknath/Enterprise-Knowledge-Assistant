"""`input_guard`: stop attacks on the assistant before anything else runs.

Patterns decide first, because they are instant and cannot be argued with. The LLM judge then
catches the same intent in new words. If no model can judge, the patterns decide alone and tools
are limited to reading, so that an outage cannot open a gap.
"""

from html import escape
from typing import Any, Literal

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError
from agent.nodes.supervisor import latest_user_text
from agent.prompts.input_guard import INPUT_JUDGE_REQUEST_TEMPLATE, INPUT_JUDGE_SYSTEM_PROMPT
from agent.schemas import InputJudgement
from agent.state import AgentState
from api.events import EventType, emit_event, emit_node_finished, emit_node_started
from security.input_guard import find_input_attack

logger = structlog.get_logger()

NODE_NAME = "input_guard"
JUDGE_ROUTE = "fast"
DEFAULT_ATTACK_CATEGORY = "attack"

DEGRADED_JUDGE_UNAVAILABLE = "guard_judge_unavailable_read_tools_only"
DEGRADED_MODEL_FALLBACK = "input_guard_used_fallback_model"


async def _ask_judge(message: str, dependencies: AgentDependencies) -> tuple[str | None, list[str]]:
    """Return the attack category the judge sees, or None, and any degradation."""
    # Rule 5: the message is judged as data, so that it cannot instruct the judge.
    messages: list[BaseMessage] = [
        SystemMessage(INPUT_JUDGE_SYSTEM_PROMPT),
        HumanMessage(INPUT_JUDGE_REQUEST_TEMPLATE.format(message=escape(message, quote=False))),
    ]
    judge_result = await dependencies.llm.generate_structured(JUDGE_ROUTE, messages, InputJudgement)
    judgement = judge_result.value
    # A judge can flag an attack without naming a kind, and that must still count as blocked.
    category = (judgement.category or DEFAULT_ATTACK_CATEGORY) if judgement.is_attack else None
    return category, judge_result.describe_fallbacks(DEGRADED_MODEL_FALLBACK)


async def input_guard(state: AgentState, runtime: Runtime[AgentDependencies]) -> dict[str, Any]:
    """Decide whether this message may continue."""
    emit_node_started(NODE_NAME)
    message = latest_user_text(state)

    # Step 1: rule 5 starts here. This is the main gate against direct prompt injection, and
    # known attacks need no model.
    attack_category = find_input_attack(message)
    degraded: list[str] = []
    read_only_tools = False

    # Step 2: the judge looks for what patterns cannot. If it is down, patterns stand alone.
    if attack_category is None:
        try:
            attack_category, degraded = await _ask_judge(message, runtime.context)
        except AllModelsFailedError:
            degraded = [DEGRADED_JUDGE_UNAVAILABLE]
            read_only_tools = True

    # Step 3: report the decision. The message is not logged, as it may hold personal data.
    blocked = attack_category is not None
    if blocked:
        logger.warning("input_blocked", category=attack_category)
    emit_event(
        EventType.VALIDATION,
        check=NODE_NAME,
        status="blocked" if blocked else "pass",
        reasons=[attack_category] if attack_category else [],
    )
    emit_event(EventType.STATE, degraded=degraded)
    emit_node_finished(NODE_NAME)
    return {"input_blocked": blocked, "read_only_tools": read_only_tools, "degraded": degraded}


def choose_after_input_guard(state: AgentState) -> Literal["refuse", "supervisor"]:
    """Send a blocked message to `refuse`, and anything else on to the supervisor."""
    return "refuse" if state["input_blocked"] else "supervisor"

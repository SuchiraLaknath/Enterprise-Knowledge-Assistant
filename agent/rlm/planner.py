"""Choose the Python plan for a research run.

The LLM writes the plan. If its code fails, it gets one more try with the error, and after that
a fixed template plan runs, so a poor model reply cannot stop an investigation. `plan_attempt`
counts through that ladder.

The planner sees the question and counts of what this employee may read, never document text,
so a poisoned document cannot influence the code.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.llm_gateway import AllModelsFailedError
from agent.prompts.research import (
    PLANNER_REQUEST_TEMPLATE,
    PLANNER_RETRY_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)
from agent.schemas import Plan, ResearchCode
from retrieval.models import UserContext

# Writing a good plan needs the same strong models as answering, so it shares their route.
PLANNER_ROUTE = "response"

DEGRADED_MODEL_FALLBACK = "research_planner_used_fallback_model"
DEGRADED_PLANNER_UNAVAILABLE = "research_planner_unavailable"

# Used when the LLM cannot write a working plan. It goes through the same sandbox checks, and it
# reads the employee's words from sandbox values, so that their text is never pasted into code.
# With a type or department to narrow by, the catalog lists the whole collection. A search alone
# would stop at its closest matches, and an answer could quietly miss part of the collection.
TEMPLATE_PLAN_CODE = """\
created_after = None
if recent_days:
    created_after = days_ago(recent_days)
records = search(search_query, document_types, departments, created_after)
doc_ids = sorted(group_by_doc(records))
if document_types or departments:
    documents = explore_documents(document_types, departments, created_after)
    doc_ids = [document["doc_id"] for document in documents]
sections = fetch_sections(doc_ids)
batch_size = len(sections) // MAX_BATCHES + 1
delegate(question, chunk(sections, batch_size))
"""


def is_template_attempt(plan_attempt: int, regeneration_limit: int) -> bool:
    """Return whether the LLM's attempts are used up: its first, plus each regeneration."""
    return plan_attempt > regeneration_limit


def _build_planner_messages(
    question: str,
    plan: Plan,
    user: UserContext,
    dependencies: AgentDependencies,
    previous_code: str,
    plan_error: str,
) -> list[BaseMessage]:
    """Assemble the planner's prompt from values that hold no document text."""
    limits = dependencies.research_limits
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        max_search_calls=limits.max_search_calls,
        sub_agent_chunk_limit=limits.sub_agent_chunk_limit,
    )
    # Rule 2: the counts come from the filtered catalog, so they reveal no hidden document.
    catalog_summary = dependencies.document_catalog.summarize(user)
    request_text = PLANNER_REQUEST_TEMPLATE.format(
        question=question,
        search_query=plan.search_query,
        departments=plan.departments,
        document_types=plan.document_types,
        recent_days=plan.recent_days,
        catalog_summary=catalog_summary.model_dump_json(indent=2),
    )
    if plan_error:
        request_text += PLANNER_RETRY_TEMPLATE.format(
            plan_error=plan_error, previous_code=previous_code
        )
    return [SystemMessage(system_prompt), HumanMessage(request_text)]


async def write_plan_code(
    question: str,
    plan: Plan,
    user: UserContext,
    dependencies: AgentDependencies,
    previous_code: str,
    plan_error: str,
) -> tuple[str | None, list[str]]:
    """Ask the LLM for plan code. Return None if no model can write one."""
    planner_messages = _build_planner_messages(
        question, plan, user, dependencies, previous_code, plan_error
    )
    try:
        planner_result = await dependencies.llm.generate_structured(
            PLANNER_ROUTE, planner_messages, ResearchCode
        )
    except AllModelsFailedError:
        return None, [DEGRADED_PLANNER_UNAVAILABLE]

    degraded = [
        f"{DEGRADED_MODEL_FALLBACK}:{model_spec}"
        for model_spec in planner_result.failed_model_specs
    ]
    return planner_result.value.python_code, degraded

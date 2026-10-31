"""Failure injection for research: each rung of the ladder, and one failure at a time.

One failed worker must not fail the turn, so every test here still ends with an answer.
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_agent_graph, create_checkpoint_serializer
from agent.llm_gateway import AllModelsFailedError
from agent.nodes.research_agent import DEGRADED_INCOMPLETE, DEGRADED_NO_FINDINGS
from agent.rlm.aggregate import DEGRADED_GROUPED_BY_LABEL
from agent.rlm.helpers import DEGRADED_BATCH_FAILED
from agent.rlm.planner import DEGRADED_PLANNER_UNAVAILABLE, TEMPLATE_PLAN_CODE
from agent.schemas import BatchAnalysis, ResearchCode, ThemeGrouping
from config.settings import SandboxLimits, get_limits
from retrieval.models import DegradedReason, SearchResult, UserContext
from tests.fakes.fake_dependencies import make_agent_dependencies
from tests.fakes.fake_llm import FakeLlm, make_plan
from tests.fakes.fake_research import (
    FakeResearchSearch,
    analyse_every_document,
    make_catalog,
    make_catalog_entry,
    make_research_code,
    make_section,
)

pytestmark = pytest.mark.faults

ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)
THREAD_CONFIG = {"configurable": {"thread_id": "research-fault-test"}}
INVALID_PLAN = make_research_code("import os")
WORKING_PLAN = make_research_code()
MODELS_FAILED = AllModelsFailedError("fast", ["fake:model"])


def make_llm(code_replies, batch_replies=analyse_every_document, grouping_replies=None):
    return FakeLlm(
        plans=[make_plan(needs_research=True)],
        replies={
            ResearchCode: code_replies,
            BatchAnalysis: batch_replies,
            ThemeGrouping: grouping_replies or [ThemeGrouping(themes=[])],
        },
    )


def make_dependencies(llm, degraded=(), sandbox_limits=None):
    doc_ids = ["INC-1", "INC-2"]
    search = FakeResearchSearch([make_section(doc_id) for doc_id in doc_ids], degraded)
    catalog = make_catalog(*[make_catalog_entry(doc_id) for doc_id in doc_ids])
    return make_agent_dependencies(llm, search, catalog, sandbox_limits=sandbox_limits)


async def run_turn(dependencies):
    graph = build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))
    events = []
    async for _subgraph_path, event in graph.astream(
        {"messages": [HumanMessage("Summarise the payment outages.")], "user": ANALYST},
        THREAD_CONFIG,
        context=dependencies,
        stream_mode="custom",
        subgraphs=True,
    ):
        events.append(event)
    state = await graph.aget_state(THREAD_CONFIG)
    return events, state.values


def limit_search_to_one_result(search):
    async def search_for_one_result(query_text, user, document_filters=None, **options):
        return SearchResult(chunks=search.chunks[:1], degraded=[])

    return search_for_one_result


def research_plans(events):
    return [
        event for event in events if event["type"] == "research" and event["stage"] == "plan_ready"
    ]


def started_nodes(events):
    return [
        event["node"]
        for event in events
        if event["type"] == "node" and event["status"] == "started"
    ]


async def test_an_invalid_plan_is_regenerated_with_its_error():
    llm = make_llm([INVALID_PLAN, WORKING_PLAN])

    events, state = await run_turn(make_dependencies(llm))

    assert [plan["attempt"] for plan in research_plans(events)] == [1, 2]
    assert state["research_report"] is not None
    # The second request tells the model what the sandbox refused.
    retry_messages = [
        messages for schema, messages in llm.received_structured_calls if schema is ResearchCode
    ][1]
    assert "Import is not allowed" in retry_messages[-1].text


async def test_a_second_invalid_plan_falls_back_to_the_template():
    llm = make_llm([INVALID_PLAN, INVALID_PLAN])

    events, state = await run_turn(make_dependencies(llm))

    assert [plan["source"] for plan in research_plans(events)] == ["llm", "llm", "template"]
    assert research_plans(events)[-1]["code"] == TEMPLATE_PLAN_CODE
    assert state["research_report"].documents_analysed == 2


async def test_the_template_reads_the_whole_collection_when_it_can_be_listed():
    # A live run found the template reading only what search returned: 13 of 22 incidents, in
    # an answer that sounded complete. Here search finds one document of three.
    doc_ids = ["INC-1", "INC-2", "INC-3"]
    search = FakeResearchSearch([make_section(doc_id) for doc_id in doc_ids])
    search.search = limit_search_to_one_result(search)
    catalog = make_catalog(*[make_catalog_entry(doc_id) for doc_id in doc_ids])
    llm = make_llm([INVALID_PLAN, INVALID_PLAN])
    llm.plans = [make_plan(needs_research=True, document_types=["incident"])]

    _, state = await run_turn(make_agent_dependencies(llm, search, catalog))

    assert state["research_report"].documents_analysed == 3


async def test_when_every_plan_fails_the_turn_uses_plain_retrieval():
    llm = make_llm(
        [INVALID_PLAN, INVALID_PLAN],
        batch_replies=lambda messages: BatchAnalysis(findings=[], unresolved_markers=[]),
    )

    events, state = await run_turn(make_dependencies(llm))

    assert "retrieval_agent" in started_nodes(events)
    assert state["research_report"] is None
    assert DEGRADED_NO_FINDINGS in state["degraded"]
    assert state["final_answer"]


async def test_a_plan_stuck_in_a_cpu_loop_is_stopped_and_replaced():
    # Attack: generated code that never finishes. An endless `for` needs no `while`.
    endless_plan = make_research_code("items = [1]\nfor item in items:\n    items.append(item)")
    llm = make_llm([endless_plan, WORKING_PLAN])
    fast_limit = SandboxLimits(cpu_seconds=0.05, max_range_items=10_000)

    events, state = await run_turn(make_dependencies(llm, sandbox_limits=fast_limit))

    assert any("SandboxTimeout" in event.get("error", "") for event in events)
    assert state["research_report"] is not None


async def test_a_planner_outage_goes_straight_to_the_template():
    # A retry would fail the same way, so the ladder skips to the plan that needs no model.
    llm = make_llm([AllModelsFailedError("response", ["fake:model"])])

    events, state = await run_turn(make_dependencies(llm))

    assert [plan["source"] for plan in research_plans(events)] == ["template"]
    assert DEGRADED_PLANNER_UNAVAILABLE in state["degraded"]
    assert state["research_report"] is not None


async def test_one_failed_sub_agent_does_not_fail_the_run():
    plan = make_research_code(
        """
sections = fetch_sections([document["doc_id"] for document in explore_documents()])
delegate("State the root cause.", chunk(sections, 1))
"""
    )

    def fail_for_the_first_document(messages):
        if "INC-1" in messages[-1].text:
            return MODELS_FAILED
        return analyse_every_document(messages)

    llm = make_llm([plan], batch_replies=fail_for_the_first_document)

    _, state = await run_turn(make_dependencies(llm))

    assert state["research_report"].documents_analysed == 1
    assert DEGRADED_BATCH_FAILED in state["degraded"]


async def test_findings_survive_a_plan_that_fails_partway():
    # The plan does useful work, and then crashes. That work must not be thrown away.
    plan = make_research_code(
        """
sections = fetch_sections([document["doc_id"] for document in explore_documents()])
delegate("State the root cause.", chunk(sections, 2))
first_item = sections[99]
"""
    )
    llm = make_llm([plan])

    _, state = await run_turn(make_dependencies(llm))

    assert state["research_report"].documents_analysed == 2
    assert DEGRADED_INCOMPLETE in state["degraded"]


async def test_an_aggregator_outage_groups_findings_by_their_labels():
    llm = make_llm([WORKING_PLAN], grouping_replies=[MODELS_FAILED])

    _, state = await run_turn(make_dependencies(llm))

    themes = state["research_report"].themes
    assert [(theme.name, theme.document_count) for theme in themes] == [("certificate expiry", 2)]
    assert DEGRADED_GROUPED_BY_LABEL in state["degraded"]


async def test_research_continues_when_pinecone_falls_back_to_local_search():
    llm = make_llm([WORKING_PLAN])
    degraded = [DegradedReason.PINECONE_UNAVAILABLE, DegradedReason.LOCAL_KEYWORD_SEARCH]

    _, state = await run_turn(make_dependencies(llm, degraded=degraded))

    assert state["research_report"].documents_analysed == 2
    assert DegradedReason.LOCAL_KEYWORD_SEARCH in state["degraded"]


async def test_the_sub_agent_budget_is_shared_across_plan_attempts():
    # Rule 7: the limit is per run, so a failing ladder cannot spend it three times over.
    wasteful_plan = make_research_code(
        """
sections = fetch_sections([document["doc_id"] for document in explore_documents()])
delegate("State the root cause.", chunk(sections, 1))
"""
    )
    limits = get_limits().research.model_copy(update={"max_sub_agent_calls": 2})
    llm = make_llm(
        [wasteful_plan, wasteful_plan],
        batch_replies=lambda messages: BatchAnalysis(findings=[], unresolved_markers=[]),
    )
    dependencies = make_dependencies(llm)
    dependencies = make_agent_dependencies(
        llm,
        dependencies.document_search,
        dependencies.document_catalog,
        research_limits=limits,
    )

    await run_turn(dependencies)

    sub_agent_calls = [call for call in llm.received_structured_calls if call[0] is BatchAnalysis]
    assert len(sub_agent_calls) == 2

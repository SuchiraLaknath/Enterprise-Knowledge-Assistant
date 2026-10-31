"""The research subgraph, run through the compiled graph with fakes."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_agent_graph, create_checkpoint_serializer
from agent.nodes.research_agent import DEGRADED_ROLE_DENIED, build_research_agent
from agent.nodes.supervisor import RESEARCH_LIMIT_NOTICE
from agent.rlm.sub_agents import DEGRADED_BUDGET_REACHED
from agent.schemas import (
    BatchAnalysis,
    ExtractedFinding,
    ResearchCode,
    ThemeDraft,
    ThemeGrouping,
)
from config.settings import get_limits
from retrieval.models import UserContext
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

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)
THREAD_CONFIG = {"configurable": {"thread_id": "research-test"}}
QUESTION = "Summarise all payment outages and find recurring root causes."
SOURCE_TEXT_MARKER = "UNIQUE-SOURCE-TEXT"


def make_research_llm(python_code=None, batch_replies=analyse_every_document, themes=None):
    plan_code = make_research_code(python_code) if python_code else make_research_code()
    grouping = ThemeGrouping(themes=themes or [])
    return FakeLlm(
        plans=[make_plan(needs_research=True), make_plan()],
        answers=["Certificates expired in the incidents [1].", "Second answer."],
        replies={
            ResearchCode: [plan_code, plan_code, plan_code],
            BatchAnalysis: batch_replies,
            ThemeGrouping: [grouping, grouping],
        },
    )


def make_incidents(count):
    doc_ids = [f"INC-{number}" for number in range(count)]
    chunks = [make_section(doc_id, text=f"{SOURCE_TEXT_MARKER} {doc_id}") for doc_id in doc_ids]
    entries = [make_catalog_entry(doc_id) for doc_id in doc_ids]
    return chunks, make_catalog(*entries)


async def run_turn(graph, dependencies, user=ANALYST, question=QUESTION):
    events = []
    async for _subgraph_path, event in graph.astream(
        {"messages": [HumanMessage(question)], "user": user},
        THREAD_CONFIG,
        context=dependencies,
        stream_mode="custom",
        subgraphs=True,
    ):
        events.append(event)
    state = await graph.aget_state(THREAD_CONFIG)
    return events, state.values


def started_nodes(events):
    return [
        event["node"]
        for event in events
        if event["type"] == "node" and event["status"] == "started"
    ]


def research_events(events, stage):
    return [event for event in events if event["type"] == "research" and event["stage"] == stage]


def calls_for_schema(llm, schema):
    return [call for call in llm.received_structured_calls if call[0] is schema]


@pytest.fixture
def graph():
    return build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))


async def test_an_analyst_question_runs_the_whole_research_flow_in_order(graph):
    chunks, catalog = make_incidents(4)
    llm = make_research_llm()
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    events, state = await run_turn(graph, dependencies)

    assert started_nodes(events) == [
        "load_context",
        "supervisor",
        "research_agent",
        "research_planner",
        "research_runner",
        "research_aggregator",
        "supervisor",
        "response_agent",
    ]
    assert state["research_report"].documents_analysed == 4
    assert len(calls_for_schema(llm, BatchAnalysis)) == 2
    assert [citation.chunk_id for citation in state["final_citations"]] == ["INC-0#c0"]


async def test_the_panel_sees_the_plan_batches_and_coverage(graph):
    chunks, catalog = make_incidents(4)
    dependencies = make_agent_dependencies(make_research_llm(), FakeResearchSearch(chunks), catalog)

    events, _ = await run_turn(graph, dependencies)

    assert "explore_documents" in research_events(events, "plan_ready")[0]["code"]
    assert research_events(events, "explored")[0]["document_count"] == 4
    assert research_events(events, "batches_planned")[0]["batch_sizes"] == [2, 2]
    report_event = research_events(events, "report_ready")[0]
    assert report_event["chunks_read"] == 4
    assert report_event["chunks_available"] == 4
    assert report_event["depth_reached"] == 1


async def test_a_viewer_gets_normal_retrieval_and_a_polite_note(graph):
    chunks, catalog = make_incidents(4)
    llm = make_research_llm()
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    events, state = await run_turn(graph, dependencies, user=VIEWER)

    assert "retrieval_agent" in started_nodes(events)
    assert "research_agent" not in started_nodes(events)
    assert calls_for_schema(llm, ResearchCode) == []
    assert calls_for_schema(llm, BatchAnalysis) == []
    assert state["final_answer"].endswith(RESEARCH_LIMIT_NOTICE)
    # Rule 9: the notice reaches the draft too, so a correct draft is not reported as replaced.
    draft = "".join(event["text"] for event in events if event["type"] == "token")
    assert draft == state["final_answer"]


async def test_the_subgraph_refuses_a_viewer_even_if_routing_was_bypassed():
    # Rule 2: simulates a routing bug that sends a viewer straight into the subgraph.
    chunks, catalog = make_incidents(2)
    llm = make_research_llm()
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    result = await build_research_agent().ainvoke(
        {"user": VIEWER, "plan": make_plan(needs_research=True), "messages": []},
        context=dependencies,
    )

    assert llm.received_structured_calls == []
    assert result["degraded"] == [DEGRADED_ROLE_DENIED]
    assert result.get("research_report") is None


async def test_a_batch_too_large_for_one_reader_moves_down_a_level(graph):
    chunks, catalog = make_incidents(10)
    plan = """
sections = fetch_sections([document["doc_id"] for document in explore_documents()])
delegate("State the root cause.", [sections])
"""
    llm = make_research_llm(plan)
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    _, state = await run_turn(graph, dependencies)

    assert state["research_report"].depth_reached == 2
    assert state["research_report"].sub_agent_calls == 2
    assert state["research_report"].documents_analysed == 10


async def test_a_sub_agent_can_ask_for_the_rest_of_a_document(graph):
    root_cause = make_section("INC-1", 0)
    lessons = make_section("INC-1", 1)
    catalog = make_catalog(make_catalog_entry("INC-1", chunk_count=2))

    def reply_for_batch(messages):
        # The first section does not hold the answer, so the sub-agent says so.
        if "Summary; Root cause" in messages[-1].text:
            return BatchAnalysis(findings=[], unresolved_markers=[1])
        return analyse_every_document(messages)

    llm = make_research_llm(batch_replies=reply_for_batch)
    search = FakeResearchSearch([root_cause, lessons])
    dependencies = make_agent_dependencies(llm, search, catalog)

    _, state = await run_turn(graph, dependencies)

    assert state["research_report"].depth_reached == 2
    assert state["research_report"].sub_agent_calls == 2
    assert [chunk.chunk_id for chunk in state["evidence"]] == ["INC-1#c1"]


async def test_recursion_stops_at_the_depth_limit(graph):
    # Every level keeps saying that the answer is elsewhere, which must not recurse forever.
    catalog = make_catalog(make_catalog_entry("INC-1", chunk_count=3))
    chunks = [make_section("INC-1", number) for number in range(3)]
    llm = make_research_llm(
        batch_replies=lambda messages: BatchAnalysis(findings=[], unresolved_markers=[1])
    )
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    events, _ = await run_turn(graph, dependencies)

    depths = [event["depth"] for event in research_events(events, "batch_finished")]
    assert max(depths) == get_limits().research.max_depth


async def test_the_sub_agent_budget_is_a_hard_limit(graph):
    chunks, catalog = make_incidents(8)
    limits = get_limits().research.model_copy(update={"max_sub_agent_calls": 2})
    llm = make_research_llm()
    dependencies = make_agent_dependencies(
        llm, FakeResearchSearch(chunks), catalog, research_limits=limits
    )

    _, state = await run_turn(graph, dependencies)

    assert len(calls_for_schema(llm, BatchAnalysis)) == 2
    assert DEGRADED_BUDGET_REACHED in state["degraded"]


async def test_evidence_from_the_subgraph_is_not_counted_twice(graph):
    # The additive reducer would double evidence if the subgraph returned its input as well.
    chunks, catalog = make_incidents(4)
    dependencies = make_agent_dependencies(make_research_llm(), FakeResearchSearch(chunks), catalog)

    _, state = await run_turn(graph, dependencies)

    chunk_ids = [chunk.chunk_id for chunk in state["evidence"]]
    assert len(chunk_ids) == 4
    assert len(set(chunk_ids)) == 4


async def test_the_subgraph_returns_only_what_it_added():
    # The supervisor can record degradation first. If the subgraph returned its whole input,
    # the additive reducers would count every earlier entry a second time.
    chunks, catalog = make_incidents(2)
    dependencies = make_agent_dependencies(make_research_llm(), FakeResearchSearch(chunks), catalog)

    result = await build_research_agent().ainvoke(
        {
            "user": ANALYST,
            "plan": make_plan(needs_research=True),
            "messages": [HumanMessage(QUESTION)],
            "evidence": [make_section("EARLIER")],
            "degraded": ["earlier_problem"],
        },
        context=dependencies,
    )

    assert "earlier_problem" not in result["degraded"]
    assert "EARLIER" not in [chunk.doc_id for chunk in result["evidence"]]


async def test_research_from_an_earlier_turn_cannot_be_cited(graph):
    # Rule 4: the next turn must start with neither the report nor its evidence.
    chunks, catalog = make_incidents(2)
    search = FakeResearchSearch(chunks)
    dependencies = make_agent_dependencies(make_research_llm(), search, catalog)
    await run_turn(graph, dependencies)
    search.chunks = []

    _, state = await run_turn(graph, dependencies, question="Hello again")

    assert state["research_report"] is None
    assert state["evidence"] == []
    assert state["final_citations"] == []


async def test_a_made_up_marker_never_becomes_evidence(graph):
    # Rule 4: the model cites document 99 in a batch of 2, and document 1 for real.
    chunks, catalog = make_incidents(2)

    def reply_with_an_invented_marker(messages):
        return BatchAnalysis(
            findings=[
                ExtractedFinding(claim="Invented.", theme_label="x", evidence_markers=[99]),
                ExtractedFinding(claim="Real.", theme_label="x", evidence_markers=[1]),
            ],
            unresolved_markers=[99],
        )

    llm = make_research_llm(batch_replies=reply_with_an_invented_marker)
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    _, state = await run_turn(graph, dependencies)

    claims = [
        finding.claim for theme in state["research_report"].themes for finding in theme.findings
    ]
    assert claims == ["Real."]
    assert [chunk.chunk_id for chunk in state["evidence"]] == ["INC-0#c0"]


async def test_counts_come_from_code_and_not_from_the_model(graph):
    chunks, catalog = make_incidents(4)
    themes = [ThemeDraft(name="Expiry", summary="Expired.", finding_ids=["F1", "F2", "F99"])]
    dependencies = make_agent_dependencies(
        make_research_llm(themes=themes), FakeResearchSearch(chunks), catalog
    )

    _, state = await run_turn(graph, dependencies)

    report = state["research_report"]
    assert [(theme.name, theme.document_count) for theme in report.themes] == [
        ("Expiry", 2),
        ("Other findings", 2),
    ]


async def test_the_answer_uses_findings_and_never_reloads_document_text(graph):
    # The brief: entire documents must not be loaded into the LLM context.
    chunks, catalog = make_incidents(4)
    llm = make_research_llm()
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    await run_turn(graph, dependencies)

    answer_prompt = "\n".join(str(message.content) for message in llm.received_answer_messages[0])
    assert "Finding from document 1." in answer_prompt
    assert SOURCE_TEXT_MARKER not in answer_prompt


async def test_the_planner_never_sees_document_text(graph):
    # A poisoned document must not be able to shape the code that the sandbox runs.
    chunks, catalog = make_incidents(2)
    llm = make_research_llm()
    dependencies = make_agent_dependencies(llm, FakeResearchSearch(chunks), catalog)

    await run_turn(graph, dependencies)

    planner_messages = calls_for_schema(llm, ResearchCode)[0][1]
    planner_prompt = "\n".join(str(message.content) for message in planner_messages)
    assert SOURCE_TEXT_MARKER not in planner_prompt


async def test_plans_cannot_read_a_document_above_the_users_level(graph):
    # Rule 2: the plan names a restricted document directly, and a search also returns it.
    open_chunk = make_section("OPEN")
    secret_chunk = make_section("SECRET", access_level="restricted", text="TOP-SECRET-TEXT")
    catalog = make_catalog(
        make_catalog_entry("OPEN"), make_catalog_entry("SECRET", access_level="restricted")
    )
    plan = """
sections = fetch_sections(["OPEN", "SECRET"])
delegate("State the root cause.", [sections])
"""
    llm = make_research_llm(plan)
    search = FakeResearchSearch([open_chunk, secret_chunk])
    dependencies = make_agent_dependencies(llm, search, catalog)

    _, state = await run_turn(graph, dependencies)

    assert search.fetched_chunk_ids == ["OPEN#c0"]
    assert [chunk.doc_id for chunk in state["evidence"]] == ["OPEN"]
    sub_agent_prompts = str(calls_for_schema(llm, BatchAnalysis))
    assert "TOP-SECRET-TEXT" not in sub_agent_prompts

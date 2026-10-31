"""The tool agent, run through the compiled graph with fakes."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_agent_graph, create_checkpoint_serializer
from agent.llm_gateway import AllModelsFailedError, ToolCallRequest
from agent.nodes.tool_agent import DEGRADED_NO_TOOL_MODEL, DEGRADED_TOOL_FAILED
from config.settings import get_limits
from retrieval.models import SearchResult, UserContext
from tests.fakes.fake_dependencies import FakeMcpClient, make_agent_dependencies
from tests.fakes.fake_llm import FakeLlm, make_plan
from tests.fakes.fake_search import make_chunk
from tools.audit_log import InMemoryAuditLog

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)
THREAD_CONFIG = {"configurable": {"thread_id": "tool-test"}}
TOOL_PLAN = make_plan(needs_documents=False, needs_tools=True)
PLAIN_ANSWER = "Asha Fernando owns the Card Switch."
MCP_DATA = {"service_catalog": [{"name": "Card Switch", "owner": "Asha Fernando"}]}


class SearchReturning:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)

    async def search(self, query_text, user, document_filters=None, namespaces=None, **options):
        return SearchResult(chunks=self.chunks)


def tool_call(name, arguments=None, call_id="call-1"):
    return ToolCallRequest(call_id, name, arguments or {})


async def run_turn(dependencies, user=ANALYST):
    graph = build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))
    events = []
    async for _subgraph_path, event in graph.astream(
        {"messages": [HumanMessage("Who owns the Card Switch?")], "user": user},
        THREAD_CONFIG,
        context=dependencies,
        stream_mode="custom",
        subgraphs=True,
    ):
        events.append(event)
    state = await graph.aget_state(THREAD_CONFIG)
    return events, state.values


def make_dependencies(llm, **options):
    options.setdefault("mcp_client", FakeMcpClient(MCP_DATA))
    return make_agent_dependencies(llm, SearchReturning(), **options)


def started_nodes(events):
    return [
        event["node"]
        for event in events
        if event["type"] == "node" and event["status"] == "started"
    ]


def tool_events(events):
    return [event for event in events if event["type"] == "tool"]


async def test_a_tool_question_runs_the_tool_and_answers_from_it():
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[TOOL_PLAN],
        tool_call_rounds=[[tool_call("mcp__service_catalog")], []],
    )

    events, state = await run_turn(make_dependencies(llm))

    assert started_nodes(events) == [
        "load_context",
        "input_guard",
        "supervisor",
        "tool_agent",
        "supervisor",
        "response_agent",
        "validator",
    ]
    assert tool_events(events)[0]["tool_name"] == "mcp.service_catalog"
    assert tool_events(events)[0]["status"] == "ok"
    assert "Asha Fernando" in str(llm.received_answer_messages[0])


async def test_the_model_is_only_offered_tools_for_its_role():
    # Rule 2's first check, seen from what really reaches the model.
    llm = FakeLlm(answers=[PLAIN_ANSWER], plans=[TOOL_PLAN], tool_call_rounds=[[]])

    await run_turn(make_dependencies(llm), user=VIEWER)

    offered = {schema["function"]["name"] for schema in llm.received_tool_schemas[0]}
    assert offered == {"knowledge_search"}


async def test_a_tool_the_model_was_never_offered_is_denied_and_audited():
    # Attack: an injected model calls an admin tool by name anyway.
    audit_log = InMemoryAuditLog()
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[TOOL_PLAN],
        tool_call_rounds=[[tool_call("admin__view_audit_log")], []],
    )

    events, state = await run_turn(make_dependencies(llm, audit_log=audit_log), user=VIEWER)

    assert tool_events(events)[0]["status"] == "denied"
    assert audit_log.entries[0]["outcome"] == "denied"
    assert state["final_answer"]


async def test_the_verified_user_reaches_the_mcp_client():
    # Rule 2: the role comes from the token, and never from the model's arguments.
    mcp_client = FakeMcpClient(MCP_DATA)
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[TOOL_PLAN],
        tool_call_rounds=[[tool_call("mcp__service_catalog")], []],
    )

    await run_turn(make_dependencies(llm, mcp_client=mcp_client))

    assert mcp_client.calls[0]["user"] == ANALYST
    assert "role" not in mcp_client.calls[0]["arguments"]


async def test_tool_search_results_become_citable_evidence():
    chunk = make_chunk("DOC#c0")
    llm = FakeLlm(
        plans=[TOOL_PLAN],
        answers=["The cause was a timeout [1]."],
        tool_call_rounds=[[tool_call("knowledge_search", {"query": "timeout"})], []],
    )
    dependencies = make_agent_dependencies(llm, SearchReturning([chunk]))

    _, state = await run_turn(dependencies)

    assert [citation.chunk_id for citation in state["final_citations"]] == ["DOC#c0"]


async def test_analysis_sees_metadata_and_never_document_text():
    # Rule 7: `records` must not carry a chunk's text into generated code.
    chunk = make_chunk("DOC#c0", text="UNIQUE-SOURCE-TEXT")
    code = "result = records"
    plan = make_plan(needs_documents=True, needs_tools=True)
    llm = FakeLlm(
        plans=[plan], tool_call_rounds=[[tool_call("python_analysis", {"code": code})], []]
    )
    dependencies = make_agent_dependencies(llm, SearchReturning([chunk]))

    _, state = await run_turn(dependencies)

    assert "DOC#c0" in state["tool_results"][0].data
    assert "UNIQUE-SOURCE-TEXT" not in state["tool_results"][0].data


async def test_search_is_not_offered_again_after_documents_were_searched():
    # A live run called knowledge_search three times in a row instead of analysing the results.
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[make_plan(needs_documents=True, needs_tools=True)],
        tool_call_rounds=[[]],
    )

    await run_turn(make_agent_dependencies(llm, SearchReturning()))

    offered = {schema["function"]["name"] for schema in llm.received_tool_schemas[0]}
    assert "knowledge_search" not in offered
    assert "python_analysis" in offered


async def test_a_viewer_with_no_tools_left_makes_no_tool_call():
    llm = FakeLlm(answers=[PLAIN_ANSWER], plans=[make_plan(needs_documents=True, needs_tools=True)])

    await run_turn(make_agent_dependencies(llm, SearchReturning()), user=VIEWER)

    assert llm.received_tool_schemas == []


async def test_tool_rounds_are_bounded():
    # Attack: a model that keeps asking for one more tool must not loop forever.
    rounds = [[tool_call("mcp__service_catalog")] for _ in range(20)]
    llm = FakeLlm(answers=[PLAIN_ANSWER], plans=[TOOL_PLAN], tool_call_rounds=rounds)

    await run_turn(make_dependencies(llm))

    assert len(llm.received_tool_schemas) == get_limits().tools.max_rounds


async def test_tool_calls_per_turn_are_bounded():
    many_calls = [tool_call("mcp__service_catalog") for _ in range(50)]
    llm = FakeLlm(
        answers=[PLAIN_ANSWER], plans=[TOOL_PLAN], tool_call_rounds=[many_calls, many_calls]
    )

    events, _ = await run_turn(make_dependencies(llm))

    assert len(tool_events(events)) == get_limits().tools.max_calls_per_turn


async def test_tool_results_from_an_earlier_turn_are_gone():
    graph = build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))
    llm = FakeLlm(
        plans=[TOOL_PLAN, make_plan(needs_documents=False)],
        answers=["First.", "Second."],
        tool_call_rounds=[[tool_call("mcp__service_catalog")], []],
    )
    dependencies = make_dependencies(llm)
    for question in ("Who owns it?", "Thanks"):
        async for _ in graph.astream(
            {"messages": [HumanMessage(question)], "user": ANALYST},
            THREAD_CONFIG,
            context=dependencies,
            stream_mode="custom",
            subgraphs=True,
        ):
            pass

    state = await graph.aget_state(THREAD_CONFIG)

    assert state.values["tool_results"] == []


@pytest.mark.faults
async def test_an_mcp_outage_still_gives_an_answer():
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[TOOL_PLAN],
        tool_call_rounds=[[tool_call("mcp__service_catalog")], []],
    )
    dependencies = make_dependencies(llm, mcp_client=FakeMcpClient(fails=True))

    events, state = await run_turn(dependencies)

    assert tool_events(events)[0]["status"] == "error"
    assert DEGRADED_TOOL_FAILED in state["degraded"]
    assert state["final_answer"]


@pytest.mark.faults
async def test_a_tool_timeout_fault_still_gives_an_answer():
    llm = FakeLlm(
        answers=[PLAIN_ANSWER],
        plans=[TOOL_PLAN],
        tool_call_rounds=[[tool_call("mcp__service_catalog")], []],
    )
    dependencies = make_dependencies(llm, enabled_faults=("tool_timeout",))

    events, state = await run_turn(dependencies)

    assert tool_events(events)[0]["status"] == "timeout"
    assert DEGRADED_TOOL_FAILED in state["degraded"]
    assert state["final_answer"]


@pytest.mark.faults
async def test_no_model_for_tool_selection_still_gives_an_answer():
    outage = AllModelsFailedError("fast", ["fake:model"])
    llm = FakeLlm(answers=[PLAIN_ANSWER], plans=[TOOL_PLAN], tool_call_rounds=[outage])

    _, state = await run_turn(make_dependencies(llm))

    assert DEGRADED_NO_TOOL_MODEL in state["degraded"]
    assert state["final_answer"]

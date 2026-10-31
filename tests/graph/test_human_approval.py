"""Write tools and human approval, through the compiled graph and the API route."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_agent_graph, create_checkpoint_serializer
from agent.llm_gateway import ToolCallRequest
from agent.schemas import ApprovalDecision, InputJudgement
from retrieval.models import UserContext
from tests.fakes.fake_dependencies import FakeMcpClient, make_agent_dependencies
from tests.fakes.fake_llm import FakeLlm, make_plan
from tests.fakes.fake_research import FakeResearchSearch
from tools.audit_log import InMemoryAuditLog

ADMIN = UserContext(
    user_id="admin",
    role="admin",
    allowed_access_levels=("public", "internal", "confidential", "restricted"),
)
ANALYST = UserContext(user_id="nadee", role="analyst", allowed_access_levels=("internal",))
THREAD_CONFIG = {"configurable": {"thread_id": "approval-test"}}
TOOL_PLAN = make_plan(needs_documents=False, needs_tools=True)
ARGUMENTS = {"incident_id": "LIVE-1041", "status": "resolved"}
WRITE_CALL = ToolCallRequest("call-7", "admin__update_incident_status", ARGUMENTS)


def make_setup(rounds=None, judge_replies=None):
    llm = FakeLlm(
        plans=[TOOL_PLAN],
        answers=["Done."],
        tool_call_rounds=rounds or [[WRITE_CALL]],
        replies=judge_replies or {},
    )
    mcp_client = FakeMcpClient({"update_incident_status": {"updated": True}})
    audit_log = InMemoryAuditLog()
    dependencies = make_agent_dependencies(
        llm, FakeResearchSearch(), mcp_client=mcp_client, audit_log=audit_log
    )
    graph = build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))
    return graph, dependencies, mcp_client, audit_log


async def start_turn(graph, dependencies, user=ADMIN):
    async for _ in graph.astream(
        {"messages": [HumanMessage("Mark LIVE-1041 as resolved.")], "user": user},
        THREAD_CONFIG,
        context=dependencies,
        stream_mode="custom",
        subgraphs=True,
    ):
        pass
    return await graph.aget_state(THREAD_CONFIG)


async def resume_turn(graph, dependencies, decision):
    events = []
    async for _subgraph_path, event in graph.astream(
        Command(resume=decision),
        THREAD_CONFIG,
        context=dependencies,
        stream_mode="custom",
        subgraphs=True,
    ):
        events.append(event)
    return events, await graph.aget_state(THREAD_CONFIG)


async def test_a_write_tool_pauses_before_anything_changes():
    # Rule 3: the model asked to change data, and nothing may happen until a person agrees.
    graph, dependencies, mcp_client, _ = make_setup()

    state = await start_turn(graph, dependencies)

    assert state.next == ("human_approval",)
    assert state.values["pending_approval"].arguments == ARGUMENTS
    assert mcp_client.calls == []


async def test_approval_runs_the_stored_arguments_once():
    graph, dependencies, mcp_client, _ = make_setup()
    await start_turn(graph, dependencies)

    _, state = await resume_turn(
        graph, dependencies, ApprovalDecision(tool_call_id="call-7", approved=True)
    )

    assert [call["arguments"] for call in mcp_client.calls] == [ARGUMENTS]
    assert state.values["pending_approval"] is None
    assert state.values["final_answer"] == "Done."


async def test_rejection_changes_nothing_and_is_audited():
    graph, dependencies, mcp_client, audit_log = make_setup()
    await start_turn(graph, dependencies)

    _, state = await resume_turn(
        graph, dependencies, ApprovalDecision(tool_call_id="call-7", approved=False)
    )

    assert mcp_client.calls == []
    assert audit_log.entries[0]["outcome"] == "rejected"
    assert state.values["tool_results"][0].status == "denied"


async def test_an_approval_for_a_different_call_runs_nothing():
    # Attack: an approval for one call must not unlock another.
    graph, dependencies, mcp_client, _ = make_setup()
    await start_turn(graph, dependencies)

    await resume_turn(
        graph, dependencies, ApprovalDecision(tool_call_id="call-other", approved=True)
    )

    assert mcp_client.calls == []


async def test_the_tool_cannot_run_twice_if_the_turn_is_resumed_again():
    # LangGraph re-runs a node from its start, so `human_approval` must do nothing but pause.
    graph, dependencies, mcp_client, _ = make_setup()
    await start_turn(graph, dependencies)
    decision = ApprovalDecision(tool_call_id="call-7", approved=True)

    await resume_turn(graph, dependencies, decision)
    await resume_turn(graph, dependencies, decision)

    assert len(mcp_client.calls) == 1


async def test_writes_are_not_offered_when_the_guard_judge_is_down():
    # With only deterministic guards standing, nothing may be changed.
    from agent.llm_gateway import AllModelsFailedError

    outage = AllModelsFailedError("fast", ["fake:model"])
    graph, dependencies, _, _ = make_setup(judge_replies={InputJudgement: [outage]})

    await start_turn(graph, dependencies)

    offered = {schema["function"]["name"] for schema in dependencies.llm.received_tool_schemas[0]}
    assert "admin__update_incident_status" not in offered


@pytest.mark.parametrize("status_value", ["deleted", "", "resolved; DROP TABLE"])
async def test_an_invalid_write_argument_is_refused(status_value):
    # Attack: tool abuse through a write tool's arguments, even after approval.
    arguments = {"incident_id": "LIVE-1041", "status": status_value}
    call = ToolCallRequest("call-7", "admin__update_incident_status", arguments)
    graph, dependencies, mcp_client, _ = make_setup(rounds=[[call]])
    await start_turn(graph, dependencies)

    await resume_turn(graph, dependencies, ApprovalDecision(tool_call_id="call-7", approved=True))

    assert mcp_client.calls == []


async def test_a_write_tool_that_was_not_offered_is_denied_and_never_pauses():
    # A review found that a model could name a write tool it was not offered, and so pause a
    # read-only turn, or show a person an approval that their role could never give.
    graph, dependencies, mcp_client, audit_log = make_setup()

    state = await start_turn(graph, dependencies, user=ANALYST)

    assert state.next == ()
    assert mcp_client.calls == []
    assert audit_log.entries[-1]["outcome"] == "denied"


async def test_a_change_that_cannot_be_audited_does_not_happen():
    # Rule 2: during a Postgres outage a write would otherwise leave no record.
    class FailingAuditLog(InMemoryAuditLog):
        async def record(self, user, tool_name, outcome, detail):
            await super().record(user, tool_name, outcome, detail)
            return False

    llm = FakeLlm(plans=[TOOL_PLAN], answers=["Done."], tool_call_rounds=[[WRITE_CALL]])
    mcp_client = FakeMcpClient()
    dependencies = make_agent_dependencies(
        llm, FakeResearchSearch(), mcp_client=mcp_client, audit_log=FailingAuditLog()
    )
    graph = build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer()))
    await start_turn(graph, dependencies)

    await resume_turn(graph, dependencies, ApprovalDecision(tool_call_id="call-7", approved=True))

    assert mcp_client.calls == []

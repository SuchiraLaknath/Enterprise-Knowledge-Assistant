import asyncio

import pytest
from pydantic import Field

from config.settings import get_limits, get_role_access
from retrieval.models import SearchResult, UserContext
from tests.fakes.fake_dependencies import FakeMcpClient
from tests.fakes.fake_search import make_chunk
from tools.audit_log import InMemoryAuditLog
from tools.models import ToolArguments, ToolContext, ToolResult
from tools.registry import ToolDefinition, ToolRegistry, build_tool_registry

LIMITS = get_limits()
ROLE_RANKING = get_role_access().role_ranking
VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)
ADMIN = UserContext(
    user_id="admin",
    role="admin",
    allowed_access_levels=("public", "internal", "confidential", "restricted"),
)


class RecordingSearch:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    async def search(self, query_text, user, document_filters=None, namespaces=None, **options):
        self.calls.append({"user": user, "filters": document_filters, **options})
        return SearchResult(chunks=self.chunks)


def make_context(user, audit_log=None, search=None, mcp_client=None, records=()):
    return ToolContext(
        user=user,
        document_search=search or RecordingSearch(),
        audit_log=audit_log or InMemoryAuditLog(),
        mcp_client=mcp_client or FakeMcpClient(),
        sandbox_limits=LIMITS.sandbox,
        tool_limits=LIMITS.tools,
        analysis_records=list(records),
    )


def make_registry(enabled_faults=()):
    return build_tool_registry(LIMITS.tools, ROLE_RANKING, enabled_faults)


class WriteArguments(ToolArguments):
    value: str = Field(default="x")


async def fake_write(arguments, context):
    return ToolResult(tool_name="admin.write", status="ok")


def make_write_definition():
    return ToolDefinition(
        name="admin.write",
        description="Change something.",
        args_schema=WriteArguments,
        min_role="admin",
        risk="write",
        timeout_s=1.0,
        handler=fake_write,
    )


def offered_names(role):
    return {definition.name for definition in make_registry().tools_for_model(role)}


def test_each_role_is_offered_only_its_own_tools():
    # Rule 2's first check, from the Roles table in CLAUDE.md.
    assert offered_names("viewer") == {"knowledge_search"}
    assert offered_names("analyst") == {
        "knowledge_search",
        "python_analysis",
        "mcp.employee_directory",
        "mcp.service_catalog",
        "mcp.incident_records",
    }
    assert "admin.view_audit_log" in offered_names("admin")


def test_an_unknown_role_is_offered_nothing():
    assert offered_names("superuser") == set()


def test_research_is_never_offered_to_a_model():
    # Routing starts research, so a model cannot start it with a tool call.
    assert "deep_research" not in offered_names("admin")
    assert make_registry().is_role_allowed("deep_research", "analyst")
    assert not make_registry().is_role_allowed("deep_research", "viewer")


async def test_a_viewer_calling_an_admin_tool_is_denied_and_audited():
    # Attack: an injected model names a tool that it was never offered.
    audit_log = InMemoryAuditLog()

    result = await make_registry().execute(
        "admin.view_audit_log", {}, make_context(VIEWER, audit_log)
    )

    assert result.status == "denied"
    assert audit_log.entries[0]["outcome"] == "denied"
    assert audit_log.entries[0]["user_id"] == "vimal"


async def test_a_viewer_calling_an_mcp_tool_never_reaches_the_server():
    mcp_client = FakeMcpClient()

    result = await make_registry().execute(
        "mcp.incident_records", {}, make_context(VIEWER, mcp_client=mcp_client)
    )

    assert result.status == "denied"
    assert mcp_client.calls == []


async def test_a_made_up_tool_is_denied_and_audited():
    audit_log = InMemoryAuditLog()

    result = await make_registry().execute("os.system", {}, make_context(ADMIN, audit_log))

    assert result.status == "denied"
    assert len(audit_log.entries) == 1
    # A model chose that name, so the log holds a fixed value rather than its text.
    assert audit_log.entries[0]["tool_name"] == "unknown_tool"


@pytest.mark.parametrize(
    "abuse",
    [
        {"query": "outage", "access_level": "restricted"},
        {"query": "outage", "role": "admin"},
        {"query": "outage", "user_id": "admin"},
        {"query": "outage", "allowed_access_levels": ["restricted"]},
        {"query": ""},
        {"query": "x" * 501},
        {},
    ],
)
async def test_tool_abuse_arguments_are_refused(abuse):
    # Attack: rule 1, a model tries to widen access through a tool argument.
    search = RecordingSearch()

    result = await make_registry().execute(
        "knowledge_search", abuse, make_context(VIEWER, search=search)
    )

    assert result.status == "error"
    assert search.calls == []


async def test_knowledge_search_uses_the_verified_user_and_skips_the_reranker():
    chunk = make_chunk("DOC#c0")
    search = RecordingSearch([chunk])

    result = await make_registry().execute(
        "knowledge_search", {"query": "outage"}, make_context(VIEWER, search=search)
    )

    assert result.status == "ok"
    assert search.calls[0]["user"] == VIEWER
    assert search.calls[0]["use_reranker"] is False
    assert result.evidence == [chunk]
    assert "payment gateway timeout" not in str(result.data)


async def test_a_write_tool_never_runs_without_approval():
    # Rule 3: even an administrator's model cannot change anything by itself.
    registry = ToolRegistry([make_write_definition()], ROLE_RANKING)
    audit_log = InMemoryAuditLog()

    denied = await registry.execute("admin.write", {}, make_context(ADMIN, audit_log))
    approved = await registry.execute(
        "admin.write", {}, make_context(ADMIN, audit_log), approved=True
    )

    assert denied.status == "denied"
    assert approved.status == "ok"


async def test_a_slow_tool_returns_a_timeout_as_data():
    async def never_finish(arguments, context):
        await asyncio.sleep(10)

    definition = ToolDefinition(
        name="slow",
        description="Slow.",
        args_schema=WriteArguments,
        min_role="viewer",
        risk="read",
        timeout_s=0.01,
        handler=never_finish,
    )

    result = await ToolRegistry([definition], ROLE_RANKING).execute(
        "slow", {}, make_context(VIEWER)
    )

    assert result.status == "timeout"


async def test_a_crashing_tool_returns_an_error_with_no_details():
    # An exception's text can hold internal details, so none of it reaches a prompt.
    async def crash(arguments, context):
        raise RuntimeError("password=hunter2")

    definition = ToolDefinition(
        name="crash",
        description="Crash.",
        args_schema=WriteArguments,
        min_role="viewer",
        risk="read",
        timeout_s=1.0,
        handler=crash,
    )

    result = await ToolRegistry([definition], ROLE_RANKING).execute(
        "crash", {}, make_context(VIEWER)
    )

    assert result.status == "error"
    assert "hunter2" not in result.model_dump_json()


def test_every_tool_meets_rule_3():
    definitions = make_registry()._tools.values()

    for definition in definitions:
        assert definition.args_schema is not None
        assert definition.min_role in ROLE_RANKING
        assert definition.risk in ("read", "write")
        assert definition.timeout_s > 0
        # Rule 1: no args schema may have a field for identity or access.
        field_names = set(definition.args_schema.model_fields)
        assert not field_names & {"role", "user", "user_id", "access_level"}


def test_model_facing_names_contain_no_dots():
    # Provider APIs refuse dots in tool names, so a dotted name would fail every tool call.
    schemas = [item.to_model_schema() for item in make_registry().tools_for_model("admin")]
    names = [schema["function"]["name"] for schema in schemas]

    assert "admin__view_audit_log" in names
    assert not any("." in name for name in names)
    assert make_registry().resolve_model_name("admin__view_audit_log") == "admin.view_audit_log"


async def test_an_admin_can_read_the_audit_log():
    audit_log = InMemoryAuditLog()
    await audit_log.record(VIEWER, "knowledge_search", "ok", "")

    result = await make_registry().execute(
        "admin.view_audit_log", {"entry_count": 5}, make_context(ADMIN, audit_log)
    )

    assert result.status == "ok"
    assert result.data[0]["tool_name"] == "knowledge_search"


async def test_python_analysis_counts_records():
    records = [{"department": "payments"}, {"department": "payments"}, {"department": "hr"}]
    code = 'result = count_by(records, "department")'

    result = await make_registry().execute(
        "python_analysis", {"code": code}, make_context(ANALYST, records=records)
    )

    assert result.status == "ok"
    assert '"payments": 2' in result.data


@pytest.mark.parametrize(
    "attack",
    [
        "import os\nresult = 1",
        "result = ().__class__",
        "result = '{0.__class__}'.format(1)",
        "result = search('outage')",
        "result = fetch_sections(['SECRET'])",
        "result = explore_documents()",
        "result = delegate('task', [])",
    ],
)
async def test_python_analysis_refuses_attacks_and_has_no_retrieval(attack):
    # Rule 7: this tool gets data-only helpers, so it cannot read anything new.
    result = await make_registry().execute(
        "python_analysis", {"code": attack}, make_context(ANALYST)
    )

    assert result.status == "error"


async def test_python_analysis_needs_a_result():
    result = await make_registry().execute(
        "python_analysis", {"code": "total = 1"}, make_context(ANALYST)
    )

    assert result.status == "error"
    assert "result" in result.message


async def test_a_large_analysis_result_is_cut_to_the_limit():
    code = 'result = ["x" for item in range(5000)]'

    result = await make_registry().execute("python_analysis", {"code": code}, make_context(ANALYST))

    assert len(result.data) == LIMITS.tools.max_result_characters

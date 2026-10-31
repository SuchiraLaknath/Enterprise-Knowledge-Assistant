"""The services a graph test needs, so that each test names only what it changes."""

from agent.dependencies import AgentDependencies
from config.settings import get_limits, get_role_access
from retrieval.catalog import DocumentCatalog
from tools.audit_log import InMemoryAuditLog
from tools.registry import build_tool_registry


class FakeLongTermMemory:
    """Keeps memories per user, and can be told to fail."""

    def __init__(self, fails=False):
        self.memories = {}
        self.fails = fails

    async def read(self, user_id, limit):
        if self.fails:
            raise ConnectionError("Simulated store outage")
        return self.memories.get(user_id, [])[:limit]

    async def write(self, user_id, fact, doc_ids):
        if self.fails:
            raise ConnectionError("Simulated store outage")
        self.memories.setdefault(user_id, []).append({"fact": fact, "doc_ids": doc_ids})


class FakeMcpClient:
    """Returns preset data per tool, can be told to fail, and records who called it."""

    def __init__(self, data=None, fails=False):
        self.data = data or {}
        self.fails = fails
        self.calls = []

    async def call_tool(self, tool_name, arguments, user, timeout_s):
        self.calls.append({"tool_name": tool_name, "arguments": arguments, "user": user})
        if self.fails:
            raise ConnectionError("Simulated MCP failure")
        return self.data.get(tool_name, [])


def make_agent_dependencies(
    llm,
    search,
    catalog=None,
    research_limits=None,
    sandbox_limits=None,
    audit_log=None,
    mcp_client=None,
    enabled_faults=(),
    long_term_memory=None,
    conversation_limits=None,
):
    limits = get_limits()
    return AgentDependencies(
        llm=llm,
        document_search=search,
        document_catalog=catalog or DocumentCatalog([]),
        history_message_count=limits.conversation.history_message_count,
        conversation_limits=conversation_limits or limits.conversation,
        long_term_memory=long_term_memory or FakeLongTermMemory(),
        research_limits=research_limits or limits.research,
        sandbox_limits=sandbox_limits or limits.sandbox,
        tool_limits=limits.tools,
        validation_limits=limits.validation,
        tool_registry=build_tool_registry(
            limits.tools, get_role_access().role_ranking, enabled_faults
        ),
        audit_log=audit_log or InMemoryAuditLog(),
        mcp_client=mcp_client or FakeMcpClient(),
    )

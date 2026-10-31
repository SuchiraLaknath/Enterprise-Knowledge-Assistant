"""The services a graph test needs, so that each test names only what it changes."""

from agent.dependencies import AgentDependencies
from config.settings import get_limits, get_role_access
from retrieval.catalog import DocumentCatalog


def make_agent_dependencies(llm, search, catalog=None, research_limits=None, sandbox_limits=None):
    limits = get_limits()
    return AgentDependencies(
        llm=llm,
        document_search=search,
        document_catalog=catalog or DocumentCatalog([]),
        history_message_count=limits.conversation.history_message_count,
        research_limits=research_limits or limits.research,
        sandbox_limits=sandbox_limits or limits.sandbox,
        research_roles=get_role_access().research_roles,
    )

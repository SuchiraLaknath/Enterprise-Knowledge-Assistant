"""The tool registry: the one path by which any tool runs.

Rule 2 checks RBAC three times, and this file holds the first two: `tools_for_model` filters what
a model is even offered, and `execute` checks the role again before every run, in case a model
names a tool it was never given. The third check is the data filter inside each tool.

Rule 3: a tool exists only if it is registered here, with an args schema, `min_role`, `risk` and
`timeout_s`. A `write` tool never runs without approval.
"""

import asyncio
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from config.settings import ToolLimits
from tools.admin_tools import VIEW_AUDIT_LOG_TOOL, ViewAuditLogArguments, view_audit_log
from tools.knowledge_search import TOOL_NAME as KNOWLEDGE_SEARCH_TOOL
from tools.knowledge_search import KnowledgeSearchArguments, search_knowledge
from tools.mcp_tools import (
    EMPLOYEE_DIRECTORY_TOOL,
    INCIDENT_RECORDS_TOOL,
    SERVICE_CATALOG_TOOL,
    UPDATE_INCIDENT_STATUS_TOOL,
    EmployeeDirectoryArguments,
    IncidentRecordsArguments,
    ServiceCatalogArguments,
    UpdateIncidentStatusArguments,
    look_up_employees,
    look_up_incidents,
    look_up_services,
    update_incident_status,
)
from tools.models import ToolArguments, ToolContext, ToolHandler, ToolResult, ToolRisk
from tools.python_analysis import TOOL_NAME as PYTHON_ANALYSIS_TOOL
from tools.python_analysis import PythonAnalysisArguments, run_python_analysis

logger = structlog.get_logger()

DEEP_RESEARCH_TOOL = "deep_research"
TOOL_TIMEOUT_FAULT = "tool_timeout"
# Model APIs refuse dots in tool names, but the registry's names use them to show the family.
MODEL_NAME_SEPARATOR = "__"

UNKNOWN_TOOL_NAME = "unknown_tool"
UNKNOWN_TOOL_MESSAGE = "This tool does not exist."
AUDIT_UNAVAILABLE_MESSAGE = "This change cannot be recorded right now, so it was not made."
ROLE_DENIED_MESSAGE = "Your role does not allow this tool."
APPROVAL_REQUIRED_MESSAGE = "This action needs approval first."
INVALID_ARGUMENTS_MESSAGE = "The tool was called with invalid arguments."
TIMEOUT_MESSAGE = "The tool took too long."
FAILED_MESSAGE = "The tool could not complete."


class NoToolArguments(ToolArguments):
    """For an entry that the model cannot call."""


@dataclass(frozen=True)
class ToolDefinition:
    """Everything rule 3 requires for a tool."""

    name: str
    description: str
    args_schema: type[BaseModel]
    min_role: str
    risk: ToolRisk
    timeout_s: float
    # None marks a capability that routing starts, such as research, rather than a model.
    handler: ToolHandler | None

    @property
    def model_facing_name(self) -> str:
        """Return the name a model sees, which may not contain dots."""
        return self.name.replace(".", MODEL_NAME_SEPARATOR)

    def to_model_schema(self) -> dict[str, Any]:
        """Describe the tool in the format that every provider's client accepts."""
        return {
            "type": "function",
            "function": {
                "name": self.model_facing_name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolRegistry:
    """Holds the tools and runs them. Created once and shared by all requests."""

    def __init__(
        self,
        definitions: list[ToolDefinition],
        role_ranking: tuple[str, ...],
        enabled_faults: Collection[str] = (),
    ) -> None:
        """Index the tools by both of their names."""
        self._tools = {definition.name: definition for definition in definitions}
        self._names_by_model_name = {
            definition.model_facing_name: definition.name for definition in definitions
        }
        self._role_ranking = role_ranking
        self._enabled_faults = enabled_faults

    def is_role_allowed(self, tool_name: str, role: str) -> bool:
        """Return whether a role reaches a tool's `min_role`. An unknown role never does."""
        definition = self._tools.get(tool_name)
        if definition is None or role not in self._role_ranking:
            return False
        minimum_rank = self._role_ranking.index(definition.min_role)
        return self._role_ranking.index(role) >= minimum_rank

    def tools_for_model(self, role: str) -> list[ToolDefinition]:
        """Return what a model may be offered for this role: rule 2's first check."""
        return [
            definition
            for definition in self._tools.values()
            if definition.handler is not None and self.is_role_allowed(definition.name, role)
        ]

    def resolve_model_name(self, model_facing_name: str) -> str:
        """Return the registry name for a tool as a model called it."""
        return self._names_by_model_name.get(model_facing_name, model_facing_name)

    def get_definition(self, tool_name: str) -> ToolDefinition | None:
        """Return a tool's definition, or None if it does not exist."""
        return self._tools.get(tool_name)

    async def execute(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
        context: ToolContext,
        approved: bool = False,
    ) -> ToolResult:
        """Run a tool after every check. Failures are returned as data, and are never raised."""
        definition = self._tools.get(tool_name)

        # Step 1: rule 2's second check, because a model can name a tool it was never offered.
        if (
            definition is None
            or definition.handler is None
            or not self.is_role_allowed(tool_name, context.user.role)
        ):
            # A model chose this name, so an unknown one is recorded as a fixed value.
            audited_name = UNKNOWN_TOOL_NAME if definition is None else tool_name
            await context.audit_log.record(context.user, audited_name, "denied", "role")
            logger.warning("tool_denied", tool=audited_name, role=context.user.role)
            message = UNKNOWN_TOOL_MESSAGE if definition is None else ROLE_DENIED_MESSAGE
            return ToolResult(tool_name=tool_name, status="denied", message=message)

        # Step 2: rule 3, a write tool never runs without a person's approval.
        if definition.risk == "write" and not approved:
            await context.audit_log.record(context.user, tool_name, "denied", "not approved")
            return ToolResult(
                tool_name=tool_name, status="denied", message=APPROVAL_REQUIRED_MESSAGE
            )

        # Step 3: validate the arguments. Unknown fields fail here, which stops tool abuse.
        try:
            arguments = definition.args_schema.model_validate(raw_arguments)
        except ValidationError:
            await context.audit_log.record(context.user, tool_name, "error", "invalid arguments")
            return ToolResult(
                tool_name=tool_name, status="error", message=INVALID_ARGUMENTS_MESSAGE
            )

        # Step 4: a change that cannot be audited must not happen (rule 2).
        if definition.risk == "write":
            is_audited = await context.audit_log.record(
                context.user, tool_name, "started", "approved"
            )
            if not is_audited:
                return ToolResult(
                    tool_name=tool_name, status="error", message=AUDIT_UNAVAILABLE_MESSAGE
                )

        # Step 5: run it, turning every failure into data.
        result = await self._run_with_timeout(definition, arguments, context)
        await context.audit_log.record(context.user, tool_name, result.status, "")
        return result

    async def _run_with_timeout(
        self, definition: ToolDefinition, arguments: BaseModel, context: ToolContext
    ) -> ToolResult:
        """Run the handler within its timeout. Failures name no details, which may be sensitive."""
        handler = definition.handler
        # `execute` has already refused entries with no handler, and this keeps that visible.
        if handler is None:
            return ToolResult(tool_name=definition.name, status="error", message=FAILED_MESSAGE)
        try:
            if TOOL_TIMEOUT_FAULT in self._enabled_faults:
                raise TimeoutError("Simulated tool timeout from FAULTS")
            async with asyncio.timeout(definition.timeout_s):
                return await handler(arguments, context)
        except TimeoutError:
            logger.warning("tool_timed_out", tool=definition.name)
            return ToolResult(tool_name=definition.name, status="timeout", message=TIMEOUT_MESSAGE)
        except Exception as error:  # One failed tool must not fail the turn.
            logger.warning("tool_failed", tool=definition.name, error=type(error).__name__)
            return ToolResult(tool_name=definition.name, status="error", message=FAILED_MESSAGE)


def build_tool_registry(
    limits: ToolLimits,
    role_ranking: tuple[str, ...],
    enabled_faults: Collection[str] = (),
) -> ToolRegistry:
    """Return the registry with every tool. Rule 3: this list is the whole of what can run."""
    definitions = [
        ToolDefinition(
            name=KNOWLEDGE_SEARCH_TOOL,
            description="Search internal documents for a specific fact or passage.",
            args_schema=KnowledgeSearchArguments,
            min_role="viewer",
            risk="read",
            timeout_s=limits.search_timeout_s,
            handler=search_knowledge,
        ),
        # Routing starts research, so `handler` is None and no model is offered it.
        ToolDefinition(
            name=DEEP_RESEARCH_TOOL,
            description="Read and compare a whole collection of documents.",
            args_schema=NoToolArguments,
            min_role="analyst",
            risk="read",
            timeout_s=limits.search_timeout_s,
            handler=None,
        ),
        ToolDefinition(
            name=PYTHON_ANALYSIS_TOOL,
            description="Count, average or compare data already found in this conversation turn.",
            args_schema=PythonAnalysisArguments,
            min_role="analyst",
            risk="read",
            timeout_s=limits.analysis_timeout_s,
            handler=run_python_analysis,
        ),
        ToolDefinition(
            name=EMPLOYEE_DIRECTORY_TOOL,
            description="Look up employees by name or department.",
            args_schema=EmployeeDirectoryArguments,
            min_role="analyst",
            risk="read",
            timeout_s=limits.mcp_timeout_s,
            handler=look_up_employees,
        ),
        ToolDefinition(
            name=SERVICE_CATALOG_TOOL,
            description="Look up services, their owners and their tier.",
            args_schema=ServiceCatalogArguments,
            min_role="analyst",
            risk="read",
            timeout_s=limits.mcp_timeout_s,
            handler=look_up_services,
        ),
        ToolDefinition(
            name=INCIDENT_RECORDS_TOOL,
            description="Look up live incident records by service or status.",
            args_schema=IncidentRecordsArguments,
            min_role="analyst",
            risk="read",
            timeout_s=limits.mcp_timeout_s,
            handler=look_up_incidents,
        ),
        ToolDefinition(
            name=VIEW_AUDIT_LOG_TOOL,
            description="Read the newest entries in the tool audit log.",
            args_schema=ViewAuditLogArguments,
            min_role="admin",
            risk="read",
            timeout_s=limits.audit_timeout_s,
            handler=view_audit_log,
        ),
        ToolDefinition(
            name=UPDATE_INCIDENT_STATUS_TOOL,
            description="Change an incident's status. A person must approve this first.",
            args_schema=UpdateIncidentStatusArguments,
            min_role="admin",
            risk="write",
            timeout_s=limits.mcp_timeout_s,
            handler=update_incident_status,
        ),
    ]
    return ToolRegistry(definitions, role_ranking, enabled_faults)

"""The registry's view of the MCP server's tools: args schemas and thin wrappers.

The server filters by the verified role itself, and these wrappers only forward what was asked.
"""

from typing import Literal

from pydantic import Field

from tools.models import ToolArguments, ToolContext, ToolResult

EMPLOYEE_DIRECTORY_TOOL = "mcp.employee_directory"
SERVICE_CATALOG_TOOL = "mcp.service_catalog"
INCIDENT_RECORDS_TOOL = "mcp.incident_records"
UPDATE_INCIDENT_STATUS_TOOL = "admin.update_incident_status"


class EmployeeDirectoryArguments(ToolArguments):
    """Find employees by name or department."""

    name: str = Field(default="", max_length=100, description="Part of a name. Empty means any.")
    department: str = Field(default="", max_length=50, description="Department. Empty means any.")


class ServiceCatalogArguments(ToolArguments):
    """Find services and their owners."""

    service_name: str = Field(default="", max_length=100, description="Part of a service name.")


class IncidentRecordsArguments(ToolArguments):
    """Find incident records."""

    service_name: str = Field(default="", max_length=100, description="Part of a service name.")
    status: str = Field(default="", max_length=20, description="open, resolved, or empty for all.")


class UpdateIncidentStatusArguments(ToolArguments):
    """Change an incident's status. A person approves these exact values first."""

    incident_id: str = Field(pattern=r"^LIVE-\d{1,6}$", description="The incident id.")
    status: Literal["open", "resolved"] = Field(description="The new status.")


async def _call_mcp_tool(
    registry_name: str, server_tool_name: str, arguments: ToolArguments, context: ToolContext
) -> ToolResult:
    """Forward one call, with the verified user carried by the client (rule 2)."""
    data = await context.mcp_client.call_tool(
        server_tool_name,
        arguments.model_dump(),
        context.user,
        context.tool_limits.mcp_timeout_s,
    )
    return ToolResult(tool_name=registry_name, status="ok", data=data)


async def look_up_employees(
    arguments: EmployeeDirectoryArguments, context: ToolContext
) -> ToolResult:
    """Call `employee_directory`."""
    return await _call_mcp_tool(EMPLOYEE_DIRECTORY_TOOL, "employee_directory", arguments, context)


async def look_up_services(arguments: ServiceCatalogArguments, context: ToolContext) -> ToolResult:
    """Call `service_catalog`."""
    return await _call_mcp_tool(SERVICE_CATALOG_TOOL, "service_catalog", arguments, context)


async def look_up_incidents(
    arguments: IncidentRecordsArguments, context: ToolContext
) -> ToolResult:
    """Call `incident_records`."""
    return await _call_mcp_tool(INCIDENT_RECORDS_TOOL, "incident_records", arguments, context)


async def update_incident_status(
    arguments: UpdateIncidentStatusArguments, context: ToolContext
) -> ToolResult:
    """Call `update_incident_status`, which the registry runs only after approval (rule 3)."""
    return await _call_mcp_tool(
        UPDATE_INCIDENT_STATUS_TOOL, "update_incident_status", arguments, context
    )

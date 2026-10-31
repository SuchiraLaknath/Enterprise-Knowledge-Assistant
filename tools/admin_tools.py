"""Administrator tools. `min_role="admin"` in the registry is what protects them."""

from pydantic import Field

from tools.models import ToolArguments, ToolContext, ToolResult

VIEW_AUDIT_LOG_TOOL = "admin.view_audit_log"


class ViewAuditLogArguments(ToolArguments):
    """How many entries to read."""

    entry_count: int = Field(default=20, ge=1, le=100, description="How many recent entries.")


async def view_audit_log(arguments: ViewAuditLogArguments, context: ToolContext) -> ToolResult:
    """Return the newest audit entries."""
    entries = await context.audit_log.read_recent(arguments.entry_count)
    return ToolResult(tool_name=VIEW_AUDIT_LOG_TOOL, status="ok", data=entries)

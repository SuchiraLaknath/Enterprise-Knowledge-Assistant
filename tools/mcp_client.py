"""Call the MCP server, sending the verified user out of band.

Rule 2: the role travels in request headers, which the model never sees or sets. Tool arguments
hold only what was asked for, so that the server's own filter cannot be argued with.
"""

from collections.abc import Collection
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langsmith import traceable

from mcp_server.headers import USER_ROLE_HEADER
from retrieval.models import UserContext
from tools.tracing import hide_tool_inputs

MCP_FAULT = "mcp"


class SimulatedMcpFaultError(RuntimeError):
    """Raised on purpose when `FAULTS` includes `mcp`, to demonstrate degradation."""


class McpClient:
    """Opens a short connection per call, because each one carries a different user."""

    def __init__(self, server_url: str, enabled_faults: Collection[str]) -> None:
        """Keep the address. Rule 6: this is the only MCP endpoint the tools can reach."""
        self._server_url = server_url
        self._enabled_faults = enabled_faults

    @traceable(name="mcp_tool", run_type="tool", process_inputs=hide_tool_inputs)
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], user: UserContext, timeout_s: float
    ) -> Any:
        """Call one tool as this user, and return its structured result."""
        if MCP_FAULT in self._enabled_faults:
            raise SimulatedMcpFaultError("Simulated MCP failure from FAULTS")

        # Rule 2: the verified role goes in the headers, and never in the arguments.
        transport = StreamableHttpTransport(
            self._server_url,
            headers={USER_ROLE_HEADER: user.role},
        )
        async with Client(transport) as client:
            result = await client.call_tool(tool_name, arguments, timeout=timeout_s)
        return result.data

"""The FastMCP server with dummy enterprise data, for the agent's external tools.

Rule 2: the caller's role arrives in request headers from `tools/mcp_client.py`, and every result
is filtered here, on the server. Tool arguments can narrow a search but cannot name a role, so an
injected argument gains nothing. An unknown or missing role sees nothing.

Run with: uv run python -m mcp_server.server
"""

import os
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from config.settings import get_role_access
from mcp_server.enterprise_data import EMPLOYEES, INCIDENTS, SERVICES
from mcp_server.headers import USER_ROLE_HEADER

SERVER_NAME = "enterprise-data"
ADMIN_ROLE = "admin"
# Compose sets this to reach the server from other containers. Locally it stays on this machine.
SERVER_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
SERVER_PORT = 8001
HTTP_TRANSPORT = "http"

mcp = FastMCP(SERVER_NAME)


def allowed_access_levels() -> tuple[str, ...]:
    """Return what the calling role may read, which is nothing for an unknown role."""
    role = get_http_headers().get(USER_ROLE_HEADER, "")
    return get_role_access().roles.get(role, ())


def is_known_role() -> bool:
    """Return whether the caller has a real role. People and services carry no access level."""
    return bool(allowed_access_levels())


def _matches(value: str, search_text: str) -> bool:
    """Compare ignoring case, where an empty search matches everything."""
    return search_text.lower() in value.lower()


@mcp.tool
def employee_directory(name: str = "", department: str = "") -> list[dict[str, Any]]:
    """Find employees by part of a name, or by department."""
    # Rule 2: any known role may look people up, but an unknown caller sees nothing.
    if not is_known_role():
        return []
    return [
        employee
        for employee in EMPLOYEES
        if _matches(employee["name"], name) and _matches(employee["department"], department)
    ]


@mcp.tool
def service_catalog(service_name: str = "") -> list[dict[str, Any]]:
    """Find services, with their owner, tier and on-call rota."""
    # Rule 2: as for the directory, an unknown caller sees nothing.
    if not is_known_role():
        return []
    return [service for service in SERVICES if _matches(service["name"], service_name)]


@mcp.tool
def incident_records(service_name: str = "", status: str = "") -> list[dict[str, Any]]:
    """Find incidents the caller may read, by service or status."""
    # Rule 2: incidents carry access levels, so this filter runs on every call.
    permitted_levels = allowed_access_levels()
    return [
        incident
        for incident in INCIDENTS
        if incident["access_level"] in permitted_levels
        and _matches(incident["service"], service_name)
        and _matches(incident["status"], status)
    ]


@mcp.tool
def update_incident_status(incident_id: str, status: str) -> dict[str, Any]:
    """Change an incident's status, for an administrator."""
    # Rule 2: the API has already checked this, and a write is worth checking again here.
    if get_http_headers().get(USER_ROLE_HEADER, "") != ADMIN_ROLE:
        return {"updated": False, "reason": "Only an administrator may change an incident."}
    for incident in INCIDENTS:
        if incident["id"] == incident_id:
            incident["status"] = status
            return {"updated": True, "incident_id": incident_id, "status": status}
    return {"updated": False, "reason": "No incident has this id."}


if __name__ == "__main__":
    mcp.run(transport=HTTP_TRANSPORT, host=SERVER_HOST, port=SERVER_PORT)

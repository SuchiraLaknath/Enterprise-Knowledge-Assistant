"""The FastMCP server with dummy enterprise data, for the agent's external tools.

Rule 2: the caller's role arrives in a token signed by the API, from `tools/mcp_client.py`, and
every result is filtered here, on the server. Tool arguments can narrow a search but cannot name a
role, so an injected argument gains nothing. A missing or forged token sees nothing.

Run with: uv run python -m mcp_server.server
"""

import os
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from config.settings import get_environment_settings, get_role_access
from mcp_server.enterprise_data import EMPLOYEES, INCIDENTS, SERVICES
from mcp_server.service_token import SERVICE_TOKEN_HEADER, read_verified_role

SERVER_NAME = "enterprise-data"
ADMIN_ROLE = "admin"
# Compose sets this to reach the server from other containers. Locally it stays on this machine.
SERVER_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
SERVER_PORT = 8001
HTTP_TRANSPORT = "http"

mcp = FastMCP(SERVER_NAME)


def verified_role() -> str:
    """Return the role the API signed for this request, or an empty string."""
    token = get_http_headers().get(SERVICE_TOKEN_HEADER, "")
    return read_verified_role(token, get_environment_settings().jwt_secret.get_secret_value())


def allowed_access_levels() -> tuple[str, ...]:
    """Return what the calling role may read, which is nothing for an unknown role."""
    return get_role_access().roles.get(verified_role(), ())


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
def incident_records(
    incident_id: str = "", service_name: str = "", status: str = ""
) -> list[dict[str, Any]]:
    """Find incidents the caller may read, by id, service or status."""
    # Rule 2: incidents carry access levels, so this filter runs on every call.
    permitted_levels = allowed_access_levels()
    return [
        incident
        for incident in INCIDENTS
        if incident["access_level"] in permitted_levels
        and _matches(incident["id"], incident_id)
        and _matches(incident["service"], service_name)
        and _matches(incident["status"], status)
    ]


@mcp.tool
def update_incident_status(incident_id: str, status: str) -> dict[str, Any]:
    """Change an incident's status, for an administrator."""
    # Rule 2: the API has already checked this, and a write is worth checking again here.
    if verified_role() != ADMIN_ROLE:
        return {"updated": False, "reason": "Only an administrator may change an incident."}
    for incident in INCIDENTS:
        if incident["id"] == incident_id:
            incident["status"] = status
            return {"updated": True, "incident_id": incident_id, "status": status}
    return {"updated": False, "reason": "No incident has this id."}


if __name__ == "__main__":
    mcp.run(transport=HTTP_TRANSPORT, host=SERVER_HOST, port=SERVER_PORT)

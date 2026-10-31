import pytest
from fastmcp import Client

from mcp_server import server
from mcp_server.server import mcp


@pytest.fixture
def call_as(monkeypatch):
    # The real client sends the role as a header, so these tests set the same header directly.
    async def call(role, tool_name, arguments=None):
        headers = {server.USER_ROLE_HEADER: role} if role else {}
        monkeypatch.setattr(server, "get_http_headers", lambda: headers)
        async with Client(mcp) as client:
            result = await client.call_tool(tool_name, arguments or {})
        return result.data

    return call


async def test_incidents_are_filtered_by_the_verified_role(call_as):
    # Rule 2: the server filters, so a wrong client cannot widen access.
    analyst_ids = {item["id"] for item in await call_as("analyst", "incident_records")}
    viewer_ids = {item["id"] for item in await call_as("viewer", "incident_records")}
    admin_ids = {item["id"] for item in await call_as("admin", "incident_records")}

    assert "LIVE-1029" not in analyst_ids
    assert "LIVE-1036" in analyst_ids
    assert "LIVE-1036" not in viewer_ids
    assert "LIVE-1029" in admin_ids


async def test_a_missing_or_unknown_role_sees_nothing(call_as):
    # Attack: a request that skips the client, or forges a role that does not exist.
    for role in ("", "superuser"):
        assert await call_as(role, "incident_records") == []
        assert await call_as(role, "employee_directory") == []
        assert await call_as(role, "service_catalog") == []


async def test_an_argument_cannot_set_a_role(call_as):
    # Attack: rule 2, the role travels out of band, so the tool has no such argument.
    with pytest.raises(Exception):  # noqa: B017
        await call_as("viewer", "incident_records", {"role": "admin"})


async def test_results_narrow_by_their_arguments(call_as):
    employees = await call_as("analyst", "employee_directory", {"department": "security"})
    open_incidents = await call_as("analyst", "incident_records", {"status": "open"})

    assert [employee["department"] for employee in employees] == ["security"]
    assert {incident["status"] for incident in open_incidents} == {"open"}

import jwt
import pytest
from fastmcp import Client

from config.settings import EnvironmentSettings
from mcp_server import server
from mcp_server.server import mcp
from mcp_server.service_token import (
    SERVICE_TOKEN_HEADER,
    TOKEN_ALGORITHM,
    create_service_token,
    read_verified_role,
)

SECRET = "a-test-secret-that-is-long-enough-for-hs256"


@pytest.fixture
def call_with_token(monkeypatch):
    monkeypatch.setattr(
        server, "get_environment_settings", lambda: EnvironmentSettings(jwt_secret=SECRET)
    )

    async def call(token, tool_name, arguments=None):
        headers = {SERVICE_TOKEN_HEADER: token} if token else {}
        monkeypatch.setattr(server, "get_http_headers", lambda: headers)
        async with Client(mcp) as client:
            result = await client.call_tool(tool_name, arguments or {})
        return result.data

    return call


@pytest.fixture
def call_as(call_with_token):
    async def call(role, tool_name, arguments=None):
        token = create_service_token(role, SECRET, 30.0) if role else ""
        return await call_with_token(token, tool_name, arguments)

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


@pytest.mark.parametrize(
    "forged_token",
    [
        # Attack: a plain role, as a header alone would have been believed.
        "admin",
        # Attack: a token signed with a guessed secret.
        jwt.encode({"role": "admin", "aud": "mcp-server", "exp": 9999999999}, "x" * 32, "HS256"),
        # Attack: a login token for the API, which has no audience for this server.
        jwt.encode({"role": "admin", "exp": 9999999999}, SECRET, TOKEN_ALGORITHM),
        # Attack: an unsigned token, which some libraries once accepted.
        jwt.encode({"role": "admin", "aud": "mcp-server", "exp": 9999999999}, "", "none"),
    ],
)
async def test_a_forged_token_reads_nothing_and_writes_nothing(call_with_token, forged_token):
    # A review found that anything reaching the port could claim to be an admin with a header.
    incidents = await call_with_token(forged_token, "incident_records")
    update = await call_with_token(
        forged_token, "update_incident_status", {"incident_id": "LIVE-1041", "status": "resolved"}
    )

    assert incidents == []
    assert update["updated"] is False


def test_an_expired_token_is_refused():
    assert read_verified_role(create_service_token("admin", SECRET, -1.0), SECRET) == ""


def test_no_secret_means_no_trust():
    # With no secret set, an empty key must not make every token valid.
    assert read_verified_role(create_service_token("admin", "x" * 32, 30.0), "") == ""


async def test_only_an_admin_can_change_an_incident(call_as):
    update = await call_as(
        "analyst", "update_incident_status", {"incident_id": "LIVE-1041", "status": "resolved"}
    )

    assert update["updated"] is False


async def test_incidents_can_be_found_by_id(call_as):
    incidents = await call_as("analyst", "incident_records", {"incident_id": "LIVE-1038"})

    assert [incident["id"] for incident in incidents] == ["LIVE-1038"]

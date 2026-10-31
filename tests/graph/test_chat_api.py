"""Run the real API and graph with fake services, through HTTP."""

import json
import time
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_agent_graph, create_checkpoint_serializer
from api.auth import TOKEN_ALGORITHM
from api.main import create_app
from api.ratelimit import build_rate_limiter
from api.routes import chat as chat_route
from api.routes.chat import build_checkpoint_thread_id
from api.services import ApplicationServices
from config.settings import get_limits, get_role_access
from retrieval.models import SearchResult, UserContext
from tests.fakes.fake_dependencies import make_agent_dependencies
from tests.fakes.fake_llm import FakeLlm, make_plan
from tests.fakes.fake_search import make_chunk

JWT_SECRET = "test-secret-that-is-long-enough-for-hs256"


class RecordingSearch:
    def __init__(self):
        self.received_users = []

    async def search(self, query_text, user, document_filters=None, namespaces=None, **options):
        self.received_users.append(user)
        return SearchResult(chunks=[make_chunk("DOC#c0")])


@pytest.fixture
def search():
    return RecordingSearch()


class FakeCheckpointAvailability:
    def __init__(self):
        self.available = True

    async def is_available(self):
        return self.available


@pytest.fixture
def llm():
    return FakeLlm(plans=[make_plan() for _ in range(4)], answers=["Answer [1]."] * 4)


@pytest.fixture
async def client(search, llm, monkeypatch):
    async def no_trace_url(run_id, project_name, timeout_seconds):
        return "https://trace.example/run"

    monkeypatch.setattr(chat_route, "find_trace_url", no_trace_url)
    app = create_app(use_lifespan=False)
    app.state.services = ApplicationServices(
        agent_graph=build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer())),
        in_memory_graph=build_agent_graph(InMemorySaver(serde=create_checkpoint_serializer())),
        checkpoint_availability=FakeCheckpointAvailability(),
        agent_dependencies=make_agent_dependencies(llm, search),
        jwt_secret=JWT_SECRET,
        role_access=get_role_access(),
        api_limits=get_limits().api,
        langsmith_project="test",
        rate_limiter=build_rate_limiter("", get_limits().rate_limits),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def sign_in(client, username="vimal", password="viewer-demo"):
    response = await client.post("/auth/login", json={"username": username, "password": password})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def post_chat(client, headers, message="Why did payments fail?", thread_id="thread-1"):
    response = await client.post(
        "/chat", json={"message": message, "thread_id": thread_id}, headers=headers
    )
    events = [
        json.loads(line.removeprefix("data:").strip())
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    return response, events


async def test_login_returns_a_token_and_the_role(client):
    response = await client.post(
        "/auth/login", json={"username": "nadee", "password": "analyst-demo"}
    )

    assert response.status_code == 200
    assert response.json()["role"] == "analyst"


async def test_wrong_password_and_unknown_user_give_the_same_reply(client):
    wrong_password = await client.post(
        "/auth/login", json={"username": "vimal", "password": "wrong"}
    )
    unknown_user = await client.post(
        "/auth/login", json={"username": "nobody", "password": "wrong"}
    )

    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


async def test_chat_without_a_token_is_rejected(client):
    response = await client.post("/chat", json={"message": "Hello", "thread_id": "thread-1"})

    assert response.status_code == 401


async def test_a_forged_token_is_rejected(client):
    forged_token = jwt.encode(
        {"sub": "admin", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "the-attackers-own-secret-is-long-enough",
        algorithm=TOKEN_ALGORITHM,
    )

    response, _ = await post_chat(client, {"Authorization": f"Bearer {forged_token}"})

    assert response.status_code == 401


async def test_an_expired_token_is_rejected(client):
    expired_token = jwt.encode(
        {"sub": "vimal", "exp": datetime.now(UTC) - timedelta(minutes=1)},
        JWT_SECRET,
        algorithm=TOKEN_ALGORITHM,
    )

    response, _ = await post_chat(client, {"Authorization": f"Bearer {expired_token}"})

    assert response.status_code == 401


async def test_a_signed_token_without_an_expiry_is_rejected(client):
    token = jwt.encode({"sub": "admin"}, JWT_SECRET, algorithm=TOKEN_ALGORITHM)

    response, _ = await post_chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


async def test_a_role_claim_in_the_token_is_ignored(client, search):
    # Rule 1: the role comes from the server, even if a valid token claims another one.
    token = jwt.encode(
        {"sub": "vimal", "role": "admin", "exp": datetime.now(UTC) + timedelta(hours=1)},
        JWT_SECRET,
        algorithm=TOKEN_ALGORITHM,
    )

    await post_chat(client, {"Authorization": f"Bearer {token}"})

    assert search.received_users[0].role == "viewer"
    assert "restricted" not in search.received_users[0].allowed_access_levels


@pytest.mark.parametrize(
    ("username", "password", "expected_levels"),
    [
        ("vimal", "viewer-demo", ("public", "internal")),
        ("nadee", "analyst-demo", ("public", "internal", "confidential")),
        ("admin", "admin-demo", ("public", "internal", "confidential", "restricted")),
    ],
)
async def test_each_role_gets_its_configured_access_levels(
    client, search, username, password, expected_levels
):
    headers = await sign_in(client, username, password)

    await post_chat(client, headers)

    assert search.received_users[0].allowed_access_levels == expected_levels


async def test_chat_streams_activity_and_then_the_final_answer(client):
    headers = await sign_in(client)

    response, events = await post_chat(client, headers)

    event_types = [event["type"] for event in events]
    assert response.status_code == 200
    assert {"node", "state", "route", "retrieval", "token"} <= set(event_types)
    assert event_types[-2:] == ["final", "trace"]
    assert events[-2]["answer"] == "Answer [1]."
    assert events[-2]["citations"][0]["chunk_id"] == "DOC#c0"
    assert events[-2]["draft_status"] == "confirmed"
    assert events[-1]["url"] == "https://trace.example/run"


async def test_the_final_event_reports_when_the_answer_replaced_the_draft(client, llm):
    # The poisoned link is shown as plain text in the draft, and removed from the final answer.
    llm.answers = ["See https://attacker.example/x [1]."]
    headers = await sign_in(client)

    _, events = await post_chat(client, headers)

    final_event = next(event for event in events if event["type"] == "final")
    assert final_event["draft_status"] == "replaced"
    assert "attacker.example" not in final_event["answer"]


async def test_two_users_with_the_same_thread_id_do_not_share_a_conversation(client, llm):
    # Guessing another user's thread id must not reveal their conversation.
    await post_chat(client, await sign_in(client), "My private question", "shared-id")

    await post_chat(
        client, await sign_in(client, "nadee", "analyst-demo"), "Second user", "shared-id"
    )

    second_user_messages = [message.content for message in llm.received_answer_messages[1]]
    assert "My private question" not in second_user_messages


async def test_a_role_change_starts_a_new_conversation():
    # A demoted user must not keep answers from their earlier access level in the prompt.
    analyst = UserContext(user_id="sam", role="analyst", allowed_access_levels=("internal",))
    viewer = UserContext(user_id="sam", role="viewer", allowed_access_levels=("public",))

    assert build_checkpoint_thread_id(analyst, "t1") != build_checkpoint_thread_id(viewer, "t1")


async def test_a_message_over_the_limit_is_rejected(client):
    headers = await sign_in(client)

    response, _ = await post_chat(client, headers, "x" * 5000)

    assert response.status_code == 422


@pytest.mark.parametrize("thread_id", ["", "has spaces", "../../etc/passwd", "x" * 65])
async def test_an_invalid_thread_id_is_rejected(client, thread_id):
    headers = await sign_in(client)

    response, _ = await post_chat(client, headers, thread_id=thread_id)

    assert response.status_code == 422


async def test_an_empty_message_is_rejected(client):
    headers = await sign_in(client)

    response, _ = await post_chat(client, headers, "")

    assert response.status_code == 422


async def test_a_failure_in_the_graph_becomes_a_safe_error_event(client, search):
    async def broken_search(*arguments, **options):
        raise RuntimeError("secret internal detail")

    search.search = broken_search
    headers = await sign_in(client)

    response, events = await post_chat(client, headers)

    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert "secret internal detail" not in response.text


async def test_a_slow_trace_lookup_cannot_delay_the_answer(monkeypatch):
    class SlowClient:
        def read_project(self, project_name):
            time.sleep(0.5)

    monkeypatch.setattr(chat_route, "LangSmithClient", SlowClient)
    started_at = time.monotonic()

    trace_url = await chat_route.find_trace_url(uuid.uuid4(), "test", timeout_seconds=0.05)

    assert trace_url is None
    assert time.monotonic() - started_at < 0.4


async def test_health_needs_no_login(client):
    response = await client.get("/health")

    assert response.json() == {"status": "ok"}


async def test_too_many_requests_get_a_429_before_the_stream_opens(client):
    # Attack: a flood from one user. The refusal must come before any LLM call.
    headers = await sign_in(client)
    limit = get_limits().rate_limits["viewer"].capacity

    for turn in range(limit):
        await post_chat(client, headers, thread_id=f"thread-{turn}")
    response = await client.post(
        "/chat", json={"message": "One more", "thread_id": "extra"}, headers=headers
    )

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0


@pytest.mark.faults
async def test_a_postgres_outage_still_answers_and_says_so(client, search):
    client._transport.app.state.services.checkpoint_availability.available = False
    headers = await sign_in(client)

    _, events = await post_chat(client, headers)

    final_event = next(event for event in events if event["type"] == "final")
    assert final_event["answer"]
    assert "postgres_unavailable_in_memory_checkpoints" in final_event["degraded"]


async def test_an_invalid_request_gets_a_structured_422_and_no_echo(client):
    # Attack: a malformed thread id. The reply must name the field, not repeat the input.
    headers = await sign_in(client)

    response = await client.post(
        "/chat",
        json={"message": "Hello", "thread_id": "'; DROP TABLE checkpoints; --"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert response.json()["problems"][0]["field"] == "body.thread_id"
    assert "DROP TABLE" not in response.text

import pytest

from agent.rlm.helpers import (
    ResearchBudgetExceeded,
    ResearchSession,
    chunk,
    group_by_doc,
)
from config.settings import get_limits
from retrieval.models import UserContext
from tests.fakes.fake_dependencies import make_agent_dependencies
from tests.fakes.fake_llm import FakeLlm
from tests.fakes.fake_research import (
    FakeResearchSearch,
    make_catalog,
    make_catalog_entry,
    make_section,
)

ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)
VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))


@pytest.fixture(autouse=True)
def ignore_panel_events(monkeypatch):
    # These tests call helpers directly, so there is no graph run to stream events to.
    monkeypatch.setattr("agent.rlm.helpers.emit_event", lambda *arguments, **details: None)


def make_session(search, catalog=None, user=ANALYST, **limit_changes):
    limits = get_limits().research.model_copy(update=limit_changes)
    dependencies = make_agent_dependencies(
        FakeLlm(), search, catalog or make_catalog(), research_limits=limits
    )
    return ResearchSession(user, dependencies)


async def test_search_never_uses_the_reranker_and_keeps_the_verified_user():
    # The rerank quota is 500 a month, and one plan can search several times.
    search = FakeResearchSearch([make_section("INC-1")])

    await make_session(search)._search("payment outage", (), (), None)

    call = search.search_calls[0]
    assert call["use_reranker"] is False
    assert call["user"] == ANALYST
    assert call["result_count"] == get_limits().research.search_result_count


async def test_a_plan_receives_no_document_text():
    # Text stays in the session, so generated code has nothing to leak or act on.
    search = FakeResearchSearch([make_section("INC-1", text="SECRET TEXT")])

    records = await make_session(search)._search("payment outage", (), (), None)

    assert "text" not in records[0]
    assert "SECRET TEXT" not in str(records)


async def test_too_many_searches_are_refused():
    search = FakeResearchSearch([make_section("INC-1")])
    session = make_session(search, max_search_calls=1)

    await session._search("first", (), (), None)
    with pytest.raises(ResearchBudgetExceeded):
        await session._search("second", (), (), None)


async def test_a_plan_cannot_widen_a_search_with_an_access_argument():
    # Rule 1: a helper has no access parameter, so the attempt fails rather than widening.
    session = make_session(FakeResearchSearch())

    with pytest.raises(TypeError):
        session.explore_documents(access_level="restricted")


async def test_stale_catalog_access_is_caught_on_the_returned_metadata():
    # Rule 2: the catalog says internal, but the stored chunk has since been reclassified.
    reclassified_chunk = make_section("INC-1", access_level="restricted")
    search = FakeResearchSearch([reclassified_chunk])
    catalog = make_catalog(make_catalog_entry("INC-1", access_level="internal"))

    records = await make_session(search, catalog)._fetch_sections(["INC-1"], None)

    assert records == []


async def test_a_section_keyword_reads_only_matching_sections():
    search = FakeResearchSearch([make_section("INC-1", 0), make_section("INC-1", 1)])
    catalog = make_catalog(make_catalog_entry("INC-1", chunk_count=2))

    records = await make_session(search, catalog)._fetch_sections(["INC-1"], "root cause")

    assert [record["chunk_id"] for record in records] == ["INC-1#c0"]


async def test_delegate_refuses_more_batches_than_a_level_allows():
    search = FakeResearchSearch([make_section(f"INC-{number}") for number in range(3)])
    session = make_session(search, max_batches_per_level=2)
    records = await session._search("payment outage", (), (), None)

    with pytest.raises(ResearchBudgetExceeded):
        await session._delegate("task", chunk(records, 1))


async def test_a_record_the_session_never_read_cannot_reach_a_sub_agent():
    # Attack: generated code invents a chunk id for a document it was never given.
    session = make_session(FakeResearchSearch())

    assert session._resolve_batches([[{"chunk_id": "SECRET#c0"}]]) == []


async def test_a_chunk_fetched_twice_reaches_a_sub_agent_once():
    # One chunk covers several headings, so a plan that fetches by two keywords gets it twice.
    # Without this, duplicates push a good plan over the batch limit and waste a retry.
    search = FakeResearchSearch([make_section("INC-1"), make_section("INC-2")])
    session = make_session(search)
    records = await session._search("payment outage", (), (), None)

    chunk_batches = session._resolve_batches([records, records, records])

    assert [[chunk.chunk_id for chunk in batch] for batch in chunk_batches] == [
        ["INC-1#c0", "INC-2#c0"]
    ]


def test_group_by_doc_collects_each_documents_records():
    records = [{"doc_id": "A", "n": 1}, {"doc_id": "B", "n": 2}, {"doc_id": "A", "n": 3}]

    grouped_records = group_by_doc(records)

    assert [record["n"] for record in grouped_records["A"]] == [1, 3]
    assert list(grouped_records) == ["A", "B"]


def test_chunk_splits_a_list_into_batches():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

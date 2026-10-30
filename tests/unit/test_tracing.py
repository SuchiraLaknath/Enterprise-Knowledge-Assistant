from unittest.mock import MagicMock

from langsmith import Client, tracing_context

from config.settings import get_limits
from retrieval.hybrid import hybrid_search
from retrieval.models import SearchResult
from retrieval.tracing import keep_output_summary, keep_query_inputs
from tests.fakes.fake_search import FakeSearchBackend, FakeSparseEncoder, make_chunk

SECRET_TEXT = "confidential document body"


def test_inputs_keep_the_query_and_drop_user_and_clients():
    inputs = {"query_text": "outage", "user": object(), "backend": object(), "self": object()}

    assert keep_query_inputs(inputs) == {"query_text": "outage"}


def test_output_summary_of_a_chunk_list_has_no_document_text():
    summary = keep_output_summary([make_chunk("DOC#c0", text=SECRET_TEXT)])

    assert summary["chunks"][0]["chunk_id"] == "DOC#c0"
    assert "text" not in summary["chunks"][0]
    assert SECRET_TEXT not in str(summary)


def test_output_summary_of_a_result_keeps_the_degradation_details():
    result = SearchResult(
        chunks=[make_chunk(text=SECRET_TEXT)], degraded=["rerank_unavailable"], dense_weight=0.6
    )

    summary = keep_output_summary(result)

    assert summary["degraded"] == ["rerank_unavailable"]
    assert summary["dense_weight"] == 0.6
    assert SECRET_TEXT not in str(summary)


async def test_real_trace_of_a_search_contains_no_document_text():
    backend = FakeSearchBackend({"payments": [make_chunk("DOC#c0", text=SECRET_TEXT)]})
    tracing_client = MagicMock(spec=Client)

    with tracing_context(enabled=True, client=tracing_client):
        await hybrid_search(
            "outage",
            ("payments",),
            {"access_level": {"$in": ["internal"]}},
            backend,
            FakeSparseEncoder(),
            get_limits().retrieval,
        )

    assert tracing_client.update_run.called
    assert SECRET_TEXT not in str(tracing_client.update_run.call_args_list)
    assert "DOC#c0" in str(tracing_client.update_run.call_args_list)

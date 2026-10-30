from pathlib import Path

import pytest

from config.settings import get_limits
from retrieval import hybrid as hybrid_module
from retrieval.hybrid import (
    choose_dense_weight,
    hybrid_search,
    load_bm25_encoder,
    scale_hybrid_vectors,
)
from tests.fakes.fake_search import FakeSearchBackend, FakeSparseEncoder, make_chunk

LIMITS = get_limits().retrieval
FILTER = {"access_level": {"$in": ["internal"]}}


async def run_search(backend, query_text="payment outage", namespaces=("payments", "platform")):
    return await hybrid_search(query_text, namespaces, FILTER, backend, FakeSparseEncoder(), LIMITS)


@pytest.mark.parametrize(
    "query_text",
    [
        "what does ERR-PAY-5041 mean",
        "summarise INC-2026-014",
        "show runbook RB-003",
        "ISO8583 response codes",
        "err-pay-4091",
    ],
)
def test_queries_with_identifiers_favour_keyword_matching(query_text):
    assert choose_dense_weight(query_text, LIMITS) == LIMITS.keyword_query_dense_weight


@pytest.mark.parametrize(
    "query_text",
    ["why do card payments fail", "annual leave carry over", "what is the remote work policy"],
)
def test_natural_language_queries_use_the_default_weight(query_text):
    assert choose_dense_weight(query_text, LIMITS) == LIMITS.default_dense_weight


def test_scaling_applies_each_weight_to_its_own_vector():
    scaled_dense, scaled_sparse = scale_hybrid_vectors(
        [1.0, 2.0], {"indices": [5], "values": [10.0]}, dense_weight=0.25
    )

    assert scaled_dense == [0.25, 0.5]
    assert scaled_sparse == {"indices": [5], "values": [7.5]}


async def test_search_queries_every_namespace_and_sorts_across_them():
    backend = FakeSearchBackend(
        {
            "payments": [make_chunk("LOW#c0", score=0.2)],
            "platform": [make_chunk("HIGH#c0", score=0.9, department="platform")],
        }
    )

    outcome = await run_search(backend)

    assert sorted(backend.queried_namespaces) == ["payments", "platform"]
    assert [chunk.chunk_id for chunk in outcome.chunks] == ["HIGH#c0", "LOW#c0"]


async def test_the_same_filter_reaches_every_namespace():
    backend = FakeSearchBackend()

    await run_search(backend)

    assert backend.received_filters == [FILTER, FILTER]


async def test_one_failed_namespace_leaves_the_others_usable():
    backend = FakeSearchBackend(
        {"payments": [make_chunk("GOOD#c0")]}, failing_namespaces=("platform",)
    )

    outcome = await run_search(backend)

    assert [chunk.chunk_id for chunk in outcome.chunks] == ["GOOD#c0"]
    assert outcome.failed_namespaces == ["platform"]


async def test_search_raises_when_every_namespace_fails():
    backend = FakeSearchBackend(failing_namespaces=("payments", "platform"))

    with pytest.raises(ConnectionError):
        await run_search(backend)


async def test_query_with_no_sparse_terms_sends_no_sparse_vector():
    # An empty query produces no terms, and Pinecone rejects an empty sparse vector.
    backend = FakeSearchBackend()

    await run_search(backend, query_text="", namespaces=("payments",))

    assert backend.received_sparse_vectors == [None]


def test_missing_tokenizer_data_fails_clearly_instead_of_downloading(monkeypatch):
    # Rule 6: BM25Encoder would otherwise download this data from GitHub when created.
    def report_missing_data(data_path):
        raise LookupError(data_path)

    def fail_if_encoder_is_created():
        pytest.fail("BM25Encoder must not be created without its tokenizer data.")

    monkeypatch.setattr(hybrid_module.nltk.data, "find", report_missing_data)
    monkeypatch.setattr(hybrid_module, "BM25Encoder", fail_if_encoder_is_created)

    with pytest.raises(RuntimeError, match="nltk.downloader"):
        load_bm25_encoder(Path("unused.json"))

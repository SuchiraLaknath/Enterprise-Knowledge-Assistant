"""Failure injection for retrieval: each fallback from the failure handling table."""

import pytest

from config.settings import EnvironmentSettings, get_limits, get_model_names
from ingestion.chunk import DocumentChunk
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.models import UserContext
from retrieval.pinecone_client import PineconeSearchBackend, SimulatedFaultError
from retrieval.search import DocumentSearchService
from tests.fakes.fake_search import FakeSearchBackend, FakeSparseEncoder, make_chunk

pytestmark = pytest.mark.faults

LIMITS = get_limits().retrieval
VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))


def make_local_chunk(doc_id: str, access_level: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}#c0",
        doc_id=doc_id,
        title="Local title",
        section="Summary",
        department="payments",
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=1000,
        source_uri=f"docs://payments/{doc_id}",
        text="payment outage report",
    )


def make_service(backend, local_chunks=(), breaker=None):
    encoder = FakeSparseEncoder()
    return DocumentSearchService(
        backend=backend,
        sparse_encoder=encoder,
        local_keyword_index=LocalKeywordIndex(list(local_chunks), encoder),
        circuit_breaker=breaker or CircuitBreaker(failure_threshold=5, recovery_seconds=30.0),
        limits=LIMITS,
    )


async def test_pinecone_failure_falls_back_to_local_keyword_search():
    backend = FakeSearchBackend(embedding_fails=True)
    service = make_service(backend, [make_local_chunk("LOCAL", "internal")])

    result = await service.search("payment outage", VIEWER)

    assert [chunk.doc_id for chunk in result.chunks] == ["LOCAL"]
    assert result.degraded == ["pinecone_unavailable", "local_keyword_search"]


async def test_local_fallback_never_returns_a_document_above_the_users_level():
    # Rule 2: the fallback has no server-side filter, so it must enforce access itself.
    backend = FakeSearchBackend(embedding_fails=True)
    local_chunks = [
        make_local_chunk("OPEN", "internal"),
        make_local_chunk("CONFIDENTIAL", "confidential"),
        make_local_chunk("RESTRICTED", "restricted"),
    ]

    result = await make_service(backend, local_chunks).search("payment outage", VIEWER)

    assert [chunk.doc_id for chunk in result.chunks] == ["OPEN"]


async def test_all_namespaces_failing_also_falls_back():
    backend = FakeSearchBackend(failing_namespaces=LIMITS.namespaces)
    service = make_service(backend, [make_local_chunk("LOCAL", "internal")])

    result = await service.search("payment outage", VIEWER)

    assert "local_keyword_search" in result.degraded
    assert [chunk.doc_id for chunk in result.chunks] == ["LOCAL"]


async def test_repeated_failures_open_the_breaker_and_stop_calling_pinecone():
    backend = FakeSearchBackend(failing_namespaces=LIMITS.namespaces)
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0)
    service = make_service(backend, [make_local_chunk("LOCAL", "internal")], breaker)

    await service.search("payment outage", VIEWER)
    await service.search("payment outage", VIEWER)
    calls_before_breaker_opened = len(backend.queried_namespaces)
    result = await service.search("payment outage", VIEWER)

    assert len(backend.queried_namespaces) == calls_before_breaker_opened
    assert result.degraded == ["pinecone_circuit_open", "local_keyword_search"]


async def test_rerank_failure_keeps_hybrid_order():
    backend = FakeSearchBackend(
        {"payments": [make_chunk("FIRST#c0", score=0.9), make_chunk("SECOND#c0", score=0.5)]},
        rerank_fails=True,
    )

    result = await make_service(backend).search("payment outage", VIEWER)

    assert [chunk.chunk_id for chunk in result.chunks] == ["FIRST#c0", "SECOND#c0"]
    assert result.degraded == ["rerank_unavailable"]


async def test_faults_setting_makes_the_real_backend_fail_on_demand():
    settings = EnvironmentSettings(
        pinecone_api_key="test-key", pinecone_index="test-index", faults="pinecone,rerank"
    )
    backend = PineconeSearchBackend(settings, get_model_names())

    with pytest.raises(SimulatedFaultError):
        await backend.embed_query("payment outage")
    with pytest.raises(SimulatedFaultError):
        await backend.rerank("payment outage", ["some text"], 1)

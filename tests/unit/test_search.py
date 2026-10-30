from config.settings import get_limits
from ingestion.chunk import DocumentChunk
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.models import DocumentFilters, RerankedPosition, UserContext
from retrieval.search import DocumentSearchService
from tests.fakes.fake_search import FakeSearchBackend, FakeSparseEncoder, make_chunk

LIMITS = get_limits().retrieval
VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public", "internal"))
ANALYST = UserContext(
    user_id="nadee",
    role="analyst",
    allowed_access_levels=("public", "internal", "confidential"),
)


def make_local_chunk(doc_id: str, access_level: str, text: str, created_ts: int = 1000):
    return DocumentChunk(
        chunk_id=f"{doc_id}#c0",
        doc_id=doc_id,
        title="Local title",
        section="Summary",
        department="payments",
        document_type="incident",
        access_level=access_level,
        created_date="2026-01-15",
        created_ts=created_ts,
        source_uri=f"docs://payments/{doc_id}",
        text=text,
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


async def test_search_returns_reranked_results_with_attribution():
    backend = FakeSearchBackend(
        {"payments": [make_chunk("FIRST#c0", score=0.9), make_chunk("SECOND#c0", score=0.5)]}
    )

    result = await make_service(backend).search("payment outage", VIEWER)

    # The fake reranker reverses its input, which proves that its order was used.
    assert [chunk.chunk_id for chunk in result.chunks] == ["SECOND#c0", "FIRST#c0"]
    assert result.chunks[0].source_uri == "docs://payments/SECOND#c0"
    assert result.degraded == []
    assert result.dense_weight == LIMITS.default_dense_weight


async def test_chunks_the_reranker_scores_as_unrelated_are_dropped():
    class ScoringBackend(FakeSearchBackend):
        async def rerank(self, query_text, document_texts, result_count):
            return [
                RerankedPosition(input_index=0, score=0.9),
                RerankedPosition(input_index=1, score=0.003),
            ]

    backend = ScoringBackend(
        {"payments": [make_chunk("RELEVANT#c0", score=0.9), make_chunk("NOISE#c0", score=0.5)]}
    )

    result = await make_service(backend).search("payment outage", VIEWER)

    assert [chunk.chunk_id for chunk in result.chunks] == ["RELEVANT#c0"]


async def test_research_searches_skip_the_reranker_to_protect_the_quota():
    backend = FakeSearchBackend({"payments": [make_chunk()]})

    await make_service(backend).search("payment outage", VIEWER, use_reranker=False)

    assert backend.rerank_call_count == 0


async def test_user_access_levels_reach_the_query_filter():
    backend = FakeSearchBackend()

    await make_service(backend).search("payment outage", ANALYST)

    assert backend.received_filters[0]["access_level"] == {
        "$in": ["public", "internal", "confidential"]
    }


async def test_a_chunk_above_the_users_level_is_dropped_even_if_the_backend_returns_it():
    # Simulates a filter bug: the final access check must still protect the user.
    backend = FakeSearchBackend(
        {
            "payments": [
                make_chunk("OPEN#c0", access_level="internal"),
                make_chunk("SECRET#c0", access_level="restricted"),
            ]
        }
    )

    result = await make_service(backend).search("payment outage", VIEWER)

    assert [chunk.chunk_id for chunk in result.chunks] == ["OPEN#c0"]


async def test_namespaces_can_be_narrowed():
    backend = FakeSearchBackend()

    await make_service(backend).search("leave", VIEWER, namespaces=("hr-policy",))

    assert backend.queried_namespaces == ["hr-policy"]


async def test_failed_namespace_is_reported_as_degraded():
    backend = FakeSearchBackend({"payments": [make_chunk()]}, failing_namespaces=("platform",))

    result = await make_service(backend).search("payment outage", VIEWER)

    assert result.degraded == ["namespace_unavailable:platform"]
    assert len(result.chunks) == 1


async def test_local_search_applies_document_filters():
    backend = FakeSearchBackend(embedding_fails=True)
    local_chunks = [
        make_local_chunk("OLD", "internal", "payment outage", created_ts=100),
        make_local_chunk("NEW", "internal", "payment outage", created_ts=900),
    ]

    result = await make_service(backend, local_chunks).search(
        "payment outage", VIEWER, DocumentFilters(created_after_ts=500)
    )

    assert [chunk.doc_id for chunk in result.chunks] == ["NEW"]


async def test_final_check_also_covers_the_local_fallback_path():
    # Simulates a bug in the local index: the final check must still protect the user.
    class LeakyLocalIndex:
        def search(self, *search_arguments):
            return [make_chunk("SECRET#c0", access_level="restricted")]

    service = make_service(FakeSearchBackend(embedding_fails=True))
    service._local_keyword_index = LeakyLocalIndex()

    result = await service.search("payment outage", VIEWER)

    assert result.chunks == []

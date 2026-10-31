from types import SimpleNamespace

from config.settings import EnvironmentSettings, get_model_names
from retrieval.pinecone_client import PineconeSearchBackend
from tests.fakes.fake_search import make_chunk


class RecordingIndex:
    def __init__(self) -> None:
        self.query_arguments: dict = {}

    async def query(self, **query_arguments):
        self.query_arguments = query_arguments
        return SimpleNamespace(matches=[])


async def test_the_access_filter_reaches_the_pinecone_query():
    # Rule 2: if the filter is dropped here, the server returns chunks at every access level.
    settings = EnvironmentSettings(pinecone_api_key="test-key", pinecone_index="test-index")
    backend = PineconeSearchBackend(settings, get_model_names())
    backend._index = RecordingIndex()
    access_filter = {"access_level": {"$in": ["public", "internal"]}}

    await backend.query_namespace("payments", [0.1], None, access_filter, 5)

    assert backend._index.query_arguments["filter"] == access_filter
    assert backend._index.query_arguments["namespace"] == "payments"


async def test_fetched_records_become_typed_chunks():
    class FetchingIndex:
        async def fetch(self, ids, namespace):
            self.namespace = namespace
            metadata = make_chunk("DOC#c0").model_dump(exclude={"chunk_id", "score"})
            # Pinecone leaves out ids that do not exist, rather than raising an error.
            return SimpleNamespace(vectors={"DOC#c0": SimpleNamespace(metadata=metadata)})

    settings = EnvironmentSettings(pinecone_api_key="test-key", pinecone_index="test-index")
    backend = PineconeSearchBackend(settings, get_model_names())
    backend._index = FetchingIndex()

    chunks = await backend.fetch_chunks("payments", ["DOC#c0", "MISSING#c0"])

    assert [chunk.chunk_id for chunk in chunks] == ["DOC#c0"]
    assert chunks[0].score == 0.0
    assert backend._index.namespace == "payments"

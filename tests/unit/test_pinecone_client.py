from types import SimpleNamespace

from config.settings import EnvironmentSettings, get_model_names
from retrieval.pinecone_client import PineconeSearchBackend


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

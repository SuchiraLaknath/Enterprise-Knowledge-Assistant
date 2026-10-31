"""Async Pinecone access: embedding, namespace queries and reranking.

Everything specific to Pinecone stays here, so the search logic can be tested with a fake.
The `FAULTS` setting makes these calls fail on purpose, to demonstrate the fallbacks.
"""

from types import TracebackType
from typing import Any, Self, cast

from pinecone import AsyncPinecone

from config.settings import EnvironmentSettings, ModelNames
from retrieval.models import RerankedPosition, RetrievedChunk


class SimulatedFaultError(RuntimeError):
    """Raised when `FAULTS` asks for a client to fail during a demo or a test."""


class PineconeSearchBackend:
    """Implements `SearchBackend` with the async Pinecone client.

    Use it with `async with`, so that its network connections are closed.
    """

    def __init__(self, settings: EnvironmentSettings, model_names: ModelNames) -> None:
        """Keep the configuration. Connections open in `__aenter__`."""
        self._settings = settings
        self._model_names = model_names
        self._client: AsyncPinecone | None = None
        self._index: Any = None

    async def __aenter__(self) -> Self:
        """Open the client and locate the index."""
        api_key = self._settings.pinecone_api_key.get_secret_value()
        self._client = AsyncPinecone(api_key=api_key)
        self._index = await self._open_index(self._client)
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the index and client connections."""
        if self._index is not None:
            await self._index.close()
        if self._client is not None:
            await self._client.close()

    async def _open_index(self, client: AsyncPinecone) -> Any:
        """Find the index host, because data-plane calls go directly to that host."""
        index_description = await client.indexes.describe(self._settings.pinecone_index)
        index_host = index_description.host
        if not index_host:
            raise RuntimeError(f"Index {self._settings.pinecone_index} has no host yet.")
        return await client.index(host=index_host)

    def _raise_if_fault_enabled(self, fault_name: str) -> None:
        """Fail on purpose when `FAULTS` names this client."""
        if fault_name in self._settings.enabled_faults:
            raise SimulatedFaultError(f"Simulated {fault_name} failure from FAULTS")

    @property
    def _active_client(self) -> AsyncPinecone:
        """Return the open client, with a clear error if `async with` was skipped."""
        if self._client is None:
            raise RuntimeError("Use PineconeSearchBackend inside 'async with'.")
        return self._client

    async def embed_query(self, query_text: str) -> list[float]:
        """Turn a query into a dense vector."""
        self._raise_if_fault_enabled("pinecone")
        embeddings = await self._active_client.inference.embed(
            model=self._model_names.embedding_model,
            inputs=[query_text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        # The SDK types cover sparse models too, but this model always returns a dense list.
        return cast(list[float], embeddings.data[0].values)

    async def query_namespace(
        self,
        namespace: str,
        dense_vector: list[float],
        sparse_vector: dict[str, list[Any]] | None,
        metadata_filter: dict[str, Any],
        result_count: int,
    ) -> list[RetrievedChunk]:
        """Return the best matches from one namespace, as typed records."""
        self._raise_if_fault_enabled("pinecone")
        response = await self._index.query(
            namespace=namespace,
            vector=dense_vector,
            sparse_vector=sparse_vector,
            filter=metadata_filter,
            top_k=result_count,
            include_metadata=True,
        )
        return [
            RetrievedChunk(chunk_id=match.id, score=match.score, **match.metadata)
            for match in response.matches
        ]

    async def fetch_chunks(self, namespace: str, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Return the named chunks. Ids that do not exist are simply missing from the result.

        Rule 2: a fetch has no metadata filter, so the caller must check access afterwards. It
        also has no relevance score, so each chunk scores zero.
        """
        self._raise_if_fault_enabled("pinecone")
        response = await self._index.fetch(ids=chunk_ids, namespace=namespace)
        return [
            RetrievedChunk(chunk_id=chunk_id, score=0.0, **vector.metadata)
            for chunk_id, vector in response.vectors.items()
        ]

    async def rerank(
        self, query_text: str, document_texts: list[str], result_count: int
    ) -> list[RerankedPosition]:
        """Return the most relevant inputs, best first."""
        self._raise_if_fault_enabled("rerank")
        reranked = await self._active_client.inference.rerank(
            model=self._model_names.rerank_model,
            query=query_text,
            documents=document_texts,
            top_n=result_count,
            return_documents=False,
        )
        return [
            RerankedPosition(input_index=ranked_item.index, score=ranked_item.score)
            for ranked_item in reranked.data
        ]

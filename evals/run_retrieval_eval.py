"""Measure retrieval on the golden questions.

It runs hybrid search without the reranker, so that a run spends none of the monthly rerank quota.

Run with: uv run python -m evals.run_retrieval_eval
"""

import asyncio
from pathlib import Path

import yaml

from config.settings import (
    BM25_PARAMETERS_PATH,
    DOCUMENTS_DIRECTORY,
    get_environment_settings,
    get_limits,
    get_model_names,
    get_role_access,
)
from evals.scoring import is_hit, reciprocal_rank, summarize_scores
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.hybrid import load_bm25_encoder
from retrieval.models import UserContext
from retrieval.pinecone_client import PineconeSearchBackend
from retrieval.search import DocumentSearchService

GOLDEN_QUESTIONS_PATH = Path(__file__).parent / "golden_questions.yaml"


async def main() -> None:
    """Search for each golden question, and print how well the expected documents ranked."""
    limits = get_limits()
    encoder = load_bm25_encoder(BM25_PARAMETERS_PATH)
    local_index = LocalKeywordIndex.from_documents(DOCUMENTS_DIRECTORY, encoder, limits.chunking)
    golden_questions = yaml.safe_load(GOLDEN_QUESTIONS_PATH.read_text())["questions"]
    role_access = get_role_access()

    hits: list[bool] = []
    reciprocal_ranks: list[float] = []
    async with PineconeSearchBackend(get_environment_settings(), get_model_names()) as backend:
        breaker = CircuitBreaker(
            limits.circuit_breaker.failure_threshold, limits.circuit_breaker.recovery_seconds
        )
        service = DocumentSearchService(backend, encoder, local_index, breaker, limits.retrieval)
        for golden in golden_questions:
            user = UserContext(
                user_id="eval",
                role=golden["user_role"],
                allowed_access_levels=role_access.roles[golden["user_role"]],
            )
            result = await service.search(golden["question"], user, use_reranker=False)
            found_doc_ids = [chunk.doc_id for chunk in result.chunks]
            hits.append(is_hit(found_doc_ids, golden["expected_doc_ids"]))
            reciprocal_ranks.append(reciprocal_rank(found_doc_ids, golden["expected_doc_ids"]))
            print(f"{hits[-1]!s:<5} {reciprocal_ranks[-1]:.2f}  {golden['question']}")

    print(summarize_scores(hits, reciprocal_ranks))


if __name__ == "__main__":
    asyncio.run(main())

"""Score retrieval results against expected documents. Plain arithmetic, so it can be tested."""


def is_hit(found_doc_ids: list[str], expected_doc_ids: list[str]) -> bool:
    """Return whether any expected document was found."""
    return any(doc_id in expected_doc_ids for doc_id in found_doc_ids)


def reciprocal_rank(found_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Return 1 over the position of the first expected document, or 0 if none was found."""
    for position, doc_id in enumerate(found_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1 / position
    return 0.0


def summarize_scores(hits: list[bool], reciprocal_ranks: list[float]) -> dict[str, float]:
    """Return the hit rate and the mean reciprocal rank across all questions."""
    question_count = len(hits)
    if question_count == 0:
        return {"hit_rate": 0.0, "mean_reciprocal_rank": 0.0}
    return {
        "hit_rate": round(sum(hits) / question_count, 3),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / question_count, 3),
    }

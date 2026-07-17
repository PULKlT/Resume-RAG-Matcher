"""
Standard retrieval metrics computed over a list of retrieved chunk_ids
against a set of ground-truth relevant chunk_ids.
"""


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> dict:
    return {
        "precision@k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall@k": recall_at_k(retrieved_ids, relevant_ids, k),
        "rr": reciprocal_rank(retrieved_ids, relevant_ids),
    }


def aggregate_metrics(per_query_metrics: list[dict]) -> dict:
    """Average precision@k/recall@k across queries, RR averaged = MRR."""
    if not per_query_metrics:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}

    n = len(per_query_metrics)
    return {
        "precision@k": sum(m["precision@k"] for m in per_query_metrics) / n,
        "recall@k": sum(m["recall@k"] for m in per_query_metrics) / n,
        "mrr": sum(m["rr"] for m in per_query_metrics) / n,
    }


if __name__ == "__main__":
    retrieved = ["c1", "c3", "c5", "c7", "c9"]
    relevant = {"c3", "c9", "c11"}

    result = evaluate_query(retrieved, relevant, k=5)
    print(result)

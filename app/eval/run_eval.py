"""
Runs retrieval strategies against the labeled eval set and prints a
precision@k / recall@k / MRR comparison table. This is the numbers table
that goes into README.md "Eval Results".

Usage: python -m app.eval.run_eval
"""
import json
import os

from app.config import FAISS_INDEX_PATH, TOP_K_RERANK
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.eval.metrics import evaluate_query, aggregate_metrics

LABELED_SET_PATH = os.path.join(os.path.dirname(__file__), "labeled_set.json")
EVAL_K = TOP_K_RERANK  # evaluate at the same cutoff used downstream (default 5)


def load_labeled_set(path: str = None) -> list[dict]:
    path = path or LABELED_SET_PATH
    with open(path, "r") as f:
        data = json.load(f)
    queries = data["queries"]
    labeled = [q for q in queries if q["relevant_chunk_ids"]]
    skipped = len(queries) - len(labeled)
    if skipped:
        print(f"[run_eval] Skipping {skipped} unlabeled quer{'y' if skipped == 1 else 'ies'} "
              f"(empty relevant_chunk_ids) — fill in labeled_set.json to include them.")
    return labeled


def run_strategy(name: str, retrieve_fn, labeled_queries: list[dict], k: int = EVAL_K) -> dict:
    per_query = []
    for item in labeled_queries:
        retrieved_ids = retrieve_fn(item["query"])
        relevant_ids = set(item["relevant_chunk_ids"])
        per_query.append(evaluate_query(retrieved_ids, relevant_ids, k=k))
    agg = aggregate_metrics(per_query)
    agg["strategy"] = name
    return agg


def main():
    labeled_queries = load_labeled_set()
    if not labeled_queries:
        print("No labeled queries found. Fill in app/eval/labeled_set.json first.")
        return

    embedder = Embedder()
    vector_store = VectorStore().load(FAISS_INDEX_PATH)
    bm25_store = BM25Store().load(FAISS_INDEX_PATH)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)

    strategies = {
        "dense_only": lambda q: [c.chunk_id for c, _ in vector_store.search(embedder.embed_query(q), top_k=EVAL_K)],
        "bm25_only": lambda q: [c.chunk_id for c, _ in bm25_store.search(q, top_k=EVAL_K)],
        "hybrid_weighted": lambda q: [r.chunk.chunk_id for r in retriever.retrieve(q, top_k=EVAL_K, method="weighted")],
        "hybrid_rrf": lambda q: [r.chunk.chunk_id for r in retriever.retrieve(q, top_k=EVAL_K, method="rrf")],
    }

    results = []
    for name, fn in strategies.items():
        results.append(run_strategy(name, fn, labeled_queries))

    # Print markdown table (paste straight into README.md Eval Results section)
    print(f"\nEvaluated on {len(labeled_queries)} labeled queries @ k={EVAL_K}\n")
    print("| Strategy | Precision@k | Recall@k | MRR |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['strategy']} | {r['precision@k']:.3f} | {r['recall@k']:.3f} | {r['mrr']:.3f} |")


if __name__ == "__main__":
    main()

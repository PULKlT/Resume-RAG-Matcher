"""
Fuses dense (FAISS) and sparse (BM25) retrieval into a single ranked list.
Two fusion strategies:
- weighted: min-max normalize both score sets, blend via HYBRID_ALPHA
- rrf: Reciprocal Rank Fusion (rank-based, scale-free, no normalization needed)

RRF is generally more robust (avoids score-scale mismatch issues between
cosine sim and BM25); weighted is more interpretable/tunable. Compare both
in eval — worth a line in README's "Key Learnings".
"""
from dataclasses import dataclass

from app.config import TOP_K_RETRIEVE, HYBRID_ALPHA
from app.ingestion.chunker import Chunk
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.embeddings.embedder import Embedder


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}  # all equal → treat as max relevance
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store, embedder: Embedder):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = None, alpha: float = None,
                 method: str = "weighted") -> list[RetrievalResult]:
        """
        method: "weighted" (score blend) or "rrf" (rank fusion)
        alpha: weight for dense score in "weighted" mode (0=BM25 only, 1=dense only)
        """
        top_k = top_k or TOP_K_RETRIEVE
        alpha = alpha if alpha is not None else HYBRID_ALPHA

        # over-fetch from each side so fusion has enough candidates to work with
        fetch_k = top_k * 3

        query_vec = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(query_vec, top_k=fetch_k)
        sparse_hits = self.bm25_store.search(query, top_k=fetch_k)

        if method == "rrf":
            return self._fuse_rrf(dense_hits, sparse_hits, top_k)
        elif method == "weighted":
            return self._fuse_weighted(dense_hits, sparse_hits, top_k, alpha)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _fuse_weighted(self, dense_hits, sparse_hits, top_k: int, alpha: float) -> list[RetrievalResult]:
        dense_scores = {c.chunk_id: s for c, s in dense_hits}
        sparse_scores = {c.chunk_id: s for c, s in sparse_hits}

        dense_norm = _min_max_normalize(dense_scores)
        sparse_norm = _min_max_normalize(sparse_scores)

        chunk_lookup = {c.chunk_id: c for c, _ in dense_hits}
        chunk_lookup.update({c.chunk_id: c for c, _ in sparse_hits})

        all_ids = set(dense_norm) | set(sparse_norm)
        fused = []
        for cid in all_ids:
            d = dense_norm.get(cid, 0.0)
            s = sparse_norm.get(cid, 0.0)
            combined = alpha * d + (1 - alpha) * s
            fused.append(RetrievalResult(
                chunk=chunk_lookup[cid], score=combined,
                dense_score=dense_scores.get(cid), sparse_score=sparse_scores.get(cid),
            ))

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]

    def _fuse_rrf(self, dense_hits, sparse_hits, top_k: int, k_constant: int = 60) -> list[RetrievalResult]:
        """RRF score = sum(1 / (k + rank)) across the ranked lists a chunk appears in."""
        chunk_lookup = {}
        rrf_scores: dict[str, float] = {}
        dense_scores, sparse_scores = {}, {}

        for rank, (chunk, score) in enumerate(dense_hits):
            chunk_lookup[chunk.chunk_id] = chunk
            dense_scores[chunk.chunk_id] = score
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k_constant + rank + 1)

        for rank, (chunk, score) in enumerate(sparse_hits):
            chunk_lookup[chunk.chunk_id] = chunk
            sparse_scores[chunk.chunk_id] = score
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k_constant + rank + 1)

        fused = [
            RetrievalResult(
                chunk=chunk_lookup[cid], score=score,
                dense_score=dense_scores.get(cid), sparse_score=sparse_scores.get(cid),
            )
            for cid, score in rrf_scores.items()
        ]
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]


if __name__ == "__main__":
    from app.config import FAISS_INDEX_PATH

    embedder = Embedder()
    vector_store = VectorStore().load(FAISS_INDEX_PATH)
    bm25_store = BM25Store().load(FAISS_INDEX_PATH)

    retriever = HybridRetriever(vector_store, bm25_store, embedder)

    query = "RAG pipeline experience with LangChain"
    for method in ("weighted", "rrf"):
        print(f"\n--- {method} ---")
        results = retriever.retrieve(query, top_k=5, method=method)
        for r in results:
            print(f"{r.score:.3f} (d={r.dense_score} s={r.sparse_score}) {r.chunk.text[:70]}...")

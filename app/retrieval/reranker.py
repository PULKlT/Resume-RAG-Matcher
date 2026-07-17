"""
Reranks a candidate list using a cross-encoder (query+doc jointly scored,
more accurate than bi-encoder cosine sim but slower — only run on the
top-N candidates from retrieval, not the whole corpus).
"""
from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL, TOP_K_RERANK
from app.retrieval.hybrid_retriever import RetrievalResult


class Reranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or RERANKER_MODEL
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = None) -> list[RetrievalResult]:
        """Re-scores results with the cross-encoder and returns top_k, re-sorted."""
        top_k = top_k or TOP_K_RERANK
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        rerank_scores = self.model.predict(pairs)

        for r, score in zip(results, rerank_scores):
            r.score = float(score)  # overwrite fused retrieval score with cross-encoder score

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    from app.config import FAISS_INDEX_PATH
    from app.embeddings.embedder import Embedder
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.bm25_store import BM25Store
    from app.retrieval.hybrid_retriever import HybridRetriever

    embedder = Embedder()
    vector_store = VectorStore().load(FAISS_INDEX_PATH)
    bm25_store = BM25Store().load(FAISS_INDEX_PATH)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)
    reranker = Reranker()

    query = "RAG pipeline experience with LangChain"
    candidates = retriever.retrieve(query, top_k=20, method="rrf")

    print("--- before rerank ---")
    for r in candidates[:5]:
        print(f"{r.score:.3f} {r.chunk.text[:70]}...")

    reranked = reranker.rerank(query, candidates, top_k=5)
    print("\n--- after rerank ---")
    for r in reranked:
        print(f"{r.score:.3f} {r.chunk.text[:70]}...")

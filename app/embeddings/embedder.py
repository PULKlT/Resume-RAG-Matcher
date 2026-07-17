"""
Wraps a local Sentence-Transformers model (default: BGE-small) for embedding
chunks and queries. Fully local — no API calls, no cost.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

# BGE models expect an instruction prefix on queries (not on documents) for best retrieval quality.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self._is_bge = "bge" in self.model_name.lower()

    def embed(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) float32 array, L2-normalized."""
        if isinstance(texts, str):
            texts = [texts]

        if self._is_bge and is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # enables cosine sim via dot product
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query], is_query=True)[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    embedder = Embedder()
    print(f"Model: {embedder.model_name} | dim: {embedder.dimension}")

    docs = ["Built RAG pipelines using LangChain and FAISS.", "5 years experience in backend Python development."]
    query = "candidates with RAG experience"

    doc_vecs = embedder.embed(docs)
    query_vec = embedder.embed_query(query)

    sims = doc_vecs @ query_vec
    for text, sim in zip(docs, sims):
        print(f"{sim:.3f}  {text}")

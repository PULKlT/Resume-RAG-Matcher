"""
FAISS-backed dense vector store. Stores chunk embeddings + metadata,
supports build, save/load, and top-k similarity search.
Uses IndexFlatIP (exact inner-product search) since embeddings are
L2-normalized in Embedder, making inner product == cosine similarity.
Fine at this project's scale (thousands of chunks) — swap for IndexHNSWFlat
if you want to study ANN tradeoffs later.
"""
import os
import pickle
import numpy as np
import faiss

from app.config import FAISS_INDEX_PATH
from app.ingestion.chunker import Chunk

INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
META_FILE = os.path.join(FAISS_INDEX_PATH, "meta.pkl")


class VectorStore:
    def __init__(self, dimension: int = None):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []  # parallel array: index i <-> chunks[i]

    def build(self, chunks: list[Chunk], embeddings: np.ndarray):
        """Build a fresh index from chunks + their embeddings (n, dim)."""
        assert len(chunks) == embeddings.shape[0], "chunks and embeddings length mismatch"
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        """Append to an existing index."""
        if self.index is None:
            self.build(chunks, embeddings)
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """Returns [(chunk, score), ...] sorted by descending similarity."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def search_filtered(self, query_embedding: np.ndarray, top_k: int = 20,
                         doc_type: str = None, section_type: str = None) -> list[tuple[Chunk, float]]:
        """Search with post-filter on metadata (doc_type, section_type). Over-fetches to compensate."""
        raw_results = self.search(query_embedding, top_k=top_k * 4)
        filtered = []
        for chunk, score in raw_results:
            if doc_type and chunk.doc_type != doc_type:
                continue
            if section_type and chunk.metadata.get("section_type") != section_type:
                continue
            filtered.append((chunk, score))
            if len(filtered) >= top_k:
                break
        return filtered

    def save(self, path: str = None):
        index_path = os.path.join(path or FAISS_INDEX_PATH, "index.faiss")
        meta_path = os.path.join(path or FAISS_INDEX_PATH, "meta.pkl")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "dimension": self.dimension}, f)

    def load(self, path: str = None):
        index_path = os.path.join(path or FAISS_INDEX_PATH, "index.faiss")
        meta_path = os.path.join(path or FAISS_INDEX_PATH, "meta.pkl")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.dimension = data["dimension"]
        return self

    def __len__(self):
        return self.index.ntotal if self.index else 0


if __name__ == "__main__":
    from app.ingestion.parser import parse_directory
    from app.ingestion.chunker import chunk_document
    from app.ingestion.metadata_extractor import enrich_chunks
    from app.embeddings.embedder import Embedder
    from app.embeddings.cache import embed_with_cache
    from app.config import RESUME_DIR

    docs = parse_directory(RESUME_DIR, "resume")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy="section")
        chunks = enrich_chunks(chunks)
        all_chunks.extend(chunks)

    embedder = Embedder()
    texts = [c.text for c in all_chunks]
    embeddings = embed_with_cache(embedder, texts)

    store = VectorStore()
    store.build(all_chunks, embeddings)
    store.save()
    print(f"Indexed {len(store)} chunks. Saved to {FAISS_INDEX_PATH}")

    query_vec = embedder.embed_query("RAG pipeline experience")
    results = store.search(query_vec, top_k=5)
    for chunk, score in results:
        print(f"{score:.3f} [{chunk.metadata.get('section_type')}] {chunk.text[:80]}...")

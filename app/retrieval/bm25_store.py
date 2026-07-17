"""
BM25-backed sparse keyword retrieval. Parallel counterpart to vector_store.py —
same chunks, indexed by term frequency instead of embeddings. Used together
in hybrid_retriever.py to fuse dense + sparse rankings.
"""
import os
import re
import pickle

from rank_bm25 import BM25Okapi

from app.config import FAISS_INDEX_PATH  # reuse same data dir for index artifacts
from app.ingestion.chunker import Chunk

BM25_FILE = os.path.join(FAISS_INDEX_PATH, "bm25.pkl")

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer. Swap for a proper tokenizer
    (e.g. nltk) if stemming/stopwords become worth the eval score."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Store:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []  # parallel array, same convention as VectorStore

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """Returns [(chunk, score), ...] sorted by descending BM25 score."""
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]

    def save(self, path: str = None):
        bm25_path = os.path.join(path or FAISS_INDEX_PATH, "bm25.pkl")
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self, path: str = None):
        bm25_path = os.path.join(path or FAISS_INDEX_PATH, "bm25.pkl")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
        return self

    def __len__(self):
        return len(self.chunks)


if __name__ == "__main__":
    from app.ingestion.parser import parse_directory
    from app.ingestion.chunker import chunk_document
    from app.ingestion.metadata_extractor import enrich_chunks
    from app.config import RESUME_DIR

    docs = parse_directory(RESUME_DIR, "resume")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy="section")
        chunks = enrich_chunks(chunks)
        all_chunks.extend(chunks)

    store = BM25Store()
    store.build(all_chunks)
    store.save()
    print(f"Indexed {len(store)} chunks into BM25 store.")

    results = store.search("RAG pipeline LangChain FAISS", top_k=5)
    for chunk, score in results:
        print(f"{score:.3f} [{chunk.metadata.get('section_type')}] {chunk.text[:80]}...")

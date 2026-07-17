"""
Caches embeddings on disk keyed by (model_name, text_hash) so re-running
ingestion doesn't re-embed unchanged chunks. Backed by a single pickle file
for simplicity — fine at this project's scale (hundreds-thousands of chunks).
"""
import hashlib
import os
import pickle
import numpy as np

from app.config import DATA_DIR

CACHE_PATH = os.path.join(DATA_DIR, "embedding_cache.pkl")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path or CACHE_PATH
        self._cache: dict[str, np.ndarray] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def _key(self, model_name: str, text: str) -> str:
        return f"{model_name}::{_hash_text(text)}"

    def get(self, model_name: str, text: str) -> np.ndarray | None:
        return self._cache.get(self._key(model_name, text))

    def set(self, model_name: str, text: str, embedding: np.ndarray):
        self._cache[self._key(model_name, text)] = embedding

    def get_many(self, model_name: str, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Returns (results with None for misses, indices of misses)."""
        results = []
        miss_indices = []
        for i, t in enumerate(texts):
            hit = self.get(model_name, t)
            results.append(hit)
            if hit is None:
                miss_indices.append(i)
        return results, miss_indices

    def set_many(self, model_name: str, texts: list[str], embeddings: np.ndarray):
        for t, e in zip(texts, embeddings):
            self.set(model_name, t, e)
        self._save()

    def __len__(self):
        return len(self._cache)


def embed_with_cache(embedder, texts: list[str], cache: EmbeddingCache = None) -> np.ndarray:
    """Drop-in wrapper around Embedder.embed() that checks/populates the cache."""
    cache = cache or EmbeddingCache()
    model_name = embedder.model_name

    cached, miss_indices = cache.get_many(model_name, texts)

    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        new_embeddings = embedder.embed(miss_texts)
        cache.set_many(model_name, miss_texts, new_embeddings)
        for idx, emb in zip(miss_indices, new_embeddings):
            cached[idx] = emb

    return np.vstack(cached)


if __name__ == "__main__":
    from app.embeddings.embedder import Embedder

    embedder = Embedder()
    cache = EmbeddingCache()

    texts = ["Built RAG pipelines using LangChain.", "5 years backend Python experience."]

    print(f"Cache size before: {len(cache)}")
    vecs = embed_with_cache(embedder, texts, cache)
    print(f"Cache size after: {len(cache)} | shape: {vecs.shape}")

    # Second call should be all cache hits
    vecs2 = embed_with_cache(embedder, texts, cache)
    print(f"Cache size after re-run (should be unchanged): {len(cache)}")

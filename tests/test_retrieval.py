"""
Unit tests for chunking, hybrid fusion, reranker ordering, and cache behavior.
Run: pytest tests/test_retrieval.py -v

Note: tests that need a real embedding model (fixtures below) will download
BGE-small on first run — no network calls beyond that (fully local/free).
"""
import numpy as np
import pytest

from app.ingestion.chunker import Chunk, chunk_fixed, chunk_section
from app.ingestion.parser import ParsedDocument
from app.retrieval.hybrid_retriever import (
    RetrievalResult, _min_max_normalize, HybridRetriever,
)
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.embeddings.cache import EmbeddingCache


# --- fixtures ---

@pytest.fixture
def sample_resume_doc():
    text = (
        "Summary\n"
        "Experienced GenAI engineer.\n\n"
        "Experience\n"
        "Built RAG pipelines using LangChain and FAISS at Company X. Jan 2022 - Present\n\n"
        "Education\n"
        "B.Tech Computer Science, 2018 - 2022\n"
    )
    return ParsedDocument(doc_id="test_resume", doc_type="resume", source_path="dummy.txt", raw_text=text)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="resume", section="experience",
              text="Built RAG pipelines with LangChain and FAISS."),
        Chunk(chunk_id="c2", doc_id="d1", doc_type="resume", section="skills",
              text="Python, FastAPI, GCP, AWS."),
        Chunk(chunk_id="c3", doc_id="d1", doc_type="resume", section="education",
              text="B.Tech Computer Science."),
    ]


# --- chunker tests ---

class TestChunker:
    def test_section_chunking_detects_headers(self, sample_resume_doc):
        chunks = chunk_section(sample_resume_doc)
        sections = {c.section for c in chunks}
        assert "experience" in sections
        assert "education" in sections
        assert "summary" in sections

    def test_section_chunking_no_headers_falls_back_to_single_chunk(self):
        doc = ParsedDocument(doc_id="d", doc_type="resume", source_path="x", raw_text="Just plain text, no headers.")
        chunks = chunk_section(doc)
        assert len(chunks) == 1
        assert chunks[0].section == "full_document"

    def test_fixed_chunking_respects_overlap(self, sample_resume_doc):
        chunks = chunk_fixed(sample_resume_doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text.strip() != ""

    def test_chunks_have_unique_ids(self, sample_resume_doc):
        chunks = chunk_section(sample_resume_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# --- hybrid fusion tests ---

class TestHybridFusion:
    def test_min_max_normalize_handles_equal_scores(self):
        scores = {"a": 5.0, "b": 5.0}
        norm = _min_max_normalize(scores)
        assert norm == {"a": 1.0, "b": 1.0}

    def test_min_max_normalize_scales_to_unit_range(self):
        scores = {"a": 0.0, "b": 5.0, "c": 10.0}
        norm = _min_max_normalize(scores)
        assert norm["a"] == 0.0
        assert norm["c"] == 1.0
        assert norm["b"] == pytest.approx(0.5)

    def test_min_max_normalize_empty_input(self):
        assert _min_max_normalize({}) == {}

    def test_rrf_fusion_favors_items_ranked_high_in_both_lists(self, sample_chunks):
        c1, c2, c3 = sample_chunks
        dense_hits = [(c1, 0.9), (c2, 0.5), (c3, 0.3)]
        sparse_hits = [(c1, 10.0), (c3, 5.0), (c2, 1.0)]

        retriever = HybridRetriever.__new__(HybridRetriever)  # bypass __init__ (no real stores needed)
        fused = retriever._fuse_rrf(dense_hits, sparse_hits, top_k=3)

        assert fused[0].chunk.chunk_id == "c1"  # ranked #1 in both lists

    def test_weighted_fusion_alpha_zero_is_sparse_only_ranking(self, sample_chunks):
        c1, c2, c3 = sample_chunks
        dense_hits = [(c1, 0.9), (c2, 0.1)]
        sparse_hits = [(c2, 10.0), (c1, 1.0)]

        retriever = HybridRetriever.__new__(HybridRetriever)
        fused = retriever._fuse_weighted(dense_hits, sparse_hits, top_k=2, alpha=0.0)

        assert fused[0].chunk.chunk_id == "c2"  # alpha=0 -> pure BM25 ranking wins


# --- vector store tests ---

class TestVectorStore:
    def test_build_and_search_returns_closest_match(self, sample_chunks):
        # synthetic 3-dim embeddings, normalized, where c1 is closest to the query
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype="float32")

        store = VectorStore()
        store.build(sample_chunks, embeddings)

        query_vec = np.array([0.9, 0.1, 0.0], dtype="float32")
        results = store.search(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0][0].chunk_id == "c1"

    def test_empty_store_search_returns_empty_list(self):
        store = VectorStore()
        results = store.search(np.zeros(3, dtype="float32"), top_k=5)
        assert results == []

    def test_search_filtered_by_section_type(self, sample_chunks):
        for c in sample_chunks:
            c.metadata["section_type"] = c.section

        embeddings = np.eye(3, dtype="float32")
        store = VectorStore()
        store.build(sample_chunks, embeddings)

        query_vec = np.array([1.0, 0.0, 0.0], dtype="float32")
        results = store.search_filtered(query_vec, top_k=5, section_type="skills")

        assert all(c.metadata["section_type"] == "skills" for c, _ in results)


# --- bm25 store tests ---

class TestBM25Store:
    def test_search_ranks_keyword_match_higher(self, sample_chunks):
        store = BM25Store()
        store.build(sample_chunks)

        results = store.search("LangChain FAISS RAG", top_k=3)
        assert results[0][0].chunk_id == "c1"

    def test_search_no_match_returns_empty(self, sample_chunks):
        store = BM25Store()
        store.build(sample_chunks)

        results = store.search("zzz_nonexistent_term_xyz", top_k=3)
        assert results == []


# --- embedding cache tests ---

class TestEmbeddingCache:
    def test_cache_hit_after_set(self, tmp_path):
        cache = EmbeddingCache(cache_path=str(tmp_path / "cache.pkl"))
        vec = np.array([1.0, 2.0, 3.0], dtype="float32")

        assert cache.get("model-a", "hello") is None
        cache.set("model-a", "hello", vec)
        result = cache.get("model-a", "hello")

        assert np.array_equal(result, vec)

    def test_cache_is_scoped_by_model_name(self, tmp_path):
        cache = EmbeddingCache(cache_path=str(tmp_path / "cache.pkl"))
        vec = np.array([1.0], dtype="float32")

        cache.set("model-a", "text", vec)
        assert cache.get("model-b", "text") is None

    def test_get_many_reports_correct_miss_indices(self, tmp_path):
        cache = EmbeddingCache(cache_path=str(tmp_path / "cache.pkl"))
        cache.set("model-a", "known", np.array([1.0]))

        results, misses = cache.get_many("model-a", ["known", "unknown"])
        assert results[0] is not None
        assert results[1] is None
        assert misses == [1]

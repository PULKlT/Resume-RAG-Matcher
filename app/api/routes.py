"""
API routes. Components are loaded lazily/once at module level (embedder,
stores) rather than per-request — rebuilding FAISS/BM25 indices or reloading
the embedding model on every call would be wasteful.

Run `POST /ingest` after adding files to data/resumes or data/job_descriptions,
before querying /match or /query.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import FAISS_INDEX_PATH, RESUME_DIR, TOP_K_RETRIEVE, TOP_K_RERANK
from app.ingestion.parser import parse_directory
from app.ingestion.chunker import chunk_document
from app.ingestion.metadata_extractor import enrich_chunks
from app.embeddings.embedder import Embedder
from app.embeddings.cache import embed_with_cache
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.hyde import HydeRetriever
from app.query.query_expansion import multi_query_retrieve
from app.query.synthesizer import synthesize

router = APIRouter()

# --- lazy singletons ---
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_bm25_store: BM25Store | None = None
_reranker: Reranker | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_stores() -> tuple[VectorStore, BM25Store]:
    global _vector_store, _bm25_store
    if _vector_store is None or _bm25_store is None:
        try:
            _vector_store = VectorStore().load(FAISS_INDEX_PATH)
            _bm25_store = BM25Store().load(FAISS_INDEX_PATH)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="No index found. Call POST /ingest first.")
    return _vector_store, _bm25_store


# --- request/response models ---
class MatchRequest(BaseModel):
    query: str
    top_k: int = TOP_K_RERANK
    fusion_method: str = "rrf"          # "weighted" | "rrf"
    use_hyde: bool = False
    use_query_expansion: bool = False


class MatchResponse(BaseModel):
    query: str
    results: list[dict]
    synthesized_bullets: str | None = None


# --- endpoints ---
@router.post("/ingest")
def ingest():
    """Parse resumes, chunk, embed, and (re)build FAISS + BM25 indices."""
    docs = parse_directory(RESUME_DIR, "resume")
    if not docs:
        raise HTTPException(status_code=400, detail=f"No resume files found in {RESUME_DIR}")

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        chunks = enrich_chunks(chunks)
        all_chunks.extend(chunks)

    embedder = get_embedder()
    texts = [c.text for c in all_chunks]
    embeddings = embed_with_cache(embedder, texts)

    global _vector_store, _bm25_store
    _vector_store = VectorStore()
    _vector_store.build(all_chunks, embeddings)
    _vector_store.save()

    _bm25_store = BM25Store()
    _bm25_store.build(all_chunks)
    _bm25_store.save()

    return {"status": "ok", "documents_parsed": len(docs), "chunks_indexed": len(all_chunks)}


@router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """Retrieve + rerank + synthesize tailored bullets for a JD/query."""
    vector_store, bm25_store = get_stores()
    embedder = get_embedder()
    reranker = get_reranker()
    retriever = HybridRetriever(vector_store, bm25_store, embedder)

    if req.use_query_expansion:
        candidates = multi_query_retrieve(req.query, retriever, top_k=TOP_K_RETRIEVE)
    elif req.use_hyde:
        hyde = HydeRetriever(vector_store, embedder)
        candidates = hyde.retrieve(req.query, top_k=TOP_K_RETRIEVE)
    else:
        candidates = retriever.retrieve(req.query, top_k=TOP_K_RETRIEVE, method=req.fusion_method)

    top_results = reranker.rerank(req.query, candidates, top_k=req.top_k)
    bullets = synthesize(req.query, top_results)

    return MatchResponse(
        query=req.query,
        results=[
            {
                "chunk_id": r.chunk.chunk_id,
                "text": r.chunk.text,
                "section": r.chunk.metadata.get("section_type"),
                "score": round(r.score, 4),
            }
            for r in top_results
        ],
        synthesized_bullets=bullets,
    )


@router.get("/health")
def health():
    return {"status": "ok"}

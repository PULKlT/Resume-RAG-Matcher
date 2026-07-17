"""
FastAPI entrypoint. Run: uvicorn app.main:app --reload
"""
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Resume-RAG-Matcher",
    description="Hybrid retrieval (BM25 + FAISS) + reranking for matching resumes to job descriptions.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "resume-rag-matcher",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest (POST)", "/match (POST)"],
    }

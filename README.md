# Resume-RAG-Matcher

A RAG system that matches resumes to job descriptions using hybrid retrieval (BM25 + dense embeddings) and cross-encoder reranking, with an eval harness to measure retrieval quality across strategies.

## Why this project

Built to go deep on RAG internals: chunking strategy tradeoffs, hybrid search fusion, reranking impact, and query expansion — with numbers, not just intuition.

## Features

- **Multi-strategy chunking**: fixed-size, semantic, and section-based (compare retrieval quality per strategy)
- **Hybrid retrieval**: BM25 (sparse) + FAISS (dense) fused ranking
- **Cross-encoder reranking**: ms-marco-MiniLM reranks top-k candidates
- **Query expansion**: LLM rewrites vague queries into better search terms
- **HyDE**: hypothetical document embeddings as an alternative retrieval path
- **Eval harness**: precision@k, recall@k, MRR on a hand-labeled query-chunk set

## Architecture

```
Resume/JD → Parser → Chunker → Embedder → FAISS + BM25 index
                                                  ↓
Query → Query Expansion → Hybrid Retriever → Reranker → Synthesizer → Tailored output
```

## Tech Stack

100% free — no paid APIs required.

| Component | Tool |
|---|---|
| Backend | FastAPI |
| Embeddings | BGE / Sentence-Transformers (local) |
| Vector store | FAISS (local) |
| Sparse retrieval | BM25 (rank_bm25) |
| Reranker | Cross-encoder (ms-marco-MiniLM, local) |
| LLM | Ollama (local, default) or Groq free tier (hosted alt) |
| Frontend | Streamlit |

## Setup

```bash
git clone <repo-url>
cd resume-rag-matcher
pip install -r requirements.txt
cp .env.example .env
```

**LLM setup (choose one — both free):**

- **Ollama (default, fully local, no signup)**
  ```bash
  # install from https://ollama.com, then:
  ollama pull llama3.1:8b
  ```
- **Groq (hosted, free tier)**
  Get a key at https://console.groq.com, set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env`.

## Usage

```bash
# Build index from resumes + JDs
python -m app.ingestion.parser --input data/resumes

# Run API
uvicorn app.main:app --reload

# Run frontend
streamlit run frontend/streamlit_app.py

# Run eval
python -m app.eval.run_eval
```

## Eval Results

| Strategy | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| Dense only (baseline) | TBD | TBD | TBD |
| BM25 only | TBD | TBD | TBD |
| Hybrid (dense + BM25) | TBD | TBD | TBD |
| Hybrid + Reranker | TBD | TBD | TBD |
| Hybrid + Reranker + HyDE | TBD | TBD | TBD |

*(Table filled in as `eval/run_eval.py` is run against the labeled set)*

## Key Learnings / Notes

- Chunking strategy comparison: _fill in after `chunking_experiments.ipynb`_
- Reranking impact: _fill in_
- HyDE vs direct dense retrieval: _fill in_

## Roadmap

- [ ] Ingestion + chunking
- [ ] Embeddings + FAISS index
- [ ] BM25 + hybrid fusion
- [ ] Baseline eval
- [ ] Reranker + eval comparison
- [ ] Query expansion / HyDE
- [ ] Synthesizer (tailored bullet generation)
- [ ] API + frontend

## License

MIT

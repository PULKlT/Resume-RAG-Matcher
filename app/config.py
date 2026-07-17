import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
# "ollama" (fully local, free, no signup) or "groq" (free-tier hosted API)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama (local) settings — requires Ollama app running: https://ollama.com
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # or "mistral"

# Groq (free tier hosted) settings — get key at https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- Embeddings (local, free) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Reranker (local, free) ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
JD_DIR = os.path.join(DATA_DIR, "job_descriptions")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- Retrieval ---
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 20))   # candidates before rerank
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))        # final results after rerank
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))    # weight: dense vs BM25 (0=BM25 only, 1=dense only)

# --- Chunking ---
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "section")  # "fixed" | "semantic" | "section"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))            # tokens, used for "fixed" strategy
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

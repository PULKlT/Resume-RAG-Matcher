"""
HyDE (Hypothetical Document Embeddings): instead of embedding the raw query,
ask an LLM to generate a hypothetical ideal document/passage that would answer
the query, then embed THAT and search with it. Often improves recall on vague
or underspecified queries vs direct query embedding.

Includes a minimal local LLM call (Ollama default / Groq alt) — this will get
extracted into a shared app/llm/client.py once query_expansion.py and
synthesizer.py need the same call, to avoid duplicating it three times.
"""
import requests

from app.config import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL,
    TOP_K_RETRIEVE,
)
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid_retriever import RetrievalResult

HYDE_PROMPT_TEMPLATE = """You are writing a short passage that would appear in an ideal candidate's resume, \
matching the following job requirement or search query as closely as possible.
Write 2-3 sentences, concrete and specific, as if extracted directly from a resume.

Query: {query}

Passage:"""


def call_llm(prompt: str, max_tokens: int = 150) -> str:
    if LLM_PROVIDER == "ollama":
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    elif LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


class HydeRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        return call_llm(prompt)

    def retrieve(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        top_k = top_k or TOP_K_RETRIEVE
        hypothetical_doc = self.generate_hypothetical_document(query)

        # embed the hypothetical doc as a document (not query) — it's meant to
        # resemble the target passages, not a search query
        hyde_vec = self.embedder.embed([hypothetical_doc], is_query=False)[0]

        hits = self.vector_store.search(hyde_vec, top_k=top_k)
        return [RetrievalResult(chunk=c, score=s, dense_score=s) for c, s in hits]


if __name__ == "__main__":
    from app.config import FAISS_INDEX_PATH

    embedder = Embedder()
    vector_store = VectorStore().load(FAISS_INDEX_PATH)
    hyde = HydeRetriever(vector_store, embedder)

    query = "system design experience"
    hypothetical = hyde.generate_hypothetical_document(query)
    print(f"Hypothetical doc:\n{hypothetical}\n")

    results = hyde.retrieve(query, top_k=5)
    print("--- HyDE retrieval results ---")
    for r in results:
        print(f"{r.score:.3f} {r.chunk.text[:70]}...")

"""
Takes retrieved resume chunks + a job description query, and synthesizes
tailored output — e.g. "which of my projects best match this JD" with
LLM-generated framing grounded in the actual retrieved text (not hallucinated).
"""
from app.llm.client import call_llm
from app.retrieval.hybrid_retriever import RetrievalResult

SYNTHESIS_PROMPT = """You are helping a candidate tailor their resume content to a job description.
Below are the candidate's most relevant resume excerpts (retrieved by search), followed by the \
job requirement they're targeting. Using ONLY the information in these excerpts — do not invent \
experience — write 2-4 tailored bullet points that best position the candidate for this requirement.

Resume excerpts:
{context}

Job requirement: {query}

Tailored bullet points:"""


def format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        section = r.chunk.metadata.get("section_type", "unknown")
        blocks.append(f"[{i}] ({section}) {r.chunk.text}")
    return "\n\n".join(blocks)


def synthesize(query: str, results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant resume content found for this query."

    context = format_context(results)
    prompt = SYNTHESIS_PROMPT.format(context=context, query=query)
    return call_llm(prompt, max_tokens=300, temperature=0.4)


if __name__ == "__main__":
    from app.config import FAISS_INDEX_PATH
    from app.embeddings.embedder import Embedder
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.bm25_store import BM25Store
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker

    embedder = Embedder()
    vector_store = VectorStore().load(FAISS_INDEX_PATH)
    bm25_store = BM25Store().load(FAISS_INDEX_PATH)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)
    reranker = Reranker()

    query = "Looking for a candidate with production RAG system experience"
    candidates = retriever.retrieve(query, top_k=20, method="rrf")
    top_results = reranker.rerank(query, candidates, top_k=5)

    output = synthesize(query, top_results)
    print(output)

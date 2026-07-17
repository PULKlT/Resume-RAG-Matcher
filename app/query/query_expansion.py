"""
Rewrites vague/underspecified queries into more retrieval-friendly forms
before hitting the hybrid retriever. E.g. "system design experience" ->
explicit terms like "scalability", "microservices", "load balancing", etc.

Two modes:
- rewrite: single improved query string
- multi_query: generate N variants, retrieve with each, union results
  (classic multi-query RAG pattern — helps when one phrasing misses relevant chunks)
"""
import re

from app.llm.client import call_llm

REWRITE_PROMPT = """Rewrite the following search query to be more specific and effective for \
searching resumes/job descriptions. Expand vague terms into concrete skills, tools, or \
concepts a resume might actually contain. Return ONLY the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""

MULTI_QUERY_PROMPT = """Generate {n} different search query variants for the following query, \
each focusing on a different angle (specific tools, general skill, related concepts). \
Return ONLY the queries, one per line, no numbering, no extra text.

Original query: {query}

Variants:"""


def rewrite_query(query: str) -> str:
    prompt = REWRITE_PROMPT.format(query=query)
    rewritten = call_llm(prompt, max_tokens=60, temperature=0.3)
    # strip quotes/formatting the LLM sometimes adds
    return rewritten.strip('"\' \n')


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
    raw = call_llm(prompt, max_tokens=150, temperature=0.7)
    variants = [line.strip("-• \n") for line in raw.split("\n") if line.strip()]
    return variants[:n] if variants else [query]


def multi_query_retrieve(query: str, retriever, top_k: int = 20, n_variants: int = 3):
    """Retrieve with original + N variants, union + dedupe by chunk_id, keep best score per chunk."""
    queries = [query] + generate_query_variants(query, n=n_variants)

    best_by_id = {}
    for q in queries:
        results = retriever.retrieve(q, top_k=top_k, method="rrf")
        for r in results:
            cid = r.chunk.chunk_id
            if cid not in best_by_id or r.score > best_by_id[cid].score:
                best_by_id[cid] = r

    fused = sorted(best_by_id.values(), key=lambda r: r.score, reverse=True)
    return fused[:top_k]


if __name__ == "__main__":
    query = "system design experience"

    print(f"Original: {query}")
    print(f"Rewritten: {rewrite_query(query)}")
    print(f"Variants: {generate_query_variants(query, n=3)}")

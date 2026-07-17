"""
Streamlit UI for Resume-RAG-Matcher. Talks to the FastAPI backend over HTTP.
Run: streamlit run frontend/streamlit_app.py
Requires the API running separately: uvicorn app.main:app --reload
"""
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume-RAG-Matcher", layout="wide")
st.title("Resume-RAG-Matcher")
st.caption("Hybrid retrieval (BM25 + FAISS) + reranking, fully local/free stack.")

# --- Sidebar: ingestion ---
with st.sidebar:
    st.header("1. Index resumes")
    st.write("Drop files into `data/resumes/` on disk, then rebuild the index below.")

    if st.button("Rebuild index (POST /ingest)", type="primary"):
        with st.spinner("Parsing, chunking, embedding, indexing..."):
            try:
                resp = requests.post(f"{API_BASE}/ingest", timeout=300)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Indexed {data['chunks_indexed']} chunks from {data['documents_parsed']} document(s).")
            except requests.exceptions.RequestException as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    st.header("Retrieval options")
    fusion_method = st.selectbox("Fusion method", ["rrf", "weighted"], index=0)
    use_hyde = st.checkbox("Use HyDE", value=False, help="Generate a hypothetical passage via LLM, embed & search with that instead of the raw query.")
    use_expansion = st.checkbox("Use query expansion", value=False, help="Generate query variants via LLM, retrieve with each, union results.")
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)

# --- Main: query ---
st.header("2. Match a job description / query")
query = st.text_area("Job requirement or search query", placeholder="e.g. Looking for a candidate with production RAG system experience")

if st.button("Search", type="primary", disabled=not query.strip()):
    with st.spinner("Retrieving, reranking, synthesizing..."):
        payload = {
            "query": query,
            "top_k": top_k,
            "fusion_method": fusion_method,
            "use_hyde": use_hyde,
            "use_query_expansion": use_expansion,
        }
        try:
            resp = requests.post(f"{API_BASE}/match", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            st.subheader("Tailored bullet points")
            st.markdown(data["synthesized_bullets"] or "_No output generated._")

            st.subheader(f"Retrieved chunks ({len(data['results'])})")
            for r in data["results"]:
                with st.expander(f"[{r['section']}] score={r['score']:.3f} — {r['chunk_id']}"):
                    st.write(r["text"])

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 400:
                st.warning("No index found yet — click 'Rebuild index' in the sidebar first.")
            else:
                st.error(f"Request failed: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API at {API_BASE}. Is `uvicorn app.main:app --reload` running? ({e})")

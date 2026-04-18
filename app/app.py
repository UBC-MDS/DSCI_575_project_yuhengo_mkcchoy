import os
import sys
from collections import defaultdict

import nltk
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.bm25 import load_bm25, bm25_search
from src.semantic import load_faiss, semantic_search
from src.rag_pipeline import build_hybrid_rag_chain

BM25_DIR = os.path.join(PROJECT_ROOT, "bm25_index")
SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "semantic_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DISPLAY_K = 3
RETRIEVAL_K = 10


@st.cache_resource
def load_resources():
    bm25, _ = load_bm25(BM25_DIR)
    index, documents = load_faiss(SEMANTIC_DIR)
    model = SentenceTransformer(MODEL_NAME)
    hybrid_rag_chain = build_hybrid_rag_chain(docs=documents, k=5)
    return bm25, index, documents, model, hybrid_rag_chain


def extract_title(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else "Untitled result"


def extract_snippet(text, max_chars=1000):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = " ".join(lines[1:]) if len(lines) > 1 else lines[0] if lines else ""
    return body[:max_chars].rstrip() + ("..." if len(body) > max_chars else "")


def truncate_text(text, max_chars=1000):
    text = (text or "").strip()
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def format_rating(rating):
    if rating is None:
        return "N/A"

    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return "N/A"

    rounded = max(0, min(5, round(rating_value)))
    stars = "★" * rounded + "☆" * (5 - rounded)
    return f"{stars} ({rating_value:.1f})"


def result_key(result):
    metadata = result.get("metadata", {})
    return (metadata.get("source"), metadata.get("asin"), result.get("text"))


def reciprocal_rank_fusion(result_lists, k=60):
    fused_scores = defaultdict(float)
    fused_results = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = result_key(result)
            fused_scores[key] += 1.0 / (k + rank)
            fused_results[key] = result

    ranked = []
    for key, result in fused_results.items():
        item = dict(result)
        item["hybrid_score"] = fused_scores[key]
        ranked.append(item)

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return ranked


def run_search(query, mode, bm25, index, documents, model):
    if mode == "BM25":
        return bm25_search(query, bm25, documents, k=DISPLAY_K)

    if mode == "Semantic":
        return semantic_search(query, index, model, documents, k=DISPLAY_K)

    bm25_results = bm25_search(query, bm25, documents, k=RETRIEVAL_K)
    semantic_results = semantic_search(query, index, model, documents, k=RETRIEVAL_K)
    fused_results = reciprocal_rank_fusion([bm25_results, semantic_results])
    return fused_results[:DISPLAY_K]


def render_result_card(result, mode, show_score=True, show_snippet=True):
    metadata = result.get("metadata", {})
    title = extract_title(result.get("text", ""))
    snippet = extract_snippet(result.get("text", ""))
    rating = metadata.get("rating")
    source = metadata.get("source", "unknown")
    asin = metadata.get("asin", "N/A")

    with st.container(border=True):
        st.markdown(f"#### {title}")

        col1, col2, col3 = st.columns(3)
        col1.caption(f"Source: {source}")
        col2.caption(f"ASIN: {asin}")
        col3.caption(f"Average rating: {format_rating(rating)}")

        if show_score:
            if mode == "BM25":
                st.caption(f"BM25 score: {result['score']:.3f}")
            elif mode == "Semantic":
                st.caption(f"Semantic distance: {result['score']:.3f}")
            else:
                st.caption(f"Hybrid score: {result['hybrid_score']:.4f}")

        if show_snippet and snippet:
            st.write(snippet)


def main():
    st.set_page_config(
        page_title="Amazon Product Query Assistant", page_icon="🛒", layout="wide"
    )

    st.title("🛒 Amazon Product Query Assistant")
    st.write(
        "A simple product assistant result-only search and Hybrid RAG."
    )

    bm25, index, documents, model, hybrid_rag_chain = load_resources()

    with st.sidebar:
        st.header("Display Options")
        show_score = st.checkbox("Show retrieval score", value=True)
        show_snippet = st.checkbox("Show text snippet (max. 1000 characters)", value=True)

    tab_retrieval, tab_rag = st.tabs(["Retrieval Mode", "RAG Mode"])

    with tab_retrieval:
        st.subheader("Result-Only Search")
        st.caption("Search from product metadata and reviews.")

        retrieval_mode = st.radio(
            "Search mode",
            ["BM25", "Semantic", "Hybrid"],
            index=0,
            horizontal=True,
            key="retrieval_mode",
        )

        with st.form("retrieval_form"):
            retrieval_query = st.text_input(
                "Enter a product query",
                value="quiet dishwasher stainless steel",
                key="retrieval_query",
            )
            retrieval_submitted = st.form_submit_button("Search")

        if retrieval_submitted and retrieval_query.strip():
            results = run_search(
                retrieval_query.strip(), retrieval_mode, bm25, index, documents, model
            )

            st.subheader(f"Top {len(results)} results for: {retrieval_query}")
            for i, result in enumerate(results, start=1):
                st.markdown(f"### Result {i}")
                render_result_card(
                    result,
                    mode=retrieval_mode,
                    show_score=show_score,
                    show_snippet=show_snippet,
                )

    with tab_rag:
        st.subheader("Hybrid RAG")
        st.caption(
            "Generate an LLM-powered answer, then inspect the top retrieved products."
        )

        with st.form("rag_form"):
            rag_query = st.text_input(
                "Enter a product question",
                value="best dishwasher for a small apartment under $800",
                key="rag_query",
            )
            rag_submitted = st.form_submit_button("Ask")

        if rag_submitted and rag_query.strip():
            query = rag_query.strip()

            with st.spinner("Generating answer..."):
                rag_answer = hybrid_rag_chain.invoke(query)

            st.markdown("### Generated Answer")
            with st.container(border=True):
                st.write(truncate_text(rag_answer, max_chars=1000))

            rag_results = run_search(query, "Hybrid", bm25, index, documents, model)

            st.markdown("### Retrieved Products")
            for i, result in enumerate(rag_results, start=1):
                st.markdown(f"### Result {i}")
                render_result_card(
                    result,
                    mode="Hybrid",
                    show_score=show_score,
                    show_snippet=show_snippet,
                )


if __name__ == "__main__":
    main()

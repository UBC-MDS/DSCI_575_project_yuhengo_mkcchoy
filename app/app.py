import os
import sys
from collections import defaultdict

import nltk
import streamlit as st
from sentence_transformers import SentenceTransformer

from bm25 import load_bm25, bm25_search
from semantic import load_faiss, semantic_search

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.append(PROJECT_ROOT)

BM25_DIR = os.path.join(PROJECT_ROOT, "bm25_index")
SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "semantic_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DISPLAY_K = 3
RETRIEVAL_K = 10  # more for hybrid search


@st.cache_resource
def load_resources():
    bm25, _ = load_bm25(BM25_DIR)
    index, documents = load_faiss(SEMANTIC_DIR)
    model = SentenceTransformer(MODEL_NAME)
    return bm25, index, documents, model


def extract_title(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else "Untitled result"


def extract_snippet(text, max_chars=200):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = " ".join(lines[1:]) if len(lines) > 1 else lines[0] if lines else ""
    return body[:max_chars].rstrip() + ("..." if len(body) > max_chars else "")


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
        col3.caption(f"Rating: {format_rating(rating)}")

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
        "A simple retrieval prototype using BM25, semantic search, and hybrid search."
    )

    bm25, index, documents, model = load_resources()

    with st.sidebar:
        st.header("Search Options")
        mode = st.radio("Search mode", ["BM25", "Semantic", "Hybrid"], index=0)
        show_score = st.checkbox("Show retrieval score", value=True)
        show_snippet = st.checkbox("Show text snippet", value=True)

    with st.form("search_form"):
        query = st.text_input(
            "Enter a product query", value="quiet dishwasher stainless steel"
        )
        submitted = st.form_submit_button("Search")

    if submitted and query.strip():
        results = run_search(query.strip(), mode, bm25, index, documents, model)

        st.subheader(f"Top {len(results)} results for: {query}")
        for i, result in enumerate(results, start=1):
            st.markdown(f"### Result {i}")
            render_result_card(
                result, mode=mode, show_score=show_score, show_snippet=show_snippet
            )


if __name__ == "__main__":
    main()

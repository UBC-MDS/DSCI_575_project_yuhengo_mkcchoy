import gzip
import json
import os

import pandas as pd
from langchain_core.documents import Document

from bm25 import build_bm25, save_bm25
from semantic import build_embeddings, build_faiss_index, save_faiss


META_TEXT_FIELDS = ["title", "description", "features"]
REVIEW_TEXT_FIELDS = ["title", "text"]


def load_jsonl_gz(path, n=None):
    """Load data from a gzipped file, optionally limiting to n records."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            records.append(json.loads(line))
    return records


def build_documents(meta_records, review_records):
    """Combine metadata and review records into a list of Document objects."""
    meta_df = pd.DataFrame(meta_records)
    review_df = pd.DataFrame(review_records)

    documents = []

    for _, row in meta_df.iterrows():
        text_parts = []

        for field in META_TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
            elif isinstance(value, list):
                cleaned = [str(v).strip() for v in value if str(v).strip()]
                if cleaned:
                    text_parts.append("\n".join(cleaned))

        if not text_parts:
            continue

        documents.append(
            Document(
                page_content="\n\n".join(text_parts),
                metadata={
                    "source": "metadata",
                    "asin": row.get("parent_asin"),
                    "main_category": row.get("main_category"),
                    "categories": row.get("categories"),
                },
            )
        )

    for _, row in review_df.iterrows():
        text_parts = []

        for field in REVIEW_TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        if not text_parts:
            continue

        documents.append(
            Document(
                page_content="\n\n".join(text_parts),
                metadata={
                    "source": "review",
                    "asin": row.get("asin"),
                    "rating": row.get("rating"),
                    "verified_purchase": row.get("verified_purchase"),
                },
            )
        )

    return documents


def main():
    """Build and save BM25 and FAISS artifacts."""
    n = 10000

    meta_path = "../data/raw/meta_Appliances.jsonl.gz"
    review_path = "../data/raw/Appliances.jsonl.gz"

    bm25_dir = "../bm25_index"
    semantic_dir = "../semantic_index"

    meta_records = load_jsonl_gz(meta_path, n=n)
    review_records = load_jsonl_gz(review_path, n=n)

    documents = build_documents(meta_records, review_records)

    print(f"Loaded metadata rows: {len(meta_records)}")
    print(f"Loaded review rows: {len(review_records)}")
    print(f"Built documents: {len(documents)}")

    bm25, tokenized_corpus = build_bm25(documents)
    save_bm25(bm25, tokenized_corpus, save_dir=bm25_dir)
    print(f"Saved BM25 artifacts to: {bm25_dir}")

    embeddings, _ = build_embeddings(documents)
    faiss_index = build_faiss_index(embeddings)
    save_faiss(faiss_index, documents, save_dir=semantic_dir)
    print(f"Saved semantic artifacts to: {semantic_dir}")


if __name__ == "__main__":
    main()

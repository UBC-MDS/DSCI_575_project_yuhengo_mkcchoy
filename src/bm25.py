"""
BM25 keyword-based retrieval module (hybrid: functions + optional CLI)
"""

import os
import pickle
import string
import re
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words("english"))

def tokenize(text, remove_stopwords=True):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords_set]

    return tokens


# Build BM25 index
def build_bm25(documents):
    corpus = [doc.page_content for doc in documents]
    tokenized = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


# Save BM25 index + corpus
def save_bm25(bm25, tokenized_corpus, save_dir="../bm25_index/"):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    with open(os.path.join(save_dir, "tokenized_corpus.pkl"), "wb") as f:
        pickle.dump(tokenized_corpus, f)


# Load BM25 index + corpus
def load_bm25(save_dir="../bm25_index/"):
    with open(os.path.join(save_dir, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    with open(os.path.join(save_dir, "tokenized_corpus.pkl"), "rb") as f:
        tokenized = pickle.load(f)

    return bm25, tokenized


# Search
def bm25_search(query, bm25, documents, k=5):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_k = scores.argsort()[-k:][::-1]

    results = []
    for idx in top_k:
        results.append({
            "score": float(scores[idx]),
            "text": documents[idx].page_content,
            "metadata": documents[idx].metadata
        })
    return results


if __name__ == "__main__":
    print("This module provides BM25 retrieval functions. Import it or call functions directly.")

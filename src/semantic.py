"""
Semantic retrieval using sentence-transformers + FAISS
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Build embeddings
def build_embeddings(documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    corpus = [doc.page_content for doc in documents]
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    return embeddings, model


# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# Save FAISS index + documents
def save_faiss(index, documents, save_dir="../semantic_index/"):
    os.makedirs(save_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(save_dir, "faiss_index.bin"))

    with open(os.path.join(save_dir, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)


# Load FAISS index + documents
def load_faiss(save_dir="../semantic_index/"):
    index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))

    with open(os.path.join(save_dir, "documents.pkl"), "rb") as f:
        documents = pickle.load(f)

    return index, documents


def semantic_search(query, index, model, documents, k=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(dist),
            "text": documents[idx].page_content,
            "metadata": documents[idx].metadata
        })
    return results


if __name__ == "__main__":
    print("This module provides semantic retrieval functions. Import it or call functions directly.")

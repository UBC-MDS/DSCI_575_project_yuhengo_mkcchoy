"""
Retrieval evaluation metrics
"""

def precision_at_k(relevant_indices, retrieved_indices, k):
    retrieved_k = retrieved_indices[:k]
    hits = len(set(retrieved_k) & set(relevant_indices))
    return hits / k


def recall_at_k(relevant_indices, retrieved_indices, k):
    retrieved_k = retrieved_indices[:k]
    hits = len(set(retrieved_k) & set(relevant_indices))
    return hits / len(relevant_indices) if relevant_indices else 0.0


def hit_rate(relevant_indices, retrieved_indices):
    return 1.0 if set(relevant_indices) & set(retrieved_indices) else 0.0


if __name__ == "__main__":
    print("This module provides retrieval metrics.")

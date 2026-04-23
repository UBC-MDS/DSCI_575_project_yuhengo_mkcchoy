"""
Retrieval evaluation metrics
"""


def precision_at_k(relevant_indices, retrieved_indices, k):
    """Calculate the prevision of the retrieval results."""
    retrieved_k = retrieved_indices[:k]
    hits = len(set(retrieved_k) & set(relevant_indices))
    return hits / k


def recall_at_k(relevant_indices, retrieved_indices, k):
    """Calculate the recall of the retrieval results."""
    retrieved_k = retrieved_indices[:k]
    hits = len(set(retrieved_k) & set(relevant_indices))
    return hits / len(relevant_indices) if relevant_indices else 0.0


def hit_rate(relevant_indices, retrieved_indices):
    """Returns 1 if any relevant item is retrieved, otherwise 0."""
    return 1.0 if set(relevant_indices) & set(retrieved_indices) else 0.0

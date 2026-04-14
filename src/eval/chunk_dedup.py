"""Chunk deduplication for RAG evaluation — fixes overlapping chunk penalty."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def deduplicate_chunks(chunks: list[str], threshold: float = 0.85,
                       method: str = "agglomerative") -> list[str]:
    if len(chunks) <= 1:
        return chunks

    model = get_model()
    embeddings = model.encode(chunks)

    if method == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)
    else:
        # Simple threshold-based: merge if similarity > threshold
        labels = list(range(len(chunks)))
        sim_matrix = np.inner(embeddings, embeddings)
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i][j] > threshold:
                    labels[j] = labels[i]

    # Keep the longest chunk per cluster (most context)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    deduped = []
    for cluster_indices in clusters.values():
        cluster_chunks = [(chunks[i], len(chunks[i])) for i in cluster_indices]
        best = max(cluster_chunks, key=lambda x: x[1])
        deduped.append(best[0])

    return deduped

def deduplicate_with_stats(chunks: list[str], threshold: float = 0.85) -> dict:
    original_count = len(chunks)
    deduped = deduplicate_chunks(chunks, threshold)
    return {
        "deduped_chunks": deduped,
        "original_count": original_count,
        "deduped_count": len(deduped),
        "removed": original_count - len(deduped),
        "reduction_pct": (1 - len(deduped) / original_count) * 100 if original_count > 0 else 0
    }

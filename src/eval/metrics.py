"""Custom section-aware precision metric for RAG evaluation."""

from dataclasses import dataclass

@dataclass
class SectionPrecisionResult:
    score: float
    total_groups: int
    relevant_groups: int
    raw_chunks: int
    deduped_chunks: int
    group_details: list[dict]

def section_aware_precision(
    retrieval_context: list[str],
    expected_answer: str,
    group_by_fn=None,
    chunk_metadata: list[dict] = None,
    relevance_fn=None
) -> SectionPrecisionResult:
    """
    Compute precision over grouped retrieval units instead of raw chunks.
    
    If group_by_fn is provided, chunks are grouped before scoring.
    A group is relevant if ANY chunk in it supports the answer.
    
    This fixes the overlapping chunk penalty: 3 overlapping windows 
    around the same revenue table count as 1 relevant group, not 
    "1 hit + 2 errors".
    """
    if group_by_fn and chunk_metadata:
        # Group chunks by the provided function
        groups = {}
        for i, (chunk, meta) in enumerate(zip(retrieval_context, chunk_metadata)):
            key = group_by_fn(meta)
            if key not in groups:
                groups[key] = []
            groups[key].append({"index": i, "content": chunk, "metadata": meta})
    else:
        # Each chunk is its own group (default behavior)
        groups = {i: [{"index": i, "content": c, "metadata": {}}]
                  for i, c in enumerate(retrieval_context)}

    # Score each group
    if relevance_fn is None:
        # Simple keyword overlap relevance
        answer_words = set(expected_answer.lower().split())
        def relevance_fn(chunk_text):
            chunk_words = set(chunk_text.lower().split())
            overlap = len(answer_words & chunk_words)
            return overlap / max(len(answer_words), 1) > 0.15

    group_details = []
    relevant_count = 0
    for key, chunks in groups.items():
        # A group is relevant if ANY chunk in it is relevant
        group_relevant = any(relevance_fn(c["content"]) for c in chunks)
        if group_relevant:
            relevant_count += 1
        group_details.append({
            "group_key": str(key),
            "chunk_count": len(chunks),
            "relevant": group_relevant
        })

    total_groups = len(groups)
    score = relevant_count / total_groups if total_groups > 0 else 0.0

    return SectionPrecisionResult(
        score=score,
        total_groups=total_groups,
        relevant_groups=relevant_count,
        raw_chunks=len(retrieval_context),
        deduped_chunks=total_groups,
        group_details=group_details
    )

def default_group_by(metadata: dict) -> str:
    """Group by doc_id + section_id — the natural retrieval unit for financial docs."""
    doc_id = metadata.get("doc_id", "unknown")
    section_id = metadata.get("section_id", "unknown")
    return f"{doc_id}:{section_id}"

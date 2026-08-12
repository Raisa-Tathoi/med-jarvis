"""Retrieval metrics.

Each golden question is written from a known source chunk, so that chunk is the
judged-relevant document. With one relevant document per query, recall@k and
hit-rate@k are the same number -- reported as hit@k.

A question may have MORE than one valid gold chunk. With chunk_overlap > 0 the
same source text lives in several chunks, and after remapping a golden set onto a
re-chunked corpus one old chunk can map to several new ones. Retrieving any of
them is a hit; scoring against a single id would penalise overlap for working as
intended. Rank is the best (lowest) rank among the valid chunks.

These are still lower bounds: a near-duplicate elsewhere in the corpus that
answers the question just as well counts as a miss. checks.py reports
near-duplicate density so you can tell how much slack that leaves.
"""

import math
from collections.abc import Iterable


def rank_of_gold(
    retrieved_ids: list[str], gold: str | Iterable[str]
) -> int | None:
    """1-indexed rank of the best-ranked gold chunk, or None if none appeared."""
    gold_ids = {gold} if isinstance(gold, str) else set(gold)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """With one relevant document, ideal DCG is 1, so nDCG collapses to 1/log2(rank+1)."""
    if not ranks:
        return 0.0
    return sum(
        1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= k
    ) / len(ranks)


def rank_summary(ranks: list[int | None], max_k: int) -> dict:
    found = sorted(r for r in ranks if r is not None)
    return {
        "questions": len(ranks),
        "never_retrieved": len(ranks) - len(found),
        "never_retrieved_pct": (
            (len(ranks) - len(found)) / len(ranks) * 100 if ranks else 0.0
        ),
        "median_rank_when_found": (
            found[len(found) // 2] if found else None
        ),
        "max_k_searched": max_k,
    }


def score(ranks: list[int | None], ks: tuple[int, ...]) -> dict:
    max_k = max(ks)
    return {
        "per_k": {
            str(k): {
                "hit@k": hit_at_k(ranks, k),
                "mrr@k": mrr_at_k(ranks, k),
                "ndcg@k": ndcg_at_k(ranks, k),
            }
            for k in ks
        },
        "summary": rank_summary(ranks, max_k),
    }


def distance_summary(distances: list[float]) -> dict:
    """Cosine/L2 distance of the top hit. Wide spread means weak separation."""
    if not distances:
        return {}
    ordered = sorted(distances)
    mean = sum(ordered) / len(ordered)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": mean,
        "max": ordered[-1],
    }

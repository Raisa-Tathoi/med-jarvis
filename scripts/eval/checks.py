"""Corpus health checks -- what's actually sitting in the index.

Retrieval metrics tell you how often the right chunk comes back. These tell you
whether the chunks are worth retrieving in the first place. Front matter, tables
of contents, and page-header debris get embedded like anything else and then
compete with real content for the top-k slots.

Note on thresholds: the RecursiveTokenChunker produces uniform chunks (64-143
words on the current corpus), so length alone separates nothing. The signals that
do discriminate here are boilerplate vocabulary, character-class density, and
embedding-space near-duplication.
"""

import re
from collections import Counter

from .config import MAX_DOT_RUN_RATIO, MIN_ALPHA_RATIO, MIN_CHUNK_WORDS

_DOT_RUN_RE = re.compile(r"\.{3,}")

# One of these alone means the chunk is front/back matter, not content.
STRONG_BOILERPLATE = (
    "all rights reserved",
    "isbn",
    "table of contents",
    "library of congress",
    "no part of this publication",
    "printed in the united states",
)

# Two or more of these together mean the same thing.
WEAK_BOILERPLATE = (
    "copyright",
    "e-learning",
    "workbook",
    "new edition",
    "acknowledgment",
    "acknowledgement",
    "preface",
    "www.",
    "http",
    "@gmail",
    "publisher",
    "disclaimer",
)

NEAR_DUPLICATE_THRESHOLD = 0.95


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() or c.isspace() for c in text) / len(text)


def dot_run_ratio(text: str) -> float:
    """Leader dots -- the signature of a table of contents."""
    if not text:
        return 0.0
    return sum(len(m.group()) for m in _DOT_RUN_RE.finditer(text)) / len(text)


def boilerplate_score(text: str) -> int:
    lowered = text.lower()
    if any(marker in lowered for marker in STRONG_BOILERPLATE):
        return 2
    return sum(1 for marker in WEAK_BOILERPLATE if marker in lowered)


def junk_reason(text: str) -> str | None:
    if len(text.split()) < MIN_CHUNK_WORDS:
        return "too_short"
    if dot_run_ratio(text) > MAX_DOT_RUN_RATIO:
        return "toc_leader_dots"
    if alpha_ratio(text) < MIN_ALPHA_RATIO:
        return "low_alpha_ratio"
    if boilerplate_score(text) >= 2:
        return "boilerplate"
    return None


def is_prose(text: str) -> bool:
    return junk_reason(text) is None


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def near_duplicate_pairs(threshold: float = NEAR_DUPLICATE_THRESHOLD) -> dict:
    """Cosine similarity over the stored embeddings.

    Near-duplicates matter twice over: they crowd out distinct content in the
    top-k, and they make hit@k pessimistic, since a duplicate of the gold chunk
    is scored as a miss.
    """
    import numpy as np

    from .pipeline import collection

    stored = collection.get(include=["embeddings"])
    vectors = np.asarray(stored["embeddings"], dtype=np.float32)
    if vectors.size == 0:
        return {"pairs": 0, "chunks_involved": 0, "threshold": threshold, "examples": []}

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    similarity = (vectors / norms) @ (vectors / norms).T
    np.fill_diagonal(similarity, 0.0)

    rows, cols = np.where(np.triu(similarity) >= threshold)
    ids = stored["ids"]
    examples = [
        {
            "a": ids[r],
            "b": ids[c],
            "similarity": round(float(similarity[r, c]), 4),
        }
        for r, c in list(zip(rows, cols))[:5]
    ]
    return {
        "pairs": int(len(rows)),
        "chunks_involved": int(len({*rows.tolist(), *cols.tolist()})),
        "threshold": threshold,
        "examples": examples,
    }


def orphan_collections() -> list[dict]:
    """Collections in the database that retrieve.py never queries.

    Indexed documents that no query can reach are invisible in every retrieval
    metric -- the questions they would answer simply come back wrong.
    """
    from .pipeline import chroma_client, collection

    return [
        {"name": c.name, "chunks": c.count()}
        for c in chroma_client.list_collections()
        if c.name != collection.name
    ]


def run(chunks: list[dict], include_near_duplicates: bool = True) -> dict:
    """chunks: [{"id": ..., "document": ...}]"""
    word_counts = sorted(len(c["document"].split()) for c in chunks)
    reasons = Counter()
    junk_examples: list[dict] = []

    for chunk in chunks:
        reason = junk_reason(chunk["document"])
        if reason is None:
            continue
        reasons[reason] += 1
        if len(junk_examples) < 5:
            junk_examples.append(
                {
                    "id": chunk["id"],
                    "reason": reason,
                    "preview": " ".join(chunk["document"].split())[:120],
                }
            )

    exact = Counter(normalized(c["document"]) for c in chunks)
    duplicate_groups = {text: n for text, n in exact.items() if n > 1}
    empty = sum(1 for c in chunks if not c["document"].strip())

    total = len(chunks) or 1
    report = {
        "chunks": len(chunks),
        "empty_chunks": empty,
        "junk_chunks": sum(reasons.values()),
        "junk_pct": sum(reasons.values()) / total * 100,
        "junk_by_reason": dict(reasons),
        "junk_examples": junk_examples,
        "exact_duplicate_chunks": sum(duplicate_groups.values()) - len(duplicate_groups),
        "word_count": {
            "min": word_counts[0] if word_counts else 0,
            "p25": word_counts[len(word_counts) // 4] if word_counts else 0,
            "median": word_counts[len(word_counts) // 2] if word_counts else 0,
            "p90": word_counts[int(len(word_counts) * 0.9)] if word_counts else 0,
            "max": word_counts[-1] if word_counts else 0,
        },
    }
    report["orphan_collections"] = orphan_collections()
    if include_near_duplicates:
        report["near_duplicates"] = near_duplicate_pairs()
    return report

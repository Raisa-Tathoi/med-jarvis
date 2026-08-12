"""Verify the harness itself before trusting its numbers.

Imports nothing heavy -- no Chroma, no models -- so it runs in about a second:

    venv/bin/python -m scripts.eval.selftest
"""

# Lets this run either way: `python -m scripts.eval.selftest` or
# `python scripts/eval/selftest.py` (the IDE Run button uses the second).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "scripts.eval"

import math

from . import answer_checks as ac
from . import checks, metrics, remap


def test_ranking() -> None:
    assert metrics.rank_of_gold(["a", "b", "c"], "c") == 3
    assert metrics.rank_of_gold(["a", "b"], "z") is None
    assert metrics.rank_of_gold([], "z") is None


def test_multi_gold_ranking() -> None:
    # Any valid gold counts, and the best rank wins -- this is what stops
    # chunk_overlap from looking harmful when it is working.
    assert metrics.rank_of_gold(["a", "b", "c"], ["c", "b"]) == 2
    assert metrics.rank_of_gold(["a", "b"], ["x", "y"]) is None
    assert metrics.rank_of_gold(["a"], ["a"]) == 1


def test_remap_matching() -> None:
    chunks = [
        ("chunk_0", "the patient presents with fever and a new murmur"),
        ("chunk_1", "fever and a new murmur suggest acute rheumatic fever"),
        ("chunk_2", "unrelated content about renal calculi"),
    ]
    # Verbatim containment in two overlapping chunks -> both are valid golds.
    hit = remap.match("fever and a new murmur", chunks, 0.5)
    assert hit["method"] == "containment", hit
    assert sorted(hit["ids"]) == ["chunk_0", "chunk_1"], hit

    # Nothing resembling the text at all.
    miss = remap.match("torque specification for head bolts", chunks, 0.5)
    assert miss["ids"] == [], miss
    assert miss["confidence"] < 0.5

    assert remap.match("", chunks, 0.5)["method"] == "no_source_text"


def test_hit_and_mrr() -> None:
    ranks = [1, 2, None, 3, 5]
    assert metrics.hit_at_k(ranks, 1) == 0.2
    assert metrics.hit_at_k(ranks, 3) == 0.6
    assert metrics.hit_at_k(ranks, 5) == 0.8
    assert math.isclose(metrics.mrr_at_k(ranks, 5), (1 + 0.5 + 1 / 3 + 0.2) / 5)
    # A rank beyond k must not be credited.
    assert math.isclose(metrics.mrr_at_k(ranks, 2), (1 + 0.5) / 5)
    assert metrics.hit_at_k([], 3) == 0.0


def test_ndcg() -> None:
    assert math.isclose(metrics.ndcg_at_k([1], 1), 1.0)
    assert math.isclose(metrics.ndcg_at_k([3], 3), 1 / math.log2(4))
    assert metrics.ndcg_at_k([None], 10) == 0.0


def test_summary() -> None:
    summary = metrics.rank_summary([1, None, 4], 10)
    assert summary["questions"] == 3
    assert summary["never_retrieved"] == 1
    assert summary["max_k_searched"] == 10


def test_answer_scoring() -> None:
    assert ac.keyword_recall("penicillin therapy", "penicillin") == 1.0
    assert ac.keyword_recall("aspirin", "penicillin") == 0.0
    assert ac.grounding("penicillin", "the drug penicillin is used") == 1.0
    assert ac.grounding("aspirin", "penicillin here") == 0.0
    # Stopwords must not manufacture overlap.
    assert ac.grounding("the and for", "penicillin") == 0.0


def test_abstention() -> None:
    assert ac.abstained("I don't know based on the context.")
    assert ac.abstained("The context does not mention this.")
    assert not ac.abstained("Treat with benzathine penicillin G.")


def test_junk_detection() -> None:
    assert checks.junk_reason("word " * 100) is None
    assert checks.junk_reason("too short") == "too_short"
    assert (
        checks.junk_reason("Chapter One " + "." * 400 + " word " * 80)
        == "toc_leader_dots"
    )
    assert (
        checks.junk_reason("All rights reserved by the publisher. " + "word " * 80)
        == "boilerplate"
    )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc or 'assertion failed'}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

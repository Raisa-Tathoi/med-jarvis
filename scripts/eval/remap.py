"""Re-point a golden set at a re-chunked corpus.

Chunk ids are positional. Change chunk_size or chunk_overlap and re-ingest, and
chunk_37 is different text -- every gold_chunk_id in the golden set now points at
the wrong content. hit@k still prints a number; the number is noise.

This rewrites those ids by matching on the stored source text instead of the id.
Each golden keeps a snapshot of the chunk it was written from (gold_chunk_text,
or gold_chunk_preview on older records); we find which chunks in the CURRENT
collection contain that text and rewrite the ids to match.

Matching runs in two tiers:

  containment  the stored text appears verbatim inside a current chunk. Exact,
               and the common case.
  overlap      no chunk contains it whole -- the snapshot straddles a new
               boundary. Falls back to word-trigram overlap and takes every
               chunk scoring above --min-confidence.

Several matches is the expected outcome with chunk_overlap > 0, not an error:
the same text genuinely lives in more than one chunk, and retrieving any of them
is a hit. All of them are written to gold_chunk_ids.

Run from the repo root. Dry run by default:

    venv/bin/python -m scripts.eval.remap
    venv/bin/python -m scripts.eval.remap --apply

--apply writes goldens.jsonl.bak first.
"""

# Lets this run either way: `python -m scripts.eval.remap` or
# `python scripts/eval/remap.py` (the IDE Run button uses the second).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "scripts.eval"

import argparse
import json
import shutil
from pathlib import Path

from .config import GOLDENS_PATH

TRIGRAM_SIZE = 3


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def trigrams(text: str) -> set[tuple[str, ...]]:
    words = text.split()
    return {
        tuple(words[i : i + TRIGRAM_SIZE])
        for i in range(len(words) - TRIGRAM_SIZE + 1)
    }


def source_text(record: dict) -> str:
    return record.get("gold_chunk_text") or record.get("gold_chunk_preview", "")


def match(needle: str, chunks: list[tuple[str, str]], min_confidence: float) -> dict:
    """chunks: [(chunk_id, normalized_text)]. Returns ids, method, confidence."""
    if not needle:
        return {"ids": [], "method": "no_source_text", "confidence": 0.0}

    contained = [chunk_id for chunk_id, text in chunks if needle in text]
    if contained:
        return {"ids": contained, "method": "containment", "confidence": 1.0}

    needle_grams = trigrams(needle)
    if not needle_grams:
        return {"ids": [], "method": "too_short_to_match", "confidence": 0.0}

    scored = []
    for chunk_id, text in chunks:
        shared = len(needle_grams & trigrams(text))
        if shared:
            scored.append((shared / len(needle_grams), chunk_id))
    if not scored:
        return {"ids": [], "method": "no_match", "confidence": 0.0}

    scored.sort(reverse=True)
    best = scored[0][0]
    if best < min_confidence:
        return {"ids": [], "method": "below_threshold", "confidence": best}

    # Keep every chunk scoring near the best -- a straddled snapshot is split
    # across adjacent chunks and both halves are legitimate targets.
    keep = [chunk_id for score, chunk_id in scored if score >= min(best, min_confidence)]
    return {"ids": keep, "method": "overlap", "confidence": best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(GOLDENS_PATH))
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="remove goldens with no match instead of leaving them untouched",
    )
    args = parser.parse_args()

    # Imported here, not at module scope: this pulls in Chroma and MiniLM, and
    # selftest.py exercises match() without wanting either.
    from . import pipeline

    path = Path(args.dataset)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        print(f"No goldens at {path}")
        return 1

    chunks = [(c["id"], normalize(c["document"])) for c in pipeline.all_chunks()]
    print(f"{len(records)} goldens against {len(chunks)} current chunks\n")

    updated: list[dict] = []
    tally = {"unchanged": 0, "remapped": 0, "expanded": 0, "unmatched": 0}
    unmatched_examples: list[str] = []

    for record in records:
        old_ids = record.get("gold_chunk_ids") or [record["gold_chunk_id"]]
        result = match(normalize(source_text(record)), chunks, args.min_confidence)
        new_ids = result["ids"]

        if not new_ids:
            tally["unmatched"] += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append(
                    f"    [{result['method']}, best={result['confidence']:.2f}] "
                    f"{record['question'][:66]}"
                )
            if not args.drop_unmatched:
                updated.append(record)
            continue

        if set(new_ids) == set(old_ids):
            tally["unchanged"] += 1
        elif len(new_ids) > len(old_ids):
            tally["expanded"] += 1
        else:
            tally["remapped"] += 1

        record["gold_chunk_ids"] = sorted(new_ids)
        record["gold_chunk_id"] = sorted(new_ids)[0]  # kept for older readers
        record["remap"] = {
            "method": result["method"],
            "confidence": round(result["confidence"], 3),
            "previous_ids": sorted(old_ids),
        }
        updated.append(record)

    print(f"  unchanged  {tally['unchanged']:>4}  already pointing at the right text")
    print(f"  remapped   {tally['remapped']:>4}  id changed")
    print(f"  expanded   {tally['expanded']:>4}  now matches several chunks (overlap)")
    print(f"  unmatched  {tally['unmatched']:>4}  source text not found in the corpus")
    for example in unmatched_examples:
        print(example)

    if tally["unmatched"] and not args.drop_unmatched:
        print(
            "\n  Unmatched goldens were left untouched and will score as misses.\n"
            "  Re-run with --drop-unmatched to remove them, or lower --min-confidence."
        )

    if not args.apply:
        print(f"\nDRY RUN -- {path} not modified. Re-run with --apply.")
        return 0

    shutil.copyfile(path, path.with_suffix(".jsonl.bak"))
    with path.open("w") as handle:
        for record in updated:
            handle.write(json.dumps(record) + "\n")
    print(f"\nWrote {len(updated)} goldens to {path} (backup: {path.name}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

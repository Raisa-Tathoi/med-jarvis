"""Run the evaluation and print a scorecard.

  venv/bin/python -m scripts.eval.run                      # retrieval only, fast
  venv/bin/python -m scripts.eval.run --with-generation    # also runs Qwen, slow
  venv/bin/python -m scripts.eval.run --compare scripts/eval/results/<earlier>.json

Every run writes a JSON file to scripts/eval/results/ so two pipeline versions
can be diffed against each other with --compare.
"""

# Lets this run either way: `python -m scripts.eval.run` or `python scripts/eval/run.py`
# (the IDE Run button uses the second, which otherwise breaks the relative imports).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "scripts.eval"

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import answer_checks, checks, metrics, pipeline
from .config import (
    DEFAULT_K,
    DEFAULT_KS,
    GOLDENS_PATH,
    NEGATIVES_PATH,
    RESULTS_DIR,
)


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_retrieval(goldens: list[dict], ks: tuple[int, ...]) -> dict:
    max_k = max(ks)
    ranks: list[int | None] = []
    top_distances: list[float] = []
    misses: list[dict] = []
    latencies: list[float] = []

    for item in goldens:
        started = time.perf_counter()
        hits = pipeline.retrieve_detailed(item["question"], max_k)
        latencies.append(time.perf_counter() - started)

        gold_ids = item.get("gold_chunk_ids") or [item["gold_chunk_id"]]
        rank = metrics.rank_of_gold([h["id"] for h in hits], gold_ids)
        ranks.append(rank)
        if hits:
            top_distances.append(hits[0]["distance"])
        if rank is None or rank > DEFAULT_K:
            misses.append(
                {
                    "question": item["question"],
                    "gold_chunk_id": "/".join(gold_ids),
                    "rank": rank,
                    "top_hit_id": hits[0]["id"] if hits else None,
                    "top_hit_preview": (
                        " ".join(hits[0]["document"].split())[:140] if hits else None
                    ),
                }
            )

    latencies.sort()
    return {
        **metrics.score(ranks, ks),
        "top_hit_distance": metrics.distance_summary(top_distances),
        "latency_s": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "misses": misses,
    }


def evaluate_generation(goldens: list[dict], negatives: list[dict], k: int) -> dict:
    pipeline.load_generator()  # pay the load cost once, before timing anything

    scored = []
    for index, item in enumerate(goldens, start=1):
        context = "\n---\n".join(pipeline.retrieve_documents(item["question"], k))
        started = time.perf_counter()
        text = pipeline.answer(item["question"], n_results=k)
        elapsed = time.perf_counter() - started
        result = answer_checks.score_answer(text, item["expected_answer"], context)
        result.update({"id": item["id"], "seconds": elapsed, "answer": text})
        scored.append(result)
        print(
            f"  [{index}/{len(goldens)}] recall={result['keyword_recall']:.2f} "
            f"grounding={result['grounding']:.2f} {elapsed:.1f}s",
            flush=True,
        )

    abstentions = []
    for index, item in enumerate(negatives, start=1):
        text = pipeline.answer(item["question"], n_results=k)
        abstentions.append({"id": item["id"], "abstained": answer_checks.abstained(text)})
        print(
            f"  [neg {index}/{len(negatives)}] abstained={abstentions[-1]['abstained']}",
            flush=True,
        )

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "answered": len(scored),
        "keyword_recall_mean": mean([s["keyword_recall"] for s in scored]),
        "grounding_mean": mean([s["grounding"] for s in scored]),
        "false_abstention_rate": mean([1.0 if s["abstained"] else 0.0 for s in scored]),
        "empty_answer_rate": mean([1.0 if s["empty"] else 0.0 for s in scored]),
        "seconds_per_answer_mean": mean([s["seconds"] for s in scored]),
        "negatives": len(abstentions),
        "correct_abstention_rate": mean(
            [1.0 if a["abstained"] else 0.0 for a in abstentions]
        ),
        "worst_grounded": sorted(
            (
                {
                    "id": s["id"],
                    "grounding": s["grounding"],
                    "answer": s["answer"][:200],
                }
                for s in scored
            ),
            key=lambda s: s["grounding"],
        )[:5],
    }


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def print_report(report: dict, ks: tuple[int, ...]) -> None:
    line = "=" * 62
    stats = report["collection"]
    print(f"\n{line}\nRAG EVALUATION  {report['timestamp']}\n{line}")
    print(
        f"collection={stats['collection']}  chunks={stats['chunk_count']}  "
        f"embeddings={stats['embedding_model']}"
    )

    health = report.get("corpus_health")
    if health:
        print(f"\nCORPUS HEALTH\n{'-' * 62}")
        print(
            f"  unusable chunks     {health['junk_chunks']:>5} "
            f"({health['junk_pct']:.1f}%)  {health['junk_by_reason']}"
        )
        print(f"  exact duplicates    {health['exact_duplicate_chunks']:>5}")
        near = health.get("near_duplicates")
        if near:
            print(
                f"  near-duplicates     {near['pairs']:>5} pairs "
                f"(cosine >= {near['threshold']}), {near['chunks_involved']} chunks"
            )
        wc = health["word_count"]
        print(
            f"  words/chunk         min {wc['min']}  p25 {wc['p25']}  "
            f"median {wc['median']}  p90 {wc['p90']}  max {wc['max']}"
        )
        for example in health["junk_examples"][:3]:
            print(f"    {example['id']} [{example['reason']}] {example['preview'][:70]}")
        for orphan in health.get("orphan_collections", []):
            print(
                f"  ORPHAN COLLECTION   '{orphan['name']}' holds {orphan['chunks']} "
                f"chunks that retrieve.py never queries"
            )

    retrieval = report["retrieval"]
    summary = retrieval["summary"]
    print(f"\nRETRIEVAL  ({summary['questions']} questions)\n{'-' * 62}")
    print(f"  {'k':>4}  {'hit@k':>7}  {'mrr@k':>7}  {'ndcg@k':>7}")
    for k in ks:
        row = retrieval["per_k"][str(k)]
        print(
            f"  {k:>4}  {pct(row['hit@k']):>7}  {row['mrr@k']:>7.3f}  "
            f"{row['ndcg@k']:>7.3f}"
        )
    print(
        f"  gold chunk never in top-{summary['max_k_searched']}: "
        f"{summary['never_retrieved']} ({summary['never_retrieved_pct']:.1f}%)"
    )
    distance = retrieval["top_hit_distance"]
    if distance:
        print(
            f"  top-hit distance    min {distance['min']:.3f}  "
            f"median {distance['median']:.3f}  max {distance['max']:.3f}"
        )
    latency = retrieval["latency_s"]
    if latency["median"] is not None:
        print(
            f"  query latency       median {latency['median'] * 1000:.0f}ms  "
            f"max {latency['max'] * 1000:.0f}ms"
        )

    if retrieval["misses"]:
        print(f"\n  worst misses (gold chunk outside top-{DEFAULT_K}):")
        for miss in retrieval["misses"][:5]:
            rank = miss["rank"] if miss["rank"] is not None else "not found"
            print(f"    rank={rank}  {miss['question'][:64]}")
            print(f"      wanted {miss['gold_chunk_id']}, got {miss['top_hit_id']}")

    generation = report.get("generation")
    if generation:
        print(f"\nGENERATION  ({generation['answered']} answers)\n{'-' * 62}")
        print(f"  keyword recall vs expected   {pct(generation['keyword_recall_mean'])}")
        print(f"  grounding in context         {pct(generation['grounding_mean'])}")
        print(f"  false abstention (in-corpus) {pct(generation['false_abstention_rate'])}")
        print(f"  empty answers                {pct(generation['empty_answer_rate'])}")
        print(
            f"  correct abstention on {generation['negatives']} out-of-corpus  "
            f"{pct(generation['correct_abstention_rate'])}"
        )
        print(f"  seconds per answer           {generation['seconds_per_answer_mean']:.1f}s")
        print("\n  least-grounded answers (possible hallucination):")
        for item in generation["worst_grounded"][:3]:
            print(f"    {item['grounding']:.2f}  {item['answer'][:70]}")
    print(line)


def print_comparison(current: dict, baseline: dict, ks: tuple[int, ...]) -> None:
    print(f"\nCOMPARED TO {baseline['timestamp']}\n{'-' * 62}")
    print(f"  {'k':>4}  {'hit@k now':>10}  {'was':>8}  {'delta':>8}")
    for k in ks:
        key = str(k)
        if key not in baseline["retrieval"]["per_k"]:
            continue
        now = current["retrieval"]["per_k"][key]["hit@k"]
        was = baseline["retrieval"]["per_k"][key]["hit@k"]
        arrow = "+" if now > was else ""
        print(f"  {k:>4}  {pct(now):>10}  {pct(was):>8}  {arrow}{(now - was) * 100:>7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k used for generation")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--dataset", default=str(GOLDENS_PATH))
    parser.add_argument("--limit", type=int, help="only evaluate the first N questions")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--compare", help="path to an earlier results JSON")
    args = parser.parse_args()

    goldens = load_jsonl(Path(args.dataset))
    if not goldens:
        print(
            f"No golden questions at {args.dataset}.\n"
            "Build some first:  venv/bin/python -m scripts.eval.build_dataset --n 40"
        )
        return 1
    if args.limit:
        goldens = goldens[: args.limit]

    ks = tuple(sorted(set(args.ks)))
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": pipeline.collection_stats(),
        "dataset": {"path": args.dataset, "questions": len(goldens)},
        "ks": list(ks),
    }

    if not args.skip_health:
        report["corpus_health"] = checks.run(pipeline.all_chunks())

    print(f"Scoring retrieval over {len(goldens)} questions...")
    report["retrieval"] = evaluate_retrieval(goldens, ks)

    if args.with_generation:
        negatives = load_jsonl(NEGATIVES_PATH)
        if args.limit:
            negatives = negatives[: args.limit]
        print(
            f"\nRunning generation over {len(goldens)} questions "
            f"+ {len(negatives)} negatives (this is slow)..."
        )
        report["generation"] = evaluate_generation(goldens, negatives, args.k)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{report['timestamp'].replace(':', '-')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print_report(report, ks)
    if args.compare:
        print_comparison(report, json.loads(Path(args.compare).read_text()), ks)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

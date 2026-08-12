"""Build a golden question/answer set from the chunks already in Chroma.

Sampling a chunk and writing a question answerable only from that chunk gives us
ground truth for free: the sampled chunk is the relevant document, so retrieval
can be scored without anyone hand-labelling anything.

Two backends:

  --backend local   the Qwen model generate.py already loads. No API key, no
                    cost, slower, and the questions are rougher. This is the
                    default because it works offline today.
  --backend claude  claude-opus-5 with structured outputs. Needs ANTHROPIC_API_KEY
                    and `pip install anthropic`. Much better questions.

Output is JSONL at scripts/eval/data/goldens.jsonl, appended and resumable --
rerunning skips chunks already covered.

  venv/bin/python -m scripts.eval.build_dataset --n 40
  venv/bin/python -m scripts.eval.build_dataset --n 100 --backend claude
"""

# Lets this run either way: `python -m scripts.eval.build_dataset` or
# `python scripts/eval/build_dataset.py` (the IDE Run button uses the second).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "scripts.eval"

import argparse
import json
import random
import re
import sys

from . import checks, pipeline
from .config import (
    CLAUDE_MODEL,
    DEFAULT_N_QUESTIONS,
    GOLDENS_PATH,
    RANDOM_SEED,
)

INSTRUCTIONS = """You are building an evaluation set for a USMLE Step 2 study assistant.

Read the passage below and write ONE exam-style question that can be answered using only this passage, plus a short factual answer drawn from it.

Rules:
- The question must be answerable from this passage alone. Do not require outside knowledge.
- Do not refer to "the passage", "the text", or "the context" in the question. Write it the way a medical student would type it into a search box.
- The answer must be one or two sentences, using the passage's own terminology.
- If the passage is boilerplate, a table of contents, or has no medical content worth testing, set "usable" to false and leave question and answer empty.

Passage:
\"\"\"
{chunk}
\"\"\""""

JSON_INSTRUCTION = """

Respond with only a JSON object, no other text:
{"usable": true, "question": "...", "answer": "..."}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["usable", "question", "answer"],
    "additionalProperties": False,
}


def parse_json_ish(text: str) -> dict | None:
    """Qwen wraps JSON in fences and prose often enough to need a fallback."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    question = re.search(r"question[\"']?\s*[:\-]\s*[\"']?(.+)", text, re.IGNORECASE)
    answer = re.search(r"answer[\"']?\s*[:\-]\s*[\"']?(.+)", text, re.IGNORECASE)
    if question and answer:
        return {
            "usable": True,
            "question": question.group(1).strip().strip('",'),
            "answer": answer.group(1).strip().strip('",'),
        }
    return None


def generate_local(chunk_text: str) -> dict | None:
    raw = pipeline.complete(
        INSTRUCTIONS.format(chunk=chunk_text) + JSON_INSTRUCTION, max_new_tokens=300
    )
    return parse_json_ish(raw)


def generate_claude(chunk_text: str, client) -> dict | None:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{"role": "user", "content": INSTRUCTIONS.format(chunk=chunk_text)}],
    )
    if response.stop_reason == "refusal":
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_existing_chunk_ids() -> set[str]:
    if not GOLDENS_PATH.exists():
        return set()
    seen = set()
    with GOLDENS_PATH.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["gold_chunk_id"])
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N_QUESTIONS)
    parser.add_argument("--backend", choices=("local", "claude"), default="local")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    client = None
    if args.backend == "claude":
        try:
            import anthropic
        except ImportError:
            print("--backend claude needs: pip install anthropic", file=sys.stderr)
            return 1
        client = anthropic.Anthropic()

    chunks = pipeline.all_chunks()
    already = load_existing_chunk_ids()
    candidates = [
        c
        for c in chunks
        if checks.is_prose(c["document"]) and c["id"] not in already
    ]
    print(
        f"{len(chunks)} chunks, {len(candidates)} usable candidates "
        f"({len(already)} already in the golden set)"
    )
    if not candidates:
        print("Nothing left to sample.")
        return 0

    random.Random(args.seed).shuffle(candidates)

    GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with GOLDENS_PATH.open("a") as handle:
        for chunk in candidates:
            if written >= args.n:
                break
            text = chunk["document"]
            try:
                record = (
                    generate_local(text)
                    if args.backend == "local"
                    else generate_claude(text, client)
                )
            except Exception as exc:  # one bad chunk shouldn't kill a long run
                print(f"  {chunk['id']}: generation failed ({exc})", file=sys.stderr)
                continue

            if not record or not record.get("usable"):
                continue
            question = (record.get("question") or "").strip()
            answer = (record.get("answer") or "").strip()
            if len(question) < 15 or len(answer) < 5:
                continue

            handle.write(
                json.dumps(
                    {
                        "id": f"gold_{chunk['id']}",
                        "question": question,
                        "expected_answer": answer,
                        "gold_chunk_id": chunk["id"],
                        "gold_chunk_ids": [chunk["id"]],
                        # Full source text, so scripts.eval.remap can re-point this
                        # golden after a re-chunk. Ids are positional and change;
                        # the text is what actually identifies the answer.
                        "gold_chunk_text": " ".join(text.split()),
                        "gold_chunk_preview": " ".join(text.split())[:200],
                        "backend": args.backend,
                    }
                )
                + "\n"
            )
            handle.flush()
            written += 1
            print(f"[{written}/{args.n}] {chunk['id']}: {question[:80]}", flush=True)

    print(f"\nWrote {written} questions to {GOLDENS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

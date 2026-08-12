# RAG evaluation harness

Measures the pipeline in `scripts/` — the same SentenceTransformer instance and
the same Chroma collection that `retrieve.py` and `generate.py` use, so the
numbers describe your actual pipeline rather than a reimplementation of it.

There is no LLM judge. Everything scored here is deterministic: retrieval is
scored against known-correct chunk IDs, and answers are scored with lexical
overlap and abstention checks.

## Run it

Use the `venv/` interpreter — that's the one with `chromadb` and `torch`; `.venv/`
is empty. Either invocation style works, from any directory:

```bash
venv/bin/python -m scripts.eval.run     # module style
venv/bin/python scripts/eval/run.py     # direct path, e.g. the IDE Run button
```

The entry points put the repo root on `sys.path` themselves, and `pipeline.py`
chdirs there on import (`chroma_store.py` opens the database at a relative path).

```bash
# Retrieval metrics + corpus health. ~5 seconds.
venv/bin/python -m scripts.eval.run

# Adds the full pipeline: Qwen generates an answer per question. ~16s each.
venv/bin/python -m scripts.eval.run --with-generation

# Did a chunking or embedding change help?
venv/bin/python -m scripts.eval.run --compare scripts/eval/results/<earlier>.json
```

Useful flags: `--limit N` (first N questions, also caps negatives), `--k N`
(retrieval depth for generation, default 3 to match `generate.py`), `--ks 1 3 5 10`,
`--skip-health`.

Every run writes JSON to `scripts/eval/results/` (gitignored) for `--compare`.

## The golden set

`datasets/goldens.jsonl` is generated from the chunks already in Chroma: sample a
chunk, ask a model to write a question answerable only from it, and record that
chunk as the correct answer. Ground truth comes free — nothing is hand-labelled.

```bash
venv/bin/python -m scripts.eval.build_dataset --n 40                  # local Qwen
venv/bin/python -m scripts.eval.build_dataset --n 100 --backend claude
```

The file is append-only and resumable: rerunning skips chunks already covered, so
you can grow the set over time. Delete lines you think are bad questions; the
harness doesn't care where the entries came from, so hand-written entries in the
same format work fine.

`--backend claude` uses `claude-opus-5` with structured outputs and produces much
sharper questions. It needs `pip install anthropic` and an `ANTHROPIC_API_KEY`
(neither is currently present, which is why `local` is the default).

`datasets/negatives.jsonl` is 12 hand-written out-of-corpus questions (car repair,
football scores, train timetables). The system should refuse all of them.

## After changing chunking

Chunk ids are positional. Change `chunk_size` or `chunk_overlap`, re-ingest, and
`chunk_37` is different text — every `gold_chunk_id` now points at the wrong
content. `hit@k` still prints a number; the number is noise.

Re-point the golden set instead of rebuilding it:

```bash
venv/bin/python -m scripts.eval.remap           # dry run, shows what would change
venv/bin/python -m scripts.eval.remap --apply   # writes goldens.jsonl.bak first
```

It matches on the source text stored in each golden (`gold_chunk_text`, or
`gold_chunk_preview` on older records) rather than on the id, so the questions
survive a re-chunk and results stay comparable across the change.

**A question can have several valid gold chunks.** With `chunk_overlap > 0` the
same source text genuinely lives in more than one chunk, so retrieving any of
them is a hit — scored via `gold_chunk_ids`. Without this, overlap would look
like it *hurt* retrieval when it helped. Verified against a simulated re-chunk at
`chunk_overlap=150`: 0 unmatched, 4 of 43 questions expanded to 2–3 golds.

Unmatched goldens are left in place (and will score as misses) unless you pass
`--drop-unmatched`; lower `--min-confidence` if the fallback matcher is being too
strict. Re-ingesting also needs the old collection dropped first, since Chroma
upserts on repeated ids and stale high-numbered chunks otherwise survive:

```python
import chromadb
chromadb.PersistentClient(path="database/chroma_db").delete_collection("first_aid_usmle")
```

## What the numbers mean

**Retrieval.** Each question has exactly one relevant chunk, so `hit@k` and
`recall@k` are the same number.

| Metric | Reading |
|---|---|
| `hit@k` | Fraction of questions where the source chunk appeared in the top k. `hit@3` is the one that matters — `generate.py` retrieves 3. |
| `mrr@k` | Mean reciprocal rank. Rewards ranking the right chunk *first*, not merely somewhere. |
| `ndcg@k` | Same signal, log-discounted. With one relevant document it reduces to `1/log2(rank+1)`. |
| `never_retrieved` | Questions whose source chunk never surfaced at all. These are the real failures. |
| top-hit distance | L2 distance of the best match. A narrow spread across very different questions means the embeddings aren't separating your content. |

**Generation** (`--with-generation`):

| Metric | Reading |
|---|---|
| `keyword_recall` | Overlap between the answer and the expected answer's content words. |
| `grounding` | Share of the answer's content words that appear in the retrieved context. **Low grounding is the hallucination signal** — the model answered from its own weights instead of your textbook. |
| `false_abstention` | Said "I don't know" when the answer was retrievable. Wasted retrieval. |
| `correct_abstention` | Said "I don't know" to the out-of-corpus negatives. Should be 100%. Anything less means it confabulates on questions your corpus can't answer — the worst failure mode for exam prep. |

**Corpus health** runs independently of the questions: boilerplate/front-matter
detection, exact and embedding-space near-duplicates, chunk length distribution,
and collections in the database that `retrieve.py` never queries.

## Known limits

- **`hit@k` is a lower bound.** If a near-duplicate chunk answers the question
  equally well, retrieving it scores as a miss. The near-duplicate count in the
  health section tells you how much slack that leaves (currently 0 pairs, so very
  little).
- **Locally-generated questions are generic.** A 3B model writes things like
  "What imaging studies help assess pulmonary involvement?", which legitimately
  match many chunks. That depresses `hit@1`/`hit@3` for reasons that aren't
  retrieval's fault. `--backend claude` largely fixes this.
- **Lexical answer scoring can't see paraphrase.** A correct answer in different
  words scores low on `keyword_recall`. Read it as a trend across runs, not as a
  grade on any single answer.

"""Shared paths and defaults for the RAG evaluation harness."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DIR = Path(__file__).resolve().parent
# Named "datasets" rather than "data" because .gitignore has a bare `data/` rule
# that would otherwise swallow the golden set.
DATA_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"

GOLDENS_PATH = DATA_DIR / "goldens.jsonl"
NEGATIVES_PATH = DATA_DIR / "negatives.jsonl"

# k values reported in the retrieval scorecard.
DEFAULT_KS = (1, 3, 5, 10)

# Must mirror the n_results default in generate.py:stream_answer. It drives the
# generation eval and the "worst misses" cutoff, so if the app retrieves 5 and
# this says 3, the harness reports failures the app would never have had.
DEFAULT_K = 5

# Dataset generation
DEFAULT_N_QUESTIONS = 40
RANDOM_SEED = 20260809

# A chunk has to look like prose before we ask a model to write a question about it.
# Filters out the table of contents, copyright page, and figure-caption debris.
MIN_CHUNK_WORDS = 60
MIN_ALPHA_RATIO = 0.65
MAX_DOT_RUN_RATIO = 0.02

CLAUDE_MODEL = "claude-opus-5"

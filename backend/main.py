import json
import sys
from pathlib import Path
from threading import Lock
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

# scripts/ modules import each other flatly (`from retrieve import retrieve`),
# so put that directory on the path rather than importing them as a package.
sys.path.insert(0, str(ROOT / "scripts"))

from generate import stream_answer  # noqa: E402
from retrieve import retrieve  # noqa: E402

app = FastAPI()

# One generation at a time: a single model instance on MPS/CUDA can't serve
# concurrent generate() calls safely.
_generation_lock = Lock()


class AskRequest(BaseModel):
    question: str
    n_results: int = 5


def _event(payload: dict) -> str:
    """One NDJSON line — the frontend reads the stream line by line."""
    return json.dumps(payload) + "\n"


def _stream(question: str, n_results: int) -> Iterator[str]:
    with _generation_lock:
        try:
            chunks = retrieve(question, n_results=n_results)
            yield _event({"type": "sources", "sources": chunks})

            for token in stream_answer(question, n_results=n_results, chunks=chunks):
                yield _event({"type": "token", "text": token})

            yield _event({"type": "done"})
        except Exception as exc:  # surface failures in the UI instead of a dead stream
            yield _event({"type": "error", "message": str(exc)})


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return StreamingResponse(
            iter([_event({"type": "error", "message": "Question is empty."})]),
            media_type="application/x-ndjson",
        )

    return StreamingResponse(
        _stream(question, req.n_results),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

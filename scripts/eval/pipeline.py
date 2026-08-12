"""Adapters onto the real pipeline in scripts/.

Everything here goes through the same objects generate.py and retrieve.py use --
the same SentenceTransformer instance and the same Chroma collection -- so the
harness measures the actual pipeline rather than a lookalike of it.

chroma_store.py opens the database at the relative path "database/chroma_db", so
importing it only works from the repo root. We chdir there on import.
"""

import os
import sys

from .config import REPO_ROOT, SCRIPTS_DIR

os.chdir(REPO_ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chroma_store as _chroma_store  # noqa: E402  (must follow the sys.path/chdir setup)
import retrieve as _retrieve  # noqa: E402

collection = _retrieve.collection
chroma_client = _chroma_store.chroma_client
embed_model = _retrieve.model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve_detailed(question: str, n_results: int) -> list[dict]:
    """Retrieve with the ids and distances that retrieve.retrieve() discards."""
    question_embedding = embed_model.encode(question)
    result = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results,
        include=["documents", "distances"],
    )
    return [
        {"id": chunk_id, "document": document, "distance": distance}
        for chunk_id, document, distance in zip(
            result["ids"][0], result["documents"][0], result["distances"][0]
        )
    ]


def retrieve_documents(question: str, n_results: int) -> list[str]:
    """The exact call generate.py makes."""
    return _retrieve.retrieve(question, n_results=n_results)


def all_chunks() -> list[dict]:
    result = collection.get(include=["documents"])
    return [
        {"id": chunk_id, "document": document}
        for chunk_id, document in zip(result["ids"], result["documents"])
    ]


def collection_stats() -> dict:
    return {
        "collection": collection.name,
        "chunk_count": collection.count(),
        "embedding_model": EMBED_MODEL_NAME,
    }


def load_generator():
    """Load the Qwen model generate.py uses. Returns (model, tokenizer)."""
    import generate

    return generate.load_model()


def answer(question: str, n_results: int, max_new_tokens: int = 512) -> str:
    """Run the full pipeline end to end, collecting the streamed answer."""
    import generate

    return "".join(
        generate.stream_answer(
            question, n_results=n_results, max_new_tokens=max_new_tokens
        )
    )


def complete(prompt: str, max_new_tokens: int = 256) -> str:
    """One-shot local completion, used for offline dataset generation."""
    import torch

    import generate

    model, tokenizer = generate.load_model()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(generate.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from retrieve import retrieve

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

SYSTEM_PROMPT = (
    "You are a medical assistant, helping students prepare for an exam. Be concise and helpful."
    "Answer the question using only the context provided. If the answer is not in the context, say you don't know."
)

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    _model.eval()

    _model.generation_config.temperature = None
    _model.generation_config.top_p = None
    _model.generation_config.top_k = None

    return _model, _tokenizer


def stream_answer(
    question: str,
    n_results: int = 5,
    max_new_tokens: int = 512,
    chunks: list[str] | None = None,
) -> Iterator[str]:
    model, tokenizer = load_model()

    if chunks is None:
        chunks = retrieve(question, n_results=n_results)
    context = "\n---\n".join(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    def _generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    thread = Thread(target=_generate)
    thread.start()

    for token in streamer:
        yield token

    thread.join()


if __name__ == "__main__":
    while True:
        try:
            question = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() == "exit":
            break
        if not question.strip():
            continue
        for token in stream_answer(question):
            print(token, end="", flush=True)
        print()

"""Deterministic answer scoring -- no LLM judge.

These are proxies, not verdicts. They catch the failures that matter most in a
medical study tool (answering from nothing, ignoring the retrieved context,
refusing to abstain) without needing a grader model.

  keyword_recall  how much of the expected answer's content vocabulary appears
  grounding       how much of the answer's vocabulary appears in the retrieved
                  context -- a low score means the model spoke from parametric
                  memory rather than the chunks, which is the hallucination path
  abstained       whether the answer is an "I don't know", which should be true
                  for out-of-corpus questions and false for in-corpus ones
"""

import re

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her",
    "was", "one", "our", "out", "his", "has", "had", "have", "this", "that",
    "with", "from", "they", "will", "would", "there", "their", "what", "which",
    "when", "where", "been", "being", "were", "does", "did", "into", "than",
    "then", "them", "these", "those", "such", "some", "most", "more", "other",
    "also", "may", "might", "should", "could", "must", "used", "use", "using",
    "including", "include", "includes", "due", "per", "via", "about", "after",
    "before", "between", "during", "each", "both", "how", "why", "who", "whom",
    "its", "it's", "your", "patient", "patients", "common", "commonly",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")

ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "not in the context",
    "not provided in the context",
    "not mentioned in the context",
    "no information",
    "cannot determine",
    "can't determine",
    "context does not",
    "context doesn't",
    "unable to answer",
    "not available in the",
)


def content_words(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def keyword_recall(answer: str, expected: str) -> float:
    expected_words = content_words(expected)
    if not expected_words:
        return 0.0
    return len(content_words(answer) & expected_words) / len(expected_words)


def grounding(answer: str, context: str) -> float:
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    return len(answer_words & content_words(context)) / len(answer_words)


def abstained(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def score_answer(answer: str, expected: str, context: str) -> dict:
    return {
        "keyword_recall": keyword_recall(answer, expected),
        "grounding": grounding(answer, context),
        "abstained": abstained(answer),
        "answer_chars": len(answer),
        "empty": not answer.strip(),
    }

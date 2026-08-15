"""Optional cross-encoder reranking via fastembed's ONNX TextCrossEncoder.

This is the retrieve-then-rerank pattern from information retrieval
research: a bi-encoder (embeddings.py) scores query and document
independently, which is fast but loses interaction effects -- it can't
represent "this document answers that specific question" as well as a
model that reads both together. A cross-encoder does exactly that: it
jointly encodes (query, document) pairs, which is strictly more accurate
for relevance ranking but too slow to run over a whole corpus in a
real search engine (hence "retrieve [cheap] then rerank [accurate]").

Here the whole transcript IS the small corpus (hundreds of chunks, not
millions), so we rerank everything rather than just a top-k shortlist --
measured at ~0.2s warm for 300 chunks, well inside the hook's latency
budget. Optional and gracefully absent, same as embeddings.py: if
fastembed's reranker isn't available, callers skip this signal entirely.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

_MODEL = None
_MODEL_LOAD_ATTEMPTED = False


def available() -> bool:
    return _get_model() is not None


def _get_model():
    global _MODEL, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL
    _MODEL_LOAD_ATTEMPTED = True
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _MODEL = TextCrossEncoder(model_name=DEFAULT_MODEL_NAME)
    except Exception:
        _MODEL = None
    return _MODEL


def rerank_scores(query: str, texts: list[str]) -> Optional[np.ndarray]:
    """Return per-text relevance scores normalized to 0..1, or None if the
    reranker isn't available.

    ms-marco-MiniLM was trained on web-search query/passage pairs, and
    Claude Code transcript chunks (conversational turns, JSON tool-call
    blobs, raw command output) look nothing like search passages -- every
    logit measured on real transcripts came back deeply negative (roughly
    -11), even for the genuinely relevant chunk. A sigmoid was tried first
    and squashed everything to ~0.00002, indistinguishable from noise, even
    though the RELATIVE ordering of those logits was correct (the truly
    relevant chunk scored highest). Min-max normalizing per transcript
    recovers that ordering as a usable 0..1 signal -- same fix as the
    bi-encoder's domain-floor problem in scorer.py's _semantic_relevance.
    """
    model = _get_model()
    if model is None or not texts:
        return None
    if not query.strip():
        return None
    raw = np.array(list(model.rerank(query, texts)), dtype=np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-6:
        return np.zeros(len(texts))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

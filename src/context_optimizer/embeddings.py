"""Optional local semantic embeddings via fastembed (ONNX Runtime, no torch).

Fully optional and gracefully absent: if fastembed isn't installed, callers
fall back to the TF-IDF lexical scorer in scorer.py. fastembed was chosen
over sentence-transformers specifically for latency -- this module runs
inside a fresh subprocess on every Claude Code UserPromptSubmit hook (hooks
don't persist state between invocations), so torch's multi-second cold
import is not acceptable in that hot path. A small quantized ONNX model
loads warm in well under a second and embeds hundreds of chunks in
milliseconds (measured: ~0.3s warm import+load, ~0.35s for 300 chunks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
CACHE_DIR = Path.home() / ".claude" / "context-optimizer" / "embed_cache"

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
        from fastembed import TextEmbedding

        _MODEL = TextEmbedding(model_name=DEFAULT_MODEL_NAME)
    except Exception:
        _MODEL = None
    return _MODEL


def _cache_path() -> Path:
    safe_name = DEFAULT_MODEL_NAME.replace("/", "__")
    return CACHE_DIR / f"{safe_name}.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(json.dumps(cache))
    except Exception:
        pass  # cache is a pure speed optimization, never allowed to break scoring


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def embed_many(texts: list[str]) -> Optional[np.ndarray]:
    """Return an (n, EMBED_DIM) array, or None if fastembed isn't available.

    Cached by content hash across ALL sessions and projects (not just this
    one), since identical strings like "File updated" recur constantly --
    every cache hit is a model call avoided.
    """
    model = _get_model()
    if model is None:
        return None
    if not texts:
        return np.zeros((0, EMBED_DIM))

    cache = _load_cache()
    hashes = [_hash(t) for t in texts]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]

    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_vecs = list(model.embed(missing_texts))
        for i, vec in zip(missing_idx, new_vecs):
            cache[hashes[i]] = np.asarray(vec).tolist()
        _save_cache(cache)

    return np.array([cache[h] for h in hashes], dtype=np.float64)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between each row of `a` and vector `b`."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros(a.shape[0])
    denom = (a_norm.flatten() * b_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, (a @ b) / denom, 0.0)
    return sims

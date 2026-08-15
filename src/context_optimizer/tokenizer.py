"""Token counting with a graceful fallback chain.

Claude does not expose a free local tokenizer, so we approximate:
1. tiktoken's cl100k_base encoding, if installed (closest widely-available proxy).
2. A char/4 heuristic (Anthropic's own published rule of thumb) otherwise.

Every count returned by this module is an ESTIMATE. Callers must not present
it as exact, and reports should say so.
"""
from __future__ import annotations

from functools import lru_cache

_ENCODING = None
_ENCODING_LOAD_ATTEMPTED = False


def _get_encoding():
    global _ENCODING, _ENCODING_LOAD_ATTEMPTED
    if _ENCODING_LOAD_ATTEMPTED:
        return _ENCODING
    _ENCODING_LOAD_ATTEMPTED = True
    try:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODING = None
    return _ENCODING


def count_tokens(text: str) -> int:
    """Estimate token count for a string. Never raises."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Fallback heuristic: ~4 characters per token (Anthropic's published estimate).
    return max(1, len(text) // 4)


def is_estimated() -> bool:
    """True if counts come from the char/4 fallback rather than a real tokenizer."""
    return _get_encoding() is None

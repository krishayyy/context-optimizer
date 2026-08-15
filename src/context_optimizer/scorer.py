"""Relevance scoring for transcript Chunks.

Relevance is hybrid, lexical + semantic:
  - TF-IDF cosine + idf-weighted term overlap (pure numpy, zero dependency,
    always available) -- strong on exact identifiers, filenames, and rare
    distinctive tokens that a general-purpose embedding model may underweight.
  - Local ONNX sentence embeddings via fastembed, when installed -- strong
    on paraphrase and synonymy ("auth" <-> "login") that no bag-of-words
    method can see. Optional and gracefully absent: falls back to lexical
    alone if fastembed isn't installed. See embeddings.py for why fastembed
    specifically (latency in a per-prompt hook) over sentence-transformers.
  - The stronger of the two signals wins per chunk, since they fail in
    different, non-overlapping cases.
  - A cross-encoder reranker (reranker.py, also fastembed/ONNX) was built
    and benchmarked as a third signal -- the standard "retrieve then
    rerank" pattern from IR research -- but is OFF by default: a weight
    sweep showed it never improves this benchmark and regresses it at full
    weight. See RelevanceScorer.__init__ for the measured numbers. Kept
    available (use_reranker=True) for anyone experimenting with a
    different cross-encoder model, not claimed as a win here.

On top of relevance:
  - Relevance propagates forward from a user message to the tool calls and
    results it triggers within the same turn (a tool result rarely restates
    the request that caused it in its own words).
  - Files currently modified per `git status` get a relevance floor,
    independent of text similarity -- that a file is being actively edited
    right now is a stronger, free signal than any text model.
  - A light recency term (must not dominate -- see scenario_old_but_relevant
    in benchmarks/, which exists specifically to catch recency dominating).
  - Staleness penalties (superseded file reads, resolved errors).

Output is a `score` in roughly [0, 1] per chunk, plus the reason(s) that
drove it, so the digest can explain *why* something is a prune candidate
instead of just asserting a number.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from . import embeddings, reranker
from .parser import Chunk

DEFAULT_HOOK_TIMEOUT_SECONDS = 8.0

# Cap on how many characters of a chunk get embedded/reranked. Bounds worst-
# case sequence length, which matters doubly for embedding: fastembed pads
# every text in a batch to its longest member, so one long chunk inflates
# its whole batch (see embeddings.py's length-sorted batching, the much
# bigger lever for that). Measured on a real 1,136-chunk session, on top of
# length-sorting: cap=2000 -> 16.3s, cap=500 -> 12.0s cold. 800 is a
# middle ground -- most of that gain without truncating real content (like
# a multi-hunk diff) down to near-nothing.
EMBED_TEXT_CAP = 800


def _run_with_timeout(fn, timeout_seconds: float | None):
    """Run fn() and return its result, or None if it doesn't finish within
    timeout_seconds (None = no timeout, run inline).

    Uses a plain daemon thread rather than concurrent.futures.ThreadPoolExecutor:
    the executor's default shutdown behavior JOINS all pending work at
    interpreter exit, which would silently defeat the whole point of a
    timeout -- the process would hang waiting for the abandoned embedding
    call anyway. A daemon thread never blocks process exit, so a timed-out
    call is genuinely abandoned, not just deferred. Discovered empirically:
    a real ~1100-chunk session's first (cold cache) embedding pass took
    ~103s, and a hook that can silently block a prompt for 100+ seconds on
    someone's first long session is not acceptable -- see
    benchmarks/methodology in docs/methodology.md for the measurement.
    """
    if timeout_seconds is None:
        return fn()
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except Exception:
            result["value"] = None

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        return None  # timed out; thread abandoned in the background, never awaited
    return result.get("value")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# Minimal stopword list -- enough to stop "the", "a", "is" from swamping
# the vocabulary without pulling in an NLP dependency.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "with", "this",
    "that", "it", "as", "by", "from", "we", "you", "i", "your", "our",
}

MODIFIED_FILE_RELEVANCE_FLOOR = 0.75


def _tokenize(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in _STOPWORDS
    ]


@dataclass
class ScoredChunk:
    chunk: Chunk
    relevance: float  # 0..1, max(lexical, semantic) after propagation/boosts
    recency: float  # 0..1, later = higher
    staleness_penalty: float  # 0..1, higher = more stale
    score: float  # final combined score, 0..1 (higher = more worth keeping)
    reasons: list[str]
    semantic_used: bool = False  # True if embeddings contributed on this chunk


class RelevanceScorer:
    def __init__(
        self,
        relevance_weight: float = 0.82,
        recency_weight: float = 0.05,
        staleness_weight: float = 0.5,
        use_embeddings: bool = True,
        semantic_weight: float = 0.5,
        use_reranker: bool = False,
        rerank_weight: float = 1.0,
    ):
        # recency is intentionally a light tie-breaker, not a first-order term:
        # weighting it heavily is exactly the recency bias this tool exists to
        # counteract -- it would let irrelevant-but-recent chunks outscore
        # relevant-but-old ones (see benchmarks/scenarios.py::scenario_old_but_relevant).
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.staleness_weight = staleness_weight
        self.use_embeddings = use_embeddings
        # semantic_weight dampens the embedding signal's contribution to the
        # lexical/semantic max-blend. 0.5 is not a guess: benchmarked sweep
        # over {0.3, 0.5, 0.7, 1.0} against benchmarks/scenarios.py gave mean
        # F1 of {0.84, 0.96, 0.94, 0.64} -- 0.5 was the clear best and 1.0
        # (naive equal-weight max) was a measured regression. See the note
        # in score() for why equal weighting hurts on this domain.
        self.semantic_weight = semantic_weight
        # use_reranker defaults OFF, on measured evidence, not caution for
        # its own sake: a benchmark sweep of rerank_weight in
        # {0.0, 0.2, ..., 1.0} against benchmarks/scenarios.py showed ZERO
        # improvement at any weight up to 0.8, and a regression at 1.0
        # (stale_reads F1 1.00 -> 0.67). ms-marco-MiniLM-L-6-v2 is trained
        # on web-search query/passage pairs; raw scores on real transcript
        # chunks (conversational turns, JSON tool blobs, command output)
        # came back deeply negative even for the genuinely relevant chunk
        # (see reranker.py), and per-transcript normalization recovers a
        # correctly-ORDERED signal that still isn't strong enough to beat
        # lexical+semantic in the max-blend. The code is kept and available
        # via use_reranker=True for anyone who wants to experiment with a
        # different cross-encoder model, but it is not a proven win here --
        # re-run benchmarks/run_benchmark.py before ever flipping this on
        # by default.
        self.use_reranker = use_reranker
        self.rerank_weight = rerank_weight

    def score(
        self,
        chunks: list[Chunk],
        task_query: str,
        modified_files: set | None = None,
        timeout_seconds: float | None = None,
    ) -> list[ScoredChunk]:
        """timeout_seconds bounds the semantic/rerank signals only (the
        lexical signal is pure numpy and always fast). None means no bound
        -- appropriate for `context-optimizer report`, where a user asked
        for the full-quality result and is fine waiting. Hooks should pass
        a real bound (see DEFAULT_HOOK_TIMEOUT_SECONDS): a cold embedding
        cache on a large real session was measured at ~103s, and a hook is
        not allowed to block someone's prompt for that long -- on timeout,
        scoring silently falls back to lexical-only for that invocation.
        """
        if not chunks:
            return []

        lexical, coverage_debug = self._lexical_relevance(chunks, task_query)
        semantic = (
            _run_with_timeout(lambda: self._semantic_relevance(chunks, task_query), timeout_seconds)
            if self.use_embeddings
            else None
        )
        rerank = (
            _run_with_timeout(lambda: self._rerank_relevance(chunks, task_query), timeout_seconds)
            if self.use_reranker
            else None
        )

        # Each signal is damped by a benchmark-calibrated weight before the
        # max-blend, not assumed to be an equal or dominant partner -- see
        # semantic_weight/rerank_weight docstrings above and
        # benchmarks/run_benchmark.py, which is the actual arbiter.
        candidates = [lexical]
        if semantic is not None:
            candidates.append(semantic * self.semantic_weight)
        if rerank is not None:
            candidates.append(rerank * self.rerank_weight)

        raw_relevance = np.maximum.reduce(candidates)
        semantic_used = (
            (candidates[1] >= lexical) if semantic is not None else np.zeros(len(chunks), dtype=bool)
        )

        relevance = self._propagate_relevance(chunks, raw_relevance)
        relevance = self._apply_modified_file_floor(chunks, relevance, modified_files)

        n = len(chunks)
        results: list[ScoredChunk] = []
        for i, chunk in enumerate(chunks):
            recency = (i + 1) / n
            staleness, reasons = self._staleness(chunk)

            combined = (
                self.relevance_weight * relevance[i]
                + self.recency_weight * recency
                - self.staleness_weight * staleness
            )
            combined = float(np.clip(combined, 0.0, 1.0))

            if relevance[i] < 0.05 and not reasons:
                reasons.append("low semantic relevance to current task")

            results.append(
                ScoredChunk(
                    chunk=chunk,
                    relevance=float(relevance[i]),
                    recency=recency,
                    staleness_penalty=staleness,
                    score=combined,
                    reasons=reasons,
                    semantic_used=bool(semantic_used[i]),
                )
            )
        return results

    @staticmethod
    def _lexical_relevance(
        chunks: list[Chunk], task_query: str
    ) -> tuple[np.ndarray, dict]:
        docs = [_tokenize(c.text) for c in chunks]
        query_tokens = _tokenize(task_query)

        vocab = sorted({tok for doc in docs for tok in doc} | set(query_tokens))
        if not vocab:
            return np.zeros(len(chunks)), {}
        vocab_index = {tok: i for i, tok in enumerate(vocab)}

        n_docs = len(docs) + 1
        df = np.zeros(len(vocab))
        for doc in docs + [query_tokens]:
            for tok in set(doc):
                df[vocab_index[tok]] += 1
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        def tfidf_vec(doc: list[str]) -> np.ndarray:
            vec = np.zeros(len(vocab))
            if not doc:
                return vec
            for tok in doc:
                vec[vocab_index[tok]] += 1.0
            vec = (vec / len(doc)) * idf
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        query_vec = tfidf_vec(query_tokens)
        query_norm = np.linalg.norm(query_vec)

        query_token_set = set(query_tokens)
        query_idf_mass = sum(idf[vocab_index[t]] for t in query_token_set) or 1.0

        relevance = np.zeros(len(chunks))
        for i, doc in enumerate(docs):
            doc_vec = tfidf_vec(doc)
            cosine = 0.0
            if query_norm > 0 and np.linalg.norm(doc_vec) > 0:
                cosine = float(np.clip(np.dot(doc_vec, query_vec), 0.0, 1.0))

            # Cosine dilutes fast on short texts: a chunk with one highly
            # distinctive query term (e.g. "snake_case") among unrelated
            # words can score near zero on cosine despite being the
            # relevant hit. Coverage asks what fraction of the QUERY's
            # distinctive vocabulary this chunk contains instead -- more
            # robust for short, sparse chunks. Take the stronger signal.
            matched = query_token_set & set(doc)
            coverage = (
                sum(idf[vocab_index[t]] for t in matched) / query_idf_mass
                if matched
                else 0.0
            )
            relevance[i] = float(np.clip(max(cosine, coverage), 0.0, 1.0))

        return relevance, {"vocab_size": len(vocab)}

    @staticmethod
    def _semantic_relevance(chunks: list[Chunk], task_query: str) -> np.ndarray | None:
        if not task_query.strip():
            return None
        texts = [c.text[:EMBED_TEXT_CAP] for c in chunks]
        chunk_vecs = embeddings.embed_many(texts)
        query_vec = embeddings.embed_many([task_query])
        if chunk_vecs is None or query_vec is None or len(query_vec) == 0:
            return None
        sims = embeddings.cosine_sim(chunk_vecs, query_vec[0])

        # bge-style embeddings carry a high "same domain" floor: two chunks
        # that are both just generically about programming sit at cosine
        # ~0.5-0.65 regardless of whether either is actually relevant to the
        # CURRENT task, and genuine relevance only separates out in roughly
        # the top 0.15-0.3 of the range (measured: an unrelated "check the
        # Python version" aside sat at 0.59-0.64 in the same session where
        # the actual active task scored 0.80-0.86). A fixed global rescale
        # assumes bounds that don't hold across embedding models or topics,
        # so normalize relative to what THIS transcript actually produced --
        # same principle as the natural-gap prune cutoff in digest.py.
        lo, hi = float(sims.min()), float(sims.max())
        if hi - lo < 1e-6:
            return np.zeros(len(chunks))
        return np.clip((sims - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _rerank_relevance(chunks: list[Chunk], task_query: str) -> np.ndarray | None:
        if not task_query.strip():
            return None
        texts = [c.text[:EMBED_TEXT_CAP] for c in chunks]
        scores = reranker.rerank_scores(task_query, texts)
        return scores

    @staticmethod
    def _propagate_relevance(
        chunks: list[Chunk], raw_relevance: np.ndarray, decay: float = 0.9
    ) -> np.ndarray:
        """Let a user message's relevance flow forward to the tool calls and
        results it triggers, within the same turn.

        A tool result like "File updated" or a tool_use JSON blob shares
        almost no words -- and often little semantic content -- with the
        request that caused it, even when it IS that request's execution.
        Treating each chunk as independent throws that causal link away.
        A turn boundary is any user_message chunk; everything until the next
        one inherits a decayed share of that message's own relevance,
        capped by its own score (propagation only raises relevance, never
        lowers a chunk that scored higher independently).
        """
        propagated = raw_relevance.copy()
        turn_relevance = 0.0
        for i, chunk in enumerate(chunks):
            if chunk.kind == "user_message":
                turn_relevance = raw_relevance[i]
            else:
                propagated[i] = max(propagated[i], turn_relevance * decay)
        return propagated

    @staticmethod
    def _apply_modified_file_floor(
        chunks: list[Chunk], relevance: np.ndarray, modified_files: set | None
    ) -> np.ndarray:
        """A file with uncommitted changes right now is part of the active
        task almost by definition -- no text model needed to establish that.
        This is a floor, not an override: it can only raise relevance for
        chunks touching that file, never lower it.
        """
        if not modified_files:
            return relevance
        boosted = relevance.copy()
        for i, chunk in enumerate(chunks):
            if chunk.file_path and chunk.file_path in modified_files:
                boosted[i] = max(boosted[i], MODIFIED_FILE_RELEVANCE_FLOOR)
        return boosted

    @staticmethod
    def _staleness(chunk: Chunk) -> tuple[float, list[str]]:
        reasons: list[str] = []
        penalty = 0.0
        if chunk.superseded:
            penalty += 0.6
            reasons.append(f"file since overwritten ({chunk.file_path})")
        if chunk.resolved_error:
            penalty += 0.5
            reasons.append("error later resolved by a successful retry")
        if chunk.kind == "tool_call":
            # Raw tool-call summaries are low value once their result exists;
            # the result chunk carries the information that matters.
            penalty += 0.15
        return min(penalty, 1.0), reasons


def build_task_query(
    chunks: list[Chunk], last_n_user_messages: int = 3, extra_context: str = ""
) -> str:
    """Derive a task query from the most recent user messages, optionally
    augmented with project signals (e.g. branch name tokens -- see
    project_context.py) that often compress the task into a few words.
    """
    user_msgs = [c.text for c in chunks if c.kind == "user_message"]
    query = "\n".join(user_msgs[-last_n_user_messages:])
    if extra_context.strip():
        query = f"{query}\n{extra_context.strip()}"
    return query

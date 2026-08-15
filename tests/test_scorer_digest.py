import time

from context_optimizer.digest import build_digest
from context_optimizer.parser import Chunk
from context_optimizer.scorer import RelevanceScorer, _run_with_timeout, build_task_query


def make_chunk(index, kind, text, tokens=None, **kwargs):
    return Chunk(
        index=index,
        uuid=f"c{index}",
        kind=kind,
        role="user" if kind == "user_message" else "assistant",
        text=text,
        tokens=tokens if tokens is not None else max(1, len(text) // 4),
        **kwargs,
    )


def test_relevant_chunk_scores_higher_than_irrelevant():
    chunks = [
        make_chunk(0, "user_message", "what's the weather like in Paris today"),
        make_chunk(1, "assistant_message", "add rate limiting to the login endpoint using IP address"),
        make_chunk(2, "user_message", "add rate limiting to the login endpoint per IP address"),
    ]
    query = "add rate limiting to the login endpoint per IP address"
    scored = RelevanceScorer().score(chunks, query)
    weather = next(s for s in scored if "weather" in s.chunk.text)
    rate_limit = next(s for s in scored if "rate limiting" in s.chunk.text)
    assert rate_limit.relevance > weather.relevance


def test_superseded_chunk_gets_penalized():
    chunks = [
        make_chunk(0, "tool_result", "old file content", tool_name="Read", file_path="x.py", superseded=True),
        make_chunk(1, "tool_result", "old file content", tool_name="Read", file_path="y.py", superseded=False),
    ]
    scored = RelevanceScorer().score(chunks, "irrelevant query")
    superseded = next(s for s in scored if s.chunk.file_path == "x.py")
    fresh = next(s for s in scored if s.chunk.file_path == "y.py")
    assert superseded.staleness_penalty > fresh.staleness_penalty
    assert superseded.score <= fresh.score


def test_build_digest_reclaimable_tokens_sum_correctly():
    chunks = [
        make_chunk(0, "tool_result", "irrelevant noise " * 50, tokens=500, tool_name="Grep"),
        make_chunk(1, "user_message", "add rate limiting to login", tokens=10),
    ]
    scored = RelevanceScorer().score(chunks, "add rate limiting to login")
    digest = build_digest(scored, window_size=10_000)
    assert digest.total_tokens == 510
    assert digest.reclaimable_tokens == sum(
        sc.chunk.tokens for sc in scored if sc.score < 0.35
    )


def test_build_task_query_uses_last_n_user_messages():
    chunks = [
        make_chunk(0, "user_message", "first task"),
        make_chunk(1, "user_message", "second task"),
        make_chunk(2, "user_message", "third task"),
    ]
    query = build_task_query(chunks, last_n_user_messages=2)
    assert "first task" not in query
    assert "second task" in query
    assert "third task" in query


def test_empty_chunks_returns_empty_digest():
    scored = RelevanceScorer().score([], "anything")
    digest = build_digest(scored, window_size=1000)
    assert digest.total_tokens == 0
    assert digest.prune_candidates == []


def test_run_with_timeout_returns_none_and_does_not_block_on_slow_call():
    def slow():
        time.sleep(2)
        return "should never see this"

    t0 = time.time()
    result = _run_with_timeout(slow, timeout_seconds=0.2)
    elapsed = time.time() - t0

    assert result is None
    assert elapsed < 1.0  # must return promptly, not wait for the slow call


def test_run_with_timeout_returns_result_when_fast_enough():
    result = _run_with_timeout(lambda: "ok", timeout_seconds=5.0)
    assert result == "ok"


def test_score_falls_back_to_lexical_when_semantic_scoring_times_out():
    """Real-world finding: a cold embedding cache on a large real session
    measured ~103s for the first pass. score() must never let that block a
    hook -- timeout_seconds must produce a valid (lexical-only) result
    instead of hanging or raising.
    """

    class SlowScorer(RelevanceScorer):
        @staticmethod
        def _semantic_relevance(chunks, task_query):
            time.sleep(2)
            return None  # would never be reached under a real timeout

    chunks = [make_chunk(0, "user_message", "add rate limiting to login")]

    t0 = time.time()
    scored = SlowScorer().score(chunks, "add rate limiting to login", timeout_seconds=0.2)
    elapsed = time.time() - t0

    assert elapsed < 1.0
    assert len(scored) == 1

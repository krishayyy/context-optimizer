"""UserPromptSubmit hook: nudges Claude (and the user) when context usage
crosses a threshold, with a short digest instead of a full report.

Design constraints, deliberately:
  - Never blocks the prompt. A context-optimization tool that breaks the
    user's actual workflow on a bug is worse than doing nothing.
  - Only nudges when usage crosses the threshold AND has grown meaningfully
    since the last nudge in this session, so it doesn't repeat itself every
    single turn once past the threshold.
  - The additionalContext string is capped at a handful of lines -- padding
    the model's context to warn about a context problem would be self-defeating.
"""
from __future__ import annotations

from .. import project_context
from ..digest import DEFAULT_WINDOW, NUDGE_USAGE_THRESHOLD, build_digest
from ..parser import parse_transcript
from ..scorer import RelevanceScorer, build_task_query
from .common import emit_json, get_env_float, get_env_int, load_state, read_hook_input, save_state

RENUDGE_DELTA = 0.10  # only re-nudge once usage grows another 10 points


def run() -> int:
    try:
        return _run()
    except Exception:
        # Advisory tool: any internal failure must be invisible to the user's
        # actual Claude Code session, never a blocker.
        return 0


def _run() -> int:
    data = read_hook_input()
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path")
    cwd = data.get("cwd", "")
    if not transcript_path:
        return 0

    window = get_env_int("CONTEXT_OPTIMIZER_WINDOW", DEFAULT_WINDOW)
    threshold = get_env_float("CONTEXT_OPTIMIZER_THRESHOLD", NUDGE_USAGE_THRESHOLD)

    chunks = parse_transcript(transcript_path)
    if not chunks:
        return 0

    # Cheap pre-check before paying for relevance scoring (which loads an
    # embedding model): most prompts in most sessions are well under the
    # threshold, and there's no reason to score anything until they aren't.
    cheap_usage_pct = sum(c.tokens for c in chunks) / window if window else 0.0
    if cheap_usage_pct < threshold:
        return 0

    proj = project_context.gather(cwd)
    task_query = build_task_query(chunks, extra_context=proj.branch_tokens_text())
    scored = RelevanceScorer().score(chunks, task_query, modified_files=proj.modified_files)
    digest = build_digest(scored, window_size=window)

    if digest.usage_pct < threshold:
        return 0

    state = load_state(session_id)
    last_nudged_pct = state.get("last_nudged_pct", 0.0)
    if digest.usage_pct < last_nudged_pct + RENUDGE_DELTA:
        return 0  # already nudged recently for roughly this usage level

    nudge = digest.short_nudge()
    if not nudge:
        return 0

    save_state(session_id, {"last_nudged_pct": digest.usage_pct})

    emit_json(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": nudge,
            },
            "systemMessage": (
                f"context-optimizer: session at {digest.usage_pct:.0%} of context window "
                f"-- run `context-optimizer report` for the full breakdown."
            ),
        }
    )
    return 0

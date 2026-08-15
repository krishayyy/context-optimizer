"""PreCompact hook: advisory only, never blocks by default.

PreCompact can exit 2 to block compaction, but the hook input doesn't
distinguish "proactive, context still has room" from "recovering from an
API error that already happened" -- both report trigger="auto". Blocking
the second case surfaces the API error and fails the user's in-flight
request. Given that ambiguity, blocking is not a safe default: this hook
only surfaces a systemMessage with a better /compact instruction string
so the user can see it and, if they want, Ctrl+C and paste it into a
manual `/compact <instructions>` next time instead.
"""
from __future__ import annotations

from .. import project_context
from ..digest import DEFAULT_WINDOW, build_digest
from ..parser import parse_transcript
from ..scorer import RelevanceScorer, build_task_query
from .common import emit_json, get_env_int, read_hook_input


def run() -> int:
    try:
        return _run()
    except Exception:
        return 0


def _run() -> int:
    data = read_hook_input()
    transcript_path = data.get("transcript_path")
    trigger = data.get("trigger", "auto")
    cwd = data.get("cwd", "")
    if not transcript_path:
        return 0

    window = get_env_int("CONTEXT_OPTIMIZER_WINDOW", DEFAULT_WINDOW)
    chunks = parse_transcript(transcript_path)
    if not chunks:
        return 0

    proj = project_context.gather(cwd)
    task_query = build_task_query(chunks, extra_context=proj.branch_tokens_text())
    scored = RelevanceScorer().score(chunks, task_query, modified_files=proj.modified_files)
    digest = build_digest(scored, window_size=window)

    if not digest.prune_candidates:
        return 0

    suggestion = digest.compact_instructions()
    kind = "Manual /compact" if trigger == "manual" else "Auto-compact"
    emit_json(
        {
            "systemMessage": (
                f"context-optimizer: {kind} is about to summarize "
                f"{digest.total_tokens:,} tokens generically. A task-aware "
                f"/compact instruction is available -- next time, try:\n"
                f"/compact {suggestion}"
            )
        }
    )
    return 0

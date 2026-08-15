"""Shared plumbing for Claude Code hook entry points.

Hook contract (per Claude Code docs): JSON arrives on stdin, common fields
are session_id/transcript_path/cwd/hook_event_name. Output is either JSON
on stdout (parsed if it starts with '{') or, for UserPromptSubmit/SessionStart,
plain text stdout that Claude Code adds as model-visible context.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

STATE_DIR = Path.home() / ".claude" / "context-optimizer" / "state"


def read_hook_input() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def get_env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def load_state(session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    path = STATE_DIR / f"{session_id}.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_state(session_id: str, state: dict[str, Any]) -> None:
    if not session_id:
        return
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / f"{session_id}.json").write_text(json.dumps(state))
    except Exception:
        pass  # advisory-only tool: a failed write must never break the session

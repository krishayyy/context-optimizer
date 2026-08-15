"""Installs/removes context-optimizer's hooks in a Claude Code settings.json,
merging safely instead of overwriting -- the whole point is that someone
with their own existing hooks configured doesn't lose them.

This exists because "open hooks/settings-snippet.json and hand-edit your
settings.json" is real friction that stops people from actually trying the
tool. `context-optimizer install` should be the entire onboarding step.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

OUR_COMMANDS = {
    "context-optimizer hook-user-prompt-submit",
    "context-optimizer hook-pre-compact",
}

_SNIPPET = {
    "hooks": {
        "UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "context-optimizer hook-user-prompt-submit"}]}
        ],
        "PreCompact": [
            {
                "matcher": "auto",
                "hooks": [{"type": "command", "command": "context-optimizer hook-pre-compact"}],
            },
            {
                "matcher": "manual",
                "hooks": [{"type": "command", "command": "context-optimizer hook-pre-compact"}],
            },
        ],
    }
}


def settings_path(project: bool = False) -> Path:
    base = Path.cwd() if project else Path.home()
    return base / ".claude" / "settings.json"


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _entry_has_our_command(entry: dict) -> bool:
    return any(h.get("command") in OUR_COMMANDS for h in entry.get("hooks", []))


def merge_hooks(settings: dict) -> dict:
    """Add our hook entries, replacing any of our own from a prior install
    (idempotent re-install) but leaving every other hook untouched."""
    settings = dict(settings)
    hooks = dict(settings.get("hooks", {}))
    for event_name, snippet_entries in _SNIPPET["hooks"].items():
        existing = [e for e in hooks.get(event_name, []) if not _entry_has_our_command(e)]
        existing.extend(snippet_entries)
        hooks[event_name] = existing
    settings["hooks"] = hooks
    return settings


def remove_hooks(settings: dict) -> dict:
    """Strip our hook entries only, leaving every other hook untouched."""
    settings = dict(settings)
    hooks = dict(settings.get("hooks", {}))
    for event_name in list(hooks.keys()):
        filtered = [e for e in hooks[event_name] if not _entry_has_our_command(e)]
        if filtered:
            hooks[event_name] = filtered
        else:
            del hooks[event_name]
    if hooks:
        settings["hooks"] = hooks
    else:
        settings.pop("hooks", None)
    return settings


def _write_with_backup(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(settings, indent=2) + "\n")


def install(project: bool = False) -> Path:
    path = settings_path(project)
    _write_with_backup(path, merge_hooks(load_settings(path)))
    return path


def uninstall(project: bool = False) -> Path:
    path = settings_path(project)
    _write_with_backup(path, remove_hooks(load_settings(path)))
    return path

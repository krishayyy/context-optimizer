"""Generates a synthetic Claude Code transcript for testing/benchmarking.

Simulates a realistic long session: an early unrelated task, a pile of
exploratory grep/read noise, a stale file read that gets overwritten, a
failed-then-fixed command, and a final task that should score highest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _user(uuid, text):
    return {"type": "user", "uuid": uuid, "timestamp": uuid, "message": {"role": "user", "content": text}}


def _assistant_text(uuid, text):
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _tool_call(uuid, tool_id, name, tool_input):
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
        },
    }


def _tool_result(uuid, tool_id, content, is_error=False):
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": is_error}
            ],
        },
    }


def build() -> list[dict]:
    records = []
    n = 0

    def uid():
        nonlocal n
        n += 1
        return f"u{n}"

    # --- Early, now-irrelevant task ---
    records.append(_user(uid(), "Can you check what Python version this project uses?"))
    records.append(_assistant_text(uid(), "Sure, let me check."))
    records.append(_tool_call(uid(), "t1", "Bash", {"command": "python --version"}))
    records.append(_tool_result(uid(), "t1", "Python 3.11.4"))
    records.append(_assistant_text(uid(), "This project uses Python 3.11.4."))

    # --- Noisy exploration, large low-signal dump ---
    records.append(_user(uid(), "search the repo for any usages of requests library"))
    records.append(_tool_call(uid(), "t2", "Grep", {"pattern": "import requests"}))
    big_dump = "\n".join(f"src/module_{i}.py:12: import requests" for i in range(200))
    records.append(_tool_result(uid(), "t2", big_dump))
    records.append(_assistant_text(uid(), "Found 200 usages across the codebase."))

    # --- A file read that later gets overwritten (stale) ---
    records.append(_user(uid(), "show me auth.py"))
    records.append(_tool_call(uid(), "t3", "Read", {"file_path": "auth.py"}))
    records.append(_tool_result(uid(), "t3", "def login(user, pw):\n    return check(user, pw)"))
    records.append(_assistant_text(uid(), "Here's the current login function."))

    # --- A failed command that gets fixed ---
    records.append(_user(uid(), "run the test suite"))
    records.append(_tool_call(uid(), "t4", "Bash", {"command": "pytest"}))
    records.append(_tool_result(uid(), "t4", "ModuleNotFoundError: pytest", is_error=True))
    records.append(_assistant_text(uid(), "pytest isn't installed, installing it now."))
    records.append(_tool_call(uid(), "t5", "Bash", {"command": "pip install pytest && pytest"}))
    records.append(_tool_result(uid(), "t5", "5 passed in 0.42s"))

    # --- The overwrite that makes the earlier Read stale ---
    records.append(_user(uid(), "actually, rewrite login() to use bcrypt for password hashing"))
    records.append(
        _tool_call(
            uid(),
            "t6",
            "Edit",
            {"file_path": "auth.py", "old_string": "def login", "new_string": "def login  # bcrypt"},
        )
    )
    records.append(_tool_result(uid(), "t6", "File updated"))
    records.append(_assistant_text(uid(), "Updated login() to use bcrypt for hashing."))

    # --- The CURRENT task -- this is what relevance scoring should favor ---
    records.append(
        _user(
            uid(),
            "Now add rate limiting to the login endpoint so it rejects more than "
            "5 attempts per minute per IP address",
        )
    )
    records.append(
        _assistant_text(
            uid(),
            "I'll add rate limiting to the login endpoint using a sliding window "
            "keyed on IP address, capped at 5 attempts per minute.",
        )
    )

    return records


def main():
    out_path = Path(__file__).parent / "sample_transcripts" / "sample_session.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in build():
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

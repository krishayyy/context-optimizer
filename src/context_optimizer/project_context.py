"""Signals about what's actively being worked on, gathered for free from the
project itself rather than inferred purely from conversation text.

Two signals, both strong and essentially zero-cost:
  - files currently modified per `git status` are, almost by definition,
    part of the active task -- no scoring model needed to know that
  - the current branch name is frequently a compressed description of the
    task itself (e.g. "fix/rate-limit-419" -> tokens "fix rate limit 419"
    feed straight into the relevance query)

Stack detection (package.json, pyproject.toml, etc.) is surfaced for
transparency in the report but deliberately NOT used to special-case
scoring weights per language/framework -- that path leads to unbenchmarked
per-stack tuning that's easy to overfit and hard to justify in a paper.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_STACK_MARKERS = {
    "package.json": "Node/JavaScript",
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java/Gradle",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
}


@dataclass
class ProjectContext:
    modified_files: set = field(default_factory=set)
    branch_name: str = ""
    stack_tags: list = field(default_factory=list)

    def branch_tokens_text(self) -> str:
        if not self.branch_name:
            return ""
        return " ".join(t for t in re.split(r"[/_\-]+", self.branch_name) if t)


def _run_git(cwd: str, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=2
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def _detect_stack(cwd: str) -> list[str]:
    try:
        p = Path(cwd)
        return [tag for fname, tag in _STACK_MARKERS.items() if (p / fname).exists()]
    except Exception:
        return []


def gather(cwd: str) -> ProjectContext:
    """Never raises: git absence, a non-repo cwd, or any subprocess failure
    just yields an empty/default ProjectContext instead of breaking scoring.
    """
    ctx = ProjectContext()
    if not cwd:
        return ctx

    status = _run_git(cwd, ["status", "--porcelain"])
    for line in status.splitlines():
        if len(line) <= 3:
            continue
        path = line[3:].strip()
        if " -> " in path:  # renames: "old -> new"
            path = path.split(" -> ")[-1].strip()
        if path:
            ctx.modified_files.add(path)

    ctx.branch_name = _run_git(cwd, ["branch", "--show-current"]).strip()
    ctx.stack_tags = _detect_stack(cwd)
    return ctx

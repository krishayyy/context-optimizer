from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import embeddings, installer, project_context
from .digest import DEFAULT_WINDOW, build_digest
from .parser import parse_transcript
from .scorer import RelevanceScorer, build_task_query
from .tokenizer import is_estimated

console = Console()


def _latest_transcript_for_cwd() -> str | None:
    """Best-effort guess at the current project's most recent transcript,
    mirroring how Claude Code lays out ~/.claude/projects/<encoded-cwd>/*.jsonl.
    """
    cwd = os.getcwd()
    encoded = cwd.replace("/", "-")
    pattern = str(Path.home() / ".claude" / "projects" / f"*{encoded}*" / "*.jsonl")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def cmd_report(args: argparse.Namespace) -> int:
    transcript_path = args.transcript or _latest_transcript_for_cwd()
    if not transcript_path:
        console.print(
            "[red]No transcript path given and none could be auto-detected. "
            "Pass one explicitly: context-optimizer report <path-to.jsonl>[/red]"
        )
        return 1

    chunks = parse_transcript(transcript_path)
    if not chunks:
        console.print(f"[yellow]No parseable chunks found in {transcript_path}[/yellow]")
        return 1

    proj = project_context.gather(os.getcwd())
    task_query = args.task or build_task_query(chunks, extra_context=proj.branch_tokens_text())
    scorer = RelevanceScorer()
    scored = scorer.score(chunks, task_query, modified_files=proj.modified_files)
    digest = build_digest(scored, window_size=args.window)

    est_note = " (estimated, no exact tokenizer installed)" if is_estimated() else ""
    semantic_note = " + local semantic embeddings" if embeddings.available() else " (lexical-only -- install the `semantic` extra for embeddings)"
    console.print(
        f"\n[bold]Context health[/bold]{est_note}: "
        f"{digest.total_tokens:,} / {digest.window_size:,} tokens "
        f"([{'red' if digest.usage_pct > 0.8 else 'yellow' if digest.usage_pct > 0.55 else 'green'}]"
        f"{digest.usage_pct:.0%}[/])"
    )
    console.print(f"Scoring: TF-IDF/overlap{semantic_note}")
    if proj.stack_tags:
        console.print(f"Detected stack: {', '.join(proj.stack_tags)}")
    if proj.branch_name:
        console.print(f"Git branch: {proj.branch_name}")
    if proj.modified_files:
        console.print(f"Modified files (relevance floor applied): {', '.join(sorted(proj.modified_files)[:8])}")
    console.print(f"Task query used for relevance scoring: [italic]{task_query[:200]!r}[/italic]\n")

    if digest.prune_candidates:
        table = Table(title=f"Top prune candidates (~{digest.reclaimable_tokens:,} tokens reclaimable)")
        table.add_column("Chunk", overflow="fold")
        table.add_column("Tokens", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Reason", overflow="fold")
        for sc in digest.prune_candidates[: args.top]:
            table.add_row(
                sc.chunk.label(),
                f"{sc.chunk.tokens:,}",
                f"{sc.score:.2f}",
                "; ".join(sc.reasons) or "-",
            )
        console.print(table)
    else:
        console.print("[green]Nothing crosses the prune threshold -- context looks lean.[/green]")

    console.print("\n[bold]Suggested /compact instructions:[/bold]")
    console.print(digest.compact_instructions())
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    path = installer.install(project=args.project)
    scope = "this project's" if args.project else "your global"
    console.print(f"[green]Installed.[/green] Hooks added to {scope} {path}")
    console.print("A backup of the previous file (if any) was saved alongside it as .bak.")
    if not embeddings.available():
        console.print(
            "\n[yellow]Note:[/yellow] semantic scoring isn't active yet -- run "
            "[bold]pip install \"context-optimizer[semantic]\"[/bold] to enable it "
            "(fully optional; lexical scoring works without it)."
        )
    console.print(
        "\nNothing else to do -- open a Claude Code session and you'll see a note "
        "in your terminal once it crosses ~55% of its context window."
    )
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    path = installer.uninstall(project=args.project)
    scope = "this project's" if args.project else "your global"
    console.print(f"[green]Removed.[/green] Hooks removed from {scope} {path}")
    return 0


def cmd_hook_user_prompt_submit(args: argparse.Namespace) -> int:
    from .hooks.user_prompt_submit import run

    return run()


def cmd_hook_pre_compact(args: argparse.Namespace) -> int:
    from .hooks.pre_compact import run

    return run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Analyze a transcript and print a context-health report")
    p_report.add_argument("transcript", nargs="?", help="Path to a Claude Code .jsonl transcript")
    p_report.add_argument("--task", help="Override the task query used for relevance scoring")
    p_report.add_argument("--top", type=int, default=10, help="Number of prune candidates to show")
    p_report.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Context window size in tokens")
    p_report.set_defaults(func=cmd_report)

    p_install = sub.add_parser("install", help="Add context-optimizer's hooks to your Claude Code settings.json")
    p_install.add_argument(
        "--project", action="store_true", help="Install to ./.claude/settings.json instead of the global one"
    )
    p_install.set_defaults(func=cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="Remove context-optimizer's hooks from settings.json")
    p_uninstall.add_argument("--project", action="store_true", help="Uninstall from the project settings.json")
    p_uninstall.set_defaults(func=cmd_uninstall)

    p_hook1 = sub.add_parser("hook-user-prompt-submit", help=argparse.SUPPRESS)
    p_hook1.set_defaults(func=cmd_hook_user_prompt_submit)

    p_hook2 = sub.add_parser("hook-pre-compact", help=argparse.SUPPRESS)
    p_hook2.set_defaults(func=cmd_hook_pre_compact)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

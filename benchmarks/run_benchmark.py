"""Benchmarks context-optimizer's relevance scoring against a naive
"prune the oldest N tokens" baseline, using synthetic scenarios with
engineered ground truth (see scenarios.py).

Metric: precision/recall/F1 of the *pruned set* against which chunks were
actually safe to prune, at a matched token budget so both strategies get
credit/blame for reclaiming the same number of tokens.

Run: .venv/bin/python benchmarks/run_benchmark.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from context_optimizer.digest import build_digest
from context_optimizer.parser import parse_transcript
from context_optimizer.scorer import RelevanceScorer

from scenarios import Scenario, all_scenarios

console = Console()


def _write(records: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def _prf(predicted: set[str], truth: set[str], labeled_universe: set[str]):
    predicted = predicted & labeled_universe
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def naive_oldest_first_prune(chunks, target_tokens: int) -> set[str]:
    """Baseline: drop the oldest chunks first until the token budget is hit.

    This mirrors the simplest possible context-management strategy (FIFO
    truncation) that a system with no relevance awareness would use.
    """
    pruned = set()
    reclaimed = 0
    for c in sorted(chunks, key=lambda c: c.index):
        if reclaimed >= target_tokens:
            break
        pruned.add(c.uuid)
        reclaimed += c.tokens
    return pruned


def run_scenario(scenario: Scenario) -> dict:
    path = _write(scenario.records)
    chunks = parse_transcript(path)
    scored = RelevanceScorer().score(chunks, scenario.task_query)
    digest = build_digest(scored, window_size=1_000_000)  # large window, we only care about ranking

    predicted_prune = {sc.chunk.uuid for sc in digest.prune_candidates}
    labeled_universe = scenario.should_prune | scenario.should_keep

    co_precision, co_recall, co_f1 = _prf(predicted_prune, scenario.should_prune, labeled_universe)

    target_tokens = sum(sc.chunk.tokens for sc in digest.prune_candidates)
    naive_prune = naive_oldest_first_prune(chunks, target_tokens)
    naive_precision, naive_recall, naive_f1 = _prf(naive_prune, scenario.should_prune, labeled_universe)

    return {
        "name": scenario.name,
        "n_chunks": len(chunks),
        "tokens_reclaimed": target_tokens,
        "co": (co_precision, co_recall, co_f1),
        "naive": (naive_precision, naive_recall, naive_f1),
    }


def main():
    results = [run_scenario(s) for s in all_scenarios()]

    table = Table(title="context-optimizer vs naive oldest-first pruning (equal token budget)")
    table.add_column("Scenario")
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens pruned", justify="right")
    table.add_column("CO Precision", justify="right")
    table.add_column("CO Recall", justify="right")
    table.add_column("CO F1", justify="right")
    table.add_column("Naive Precision", justify="right")
    table.add_column("Naive Recall", justify="right")
    table.add_column("Naive F1", justify="right")

    for r in results:
        co_p, co_r, co_f1 = r["co"]
        n_p, n_r, n_f1 = r["naive"]
        table.add_row(
            r["name"],
            str(r["n_chunks"]),
            str(r["tokens_reclaimed"]),
            f"{co_p:.2f}", f"{co_r:.2f}", f"{co_f1:.2f}",
            f"{n_p:.2f}", f"{n_r:.2f}", f"{n_f1:.2f}",
        )
    console.print(table)

    avg_co_f1 = sum(r["co"][2] for r in results) / len(results)
    avg_naive_f1 = sum(r["naive"][2] for r in results) / len(results)
    console.print(
        f"\nMean F1 -- context-optimizer: {avg_co_f1:.2f}  |  naive oldest-first: {avg_naive_f1:.2f}"
    )

    out_path = Path(__file__).parent / "results.md"
    with out_path.open("w") as f:
        f.write("# context-optimizer benchmark results\n\n")
        f.write(
            "Synthetic scenarios with engineered ground truth (see `scenarios.py`). "
            "Both strategies prune the same number of tokens per scenario; precision/recall "
            "are measured against which chunks were actually safe to prune.\n\n"
        )
        f.write("| Scenario | Chunks | Tokens pruned | CO P | CO R | CO F1 | Naive P | Naive R | Naive F1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            co_p, co_r, co_f1 = r["co"]
            n_p, n_r, n_f1 = r["naive"]
            f.write(
                f"| {r['name']} | {r['n_chunks']} | {r['tokens_reclaimed']} | "
                f"{co_p:.2f} | {co_r:.2f} | {co_f1:.2f} | {n_p:.2f} | {n_r:.2f} | {n_f1:.2f} |\n"
            )
        f.write(f"\n**Mean F1** -- context-optimizer: {avg_co_f1:.2f} | naive oldest-first: {avg_naive_f1:.2f}\n")
    console.print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

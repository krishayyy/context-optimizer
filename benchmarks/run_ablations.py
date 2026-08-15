"""Ablation sweeps: semantic_weight and rerank_weight, on the CURRENT
5-scenario benchmark set (run_benchmark.py's all_scenarios()).

These are re-run fresh here rather than reused from earlier development
notes, because the scenario set has changed since those weights were first
tuned (the `paraphrase` scenario was added after semantic_weight was
picked) -- reusing stale numbers from a different scenario set would be a
real inconsistency in a manuscript, not just a formatting issue.

Run: .venv/bin/python benchmarks/run_ablations.py
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

from run_benchmark import _prf, _write
from scenarios import all_scenarios

console = Console()


def mean_f1_at(**scorer_kwargs) -> float:
    f1s = []
    for scenario in all_scenarios():
        path = _write(scenario.records)
        chunks = parse_transcript(path)
        scored = RelevanceScorer(**scorer_kwargs).score(chunks, scenario.task_query)
        digest = build_digest(scored, window_size=1_000_000)
        predicted_prune = {sc.chunk.uuid for sc in digest.prune_candidates}
        labeled_universe = scenario.should_prune | scenario.should_keep
        _, _, f1 = _prf(predicted_prune, scenario.should_prune, labeled_universe)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def main():
    console.print("[bold]Semantic weight sweep[/bold] (5-scenario set, including paraphrase)")
    semantic_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    semantic_results = []
    for w in semantic_weights:
        f1 = mean_f1_at(semantic_weight=w, use_reranker=False)
        semantic_results.append((w, f1))

    t1 = Table()
    t1.add_column("semantic_weight", justify="right")
    t1.add_column("Mean F1", justify="right")
    for w, f1 in semantic_results:
        t1.add_row(f"{w:.1f}", f"{f1:.3f}")
    console.print(t1)

    console.print("\n[bold]Rerank weight sweep[/bold] (5-scenario set, semantic_weight=0.5 fixed)")
    rerank_weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    rerank_results = []
    for w in rerank_weights:
        f1 = mean_f1_at(semantic_weight=0.5, use_reranker=True, rerank_weight=w)
        rerank_results.append((w, f1))

    t2 = Table()
    t2.add_column("rerank_weight", justify="right")
    t2.add_column("Mean F1", justify="right")
    for w, f1 in rerank_results:
        t2.add_row(f"{w:.1f}", f"{f1:.3f}")
    console.print(t2)

    out_path = Path(__file__).parent / "ablations.md"
    with out_path.open("w") as f:
        f.write("# Ablation sweeps (5-scenario benchmark set)\n\n")
        f.write("## Semantic weight (lexical vs. embedding blend)\n\n")
        f.write("| semantic_weight | Mean F1 |\n|---|---|\n")
        for w, f1 in semantic_results:
            f.write(f"| {w:.1f} | {f1:.3f} |\n")
        f.write("\n## Rerank weight (cross-encoder contribution, semantic_weight=0.5 fixed)\n\n")
        f.write("| rerank_weight | Mean F1 |\n|---|---|\n")
        for w, f1 in rerank_results:
            f.write(f"| {w:.1f} | {f1:.3f} |\n")
    console.print(f"\nWrote {out_path}")

    return semantic_results, rerank_results


if __name__ == "__main__":
    main()

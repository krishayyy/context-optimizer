# context-optimizer benchmark results

Synthetic scenarios with engineered ground truth (see `scenarios.py`). Both strategies prune the same number of tokens per scenario; precision/recall are measured against which chunks were actually safe to prune.

| Scenario | Chunks | Tokens pruned | CO P | CO R | CO F1 | Naive P | Naive R | Naive F1 |
|---|---|---|---|---|---|---|---|---|
| stale_reads | 12 | 38 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.67 |
| resolved_errors | 9 | 60 | 0.75 | 1.00 | 0.86 | 0.75 | 1.00 | 0.86 |
| noisy_exploration | 12 | 1708 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| old_but_relevant | 11 | 47 | 1.00 | 1.00 | 1.00 | 0.60 | 0.50 | 0.55 |
| paraphrase | 8 | 24 | 1.00 | 0.50 | 0.67 | 0.00 | 0.00 | 0.00 |

**Mean F1** -- context-optimizer: 0.90 | naive oldest-first: 0.61

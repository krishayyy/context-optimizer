# Ablation sweeps (5-scenario benchmark set)

## Semantic weight (lexical vs. embedding blend)

| semantic_weight | Mean F1 |
|---|---|
| 0.0 | 0.881 |
| 0.3 | 0.805 |
| 0.5 | 0.905 |
| 0.7 | 0.833 |
| 1.0 | 0.593 |

## Rerank weight (cross-encoder contribution, semantic_weight=0.5 fixed)

| rerank_weight | Mean F1 |
|---|---|
| 0.0 | 0.905 |
| 0.2 | 0.905 |
| 0.3 | 0.905 |
| 0.4 | 0.905 |
| 0.5 | 0.905 |
| 0.6 | 0.905 |
| 0.8 | 0.905 |
| 1.0 | 0.838 |

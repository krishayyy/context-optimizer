# context-optimizer

**Your Claude Code sessions know when they're about to run out of context —
this makes them tell you what's actually safe to cut, instead of guessing.**

Long sessions accumulate junk: file reads that got overwritten an hour ago,
failed commands you already fixed, exploratory greps that went nowhere.
Claude Code's default auto-compact summarizes all of it generically. This
tool watches your context usage, figures out what's actually still relevant
to what you're working on *right now* (not just what's recent), and tells
you exactly what to drop — in your terminal, automatically, with zero
config after install.

```bash
git clone <this-repo> && cd context-optimizer
pip install -e ".[tokenizer,semantic]"
context-optimizer install
```

That's it. Nothing else to configure. Next time a session gets long, you'll
see a note like:

```
context-optimizer: session at 61% of context window -- run
`context-optimizer report` for the full breakdown.
```

Or run it yourself any time:

```bash
context-optimizer report
```

<details>
<summary><b>See it in action</b></summary>

```
Context health: 142,318 / 200,000 tokens (71%)
Scoring: TF-IDF/overlap + local semantic embeddings
Detected stack: Python
Git branch: fix/rate-limit-419
Modified files (relevance floor applied): src/api/login.py, src/middleware/throttle.py

              Top prune candidates (~38,204 tokens reclaimable)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chunk            ┃ Tokens ┃ Score ┃ Reason                               ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Grep             │ 21,004 │  0.05 │ low relevance to current task        │
│ Read(old_api.py) │  8,412 │  0.00 │ file since overwritten (old_api.py)  │
│ Bash             │  1,220 │  0.11 │ error later resolved by a retry      │
└──────────────────┴────────┴───────┴───────────────────────────────────────┘

Suggested /compact instructions:
Preserve in full: the current task/goal, all code changes actually made...
```

</details>

## Uninstall

```bash
context-optimizer uninstall
```

Removes exactly what it added and nothing else — see "How install works" below.

## Why this beats just letting Claude Code auto-compact

The obvious naive strategy (what most context management amounts to) is
"drop the oldest stuff first." That fails whenever something old is still
load-bearing — a naming convention agreed on early, a constraint stated
once and never repeated — while something recent is a throwaway tangent.
Measured on a synthetic case built exactly for this: naive oldest-first
pruning scores 0.55 F1; this tool scores 1.00. Full benchmark below.

## How it works, briefly

Runs entirely locally, no API calls, no network at runtime, ~0.5s warm on
a 300-message session:

1. Parses your session transcript into chunks and tracks which file reads
   got overwritten, and which errors got fixed by a later retry.
2. Scores relevance to *your current task* two ways — keyword/TF-IDF
   matching (catches exact identifiers and filenames) and local semantic
   embeddings (catches paraphrase — "auth" relates to "login" — that
   keyword matching alone misses).
3. Boosts anything touching a file you currently have uncommitted changes
   in — that's about as strong a "this is the active task" signal as exists.
4. Finds the natural cutoff between "still relevant" and "safe to drop"
   in your session's own score distribution, instead of a fixed threshold
   that doesn't generalize across sessions.

Full technical writeup with the actual benchmark methodology, what got
tried and rejected, and known limitations: see
[`docs/methodology.md`](docs/methodology.md).

## How install works

`context-optimizer install` merges two hook entries into your
`~/.claude/settings.json` (or `--project` for just this repo's
`.claude/settings.json`). It only ever touches its own entries — any other
hooks you have configured are left completely alone, and the previous file
is backed up to `settings.json.bak` before anything is written. See
[`installer.py`](src/context_optimizer/installer.py) if you want to read
exactly what it does before running it, or just inspect the diff yourself
after installing.

Both extras are optional and degrade gracefully if missing:
- Without `tokenizer` (`tiktoken`), token counts fall back to a char/4
  heuristic and the report says so.
- Without `semantic` (`fastembed`), scoring runs on the lexical signal
  alone — fully functional, just without paraphrase understanding.

## Manual usage

```bash
context-optimizer report                                       # latest session in this project
context-optimizer report path/to/session.jsonl --task "..."     # a specific one, with your own task description
```

## Development

```bash
git clone <this repo> && cd context-optimizer
pip install -e ".[dev,tokenizer,semantic]"
pytest tests/ -q
python benchmarks/run_benchmark.py   # rerun the ground-truth benchmark
```

# Methodology

Full technical detail behind context-optimizer: what it can and can't do,
exactly how scoring works, the benchmark it's measured against, and what
was tried and rejected. See the [README](../README.md) for the pitch and
install instructions.

## What this actually is

Claude Code does not expose a supported way for a third-party tool to
rewrite the conversation history it sends to the model. Its `PreCompact`
hook can only allow or block compaction; it cannot supply a custom summary.
`UserPromptSubmit` can only *add* plain-text context, not edit or remove
existing history.

So this is **advisory, not a drop-in replacement for Claude Code's
compaction**. Concretely, it:

- Watches context usage every prompt and, once it crosses a threshold,
  surfaces a short digest of what's safe to prune and why — visible to
  both you and the model, via `hookSpecificOutput.additionalContext` /
  `systemMessage`.
- Before an auto- or manual `/compact` runs, prints a task-aware
  `/compact <instructions>` string you can use instead of the default
  generic summarization prompt.
- Ships a standalone `context-optimizer report <transcript>` command for a
  full breakdown any time, not just at hook trigger points.

It does not, and cannot, silently rewrite your context window. Every claim
above is exactly what the two hook events are documented to support —
nothing here depends on an undocumented internal that could break on the
next Claude Code release.

## How scoring works

No API calls, no network at runtime, nothing leaves your machine — fast
enough to run synchronously inside a Claude Code hook on every prompt
submit (measured: ~0.5s warm, including model load, on a 300-chunk session):

1. **Parse** the transcript into chunks (user/assistant messages, tool
   calls, tool results), tracking which file reads got superseded by a
   later edit and which tool errors were later fixed by a successful retry.
2. **Score relevance, lexical + semantic:**
   - TF-IDF cosine + idf-weighted term overlap (zero dependency, catches
     exact identifiers/filenames that embeddings can miss).
   - Local ONNX sentence embeddings via `fastembed` (optional; catches
     paraphrase — "auth" relates to "login" — that no bag-of-words method
     can see). Chosen over `sentence-transformers` specifically because
     this runs as a **fresh subprocess on every prompt** (hooks don't
     persist state) — torch's multi-second cold import isn't acceptable
     in that hot path; a quantized ONNX model loads warm in ~0.1s.
   - The two are blended, not simply maxed: an equal-weight blend was
     **benchmarked and rejected** — it regressed mean F1 from 0.96 to
     0.64, because a small general-purpose embedding model rates any two
     pieces of programming-related text as broadly similar regardless of
     actual task relevance (measured cosine floor: ~0.5-0.65 for
     *unrelated* dev chatter). Semantic runs at 0.5x weight as a
     supporting signal instead. Re-run `benchmarks/run_benchmark.py`
     before ever changing that number.
3. **Propagate relevance forward** from a user message to the tool calls
   and results it triggers within the same turn — a tool result rarely
   restates the request that caused it in its own words.
4. **Boost files with uncommitted changes.** Anything `git status` reports
   as modified is part of the active task almost by definition — a free,
   zero-latency signal independent of any text model. The current branch
   name (tokenized: `fix/rate-limit-419` -> `fix rate limit 419`) also
   feeds into the relevance query, since branch names often compress the
   task into a few words.
5. **Penalize staleness** independent of relevance: superseded file reads
   and resolved errors are marked down regardless of topical similarity.
6. **Find the prune cutoff** by the largest natural gap in the score
   distribution within *this* transcript, rather than a fixed global
   threshold — absolute scores aren't comparable across transcripts with
   different vocabularies or topics, but the shape of the distribution
   within one transcript is.

## Benchmark results

Synthetic scenarios with engineered ground truth (precision/recall against
which chunks were *actually* safe to prune, both strategies pruning the
same token budget per scenario — see `benchmarks/scenarios.py` and
`benchmarks/run_benchmark.py`):

| Scenario | Chunks | Tokens pruned | CO F1 | Naive oldest-first F1 |
|---|---|---|---|---|
| stale_reads | 12 | 38 | 1.00 | 0.67 |
| resolved_errors | 9 | 60 | 0.86 | 0.86 |
| noisy_exploration | 12 | 1708 | 1.00 | 1.00 |
| old_but_relevant | 11 | 47 | **1.00** | **0.55** |
| paraphrase | 8 | 24 | **0.67** | **0.00** |

**Mean F1** — context-optimizer: 0.90 | naive oldest-first: 0.61

Reproduce: `python benchmarks/run_benchmark.py`

`paraphrase` is the scenario that actually justifies the semantic layer's
complexity: the relevant work ("lock an account out ... after too many bad
password guesses") and the task query ("prevent repeated failed login
attempts") share zero lexical tokens by construction, plus an adversarial
tangent shares a surface word ("attempts") with the query while being
irrelevant. Naive FIFO pruning has no way to solve this and scores 0.00.
This tool's imperfect-but-real 0.67 — catching the connection despite zero
keyword overlap, without being fooled by the trap word — is the concrete
evidence for the "semantic, not just lexical" claim, not just an assumption
that embeddings help.

## What was tried and rejected

In the interest of not overclaiming "state of the art": a cross-encoder
reranker (`reranker.py`, the standard retrieve-then-rerank pattern from
information retrieval) was built, correctly wired in as a third scoring
signal, and benchmark-swept across weights 0.0-1.0. It **never improved**
this benchmark at any weight, and **regressed** it at full weight
(`stale_reads` F1 dropped 1.00 -> 0.67). `ms-marco-MiniLM-L-6-v2` is
trained on web-search query/passage pairs; Claude Code transcript chunks
(conversational turns, JSON tool-call blobs, raw command output) are
structurally unlike that training distribution. It's shipped **disabled by
default** (`RelevanceScorer(use_reranker=True)` to experiment) rather than
claimed as a win it didn't earn. A different, larger, or code-tuned
cross-encoder might do better — untested.

## Tested against real sessions, not just synthetic scenarios

The 5-scenario benchmark above is hand-built with engineered ground truth,
which is good for catching specific regressions but small. To sanity-check
against reality, the tool was also run directly against real, large,
pre-existing Claude Code session transcripts already on disk (not
committed anywhere — these are private project data) ranging up to 64MB /
1,136 parsed chunks. It parsed and scored them without crashing and
produced sensible output (correctly identified real stale `Bash` output,
real resolved errors, real low-relevance tangents).

It also surfaced a real problem no synthetic scenario could have: the
**first** time the tool ever processes a given session's content, the
embedding cache is cold and every chunk needs fresh inference. On the
1,136-chunk real session, that first pass measured **~103 seconds**
(331 CPU-seconds across threads — confirmed via `cProfile` and repeat
runs that every subsequent pass on the same content took ~0.65s once
cached). A hook that can silently block someone's prompt for 100+ seconds
on their first long session is not acceptable, especially since the hook
is *designed* to only start scoring once a session is already large (see
the cheap pre-check in `hooks/user_prompt_submit.py`) — meaning the worst
case lands exactly on the first real invocation that matters.

Fix: `RelevanceScorer.score()` takes a `timeout_seconds` parameter that
bounds the semantic/rerank signals (the lexical signal is pure numpy and
always fast) using a daemon thread with `.join(timeout)`, not
`concurrent.futures.ThreadPoolExecutor` — the executor's default shutdown
behavior joins all pending work at interpreter exit, which would silently
defeat the timeout by making the process hang anyway. A daemon thread
never blocks process exit, so a timed-out call is genuinely abandoned.
Hooks default to an 8-second bound (`CONTEXT_OPTIMIZER_HOOK_TIMEOUT` env
var to change it) and fall back to lexical-only scoring for that
invocation on timeout; `context-optimizer report` leaves it unbounded by
default, since a user running it directly is asking for the full-quality
result and is fine waiting.

That timeout is a safety net, not a real fix -- it means the first hook
trigger on a large session (exactly when the tool matters most) silently
degrades to lexical-only. So the actual cold-start slowness was profiled
properly (`cProfile` against a real cold cache, not the accidentally-warm
profile from the first pass): **75.6 of 76.9 seconds was pure ONNX
inference itself** (`onnxruntime...run`, 6 calls, ~12.6s each) -- not
caching, not I/O, not JSON serialization, all of which were confirmed
trivial (<0.2s combined). fastembed batches ~256 texts per call and pads
every text in a batch to the length of its longest member. With a handful
of long outlier chunks (real tool dumps near the embedding cap) scattered
essentially at random across only 6 batches, those outliers were dragging
their entire batch up to their own length.

Fix, verified with a direct A/B test on the same real data before
shipping it: sort texts by length before calling `model.embed()`, then
scatter results back to the caller's original order (implemented in
`embeddings.embed_many()`; correctness -- that results map back to the
right text -- has a dedicated regression test in `test_embeddings.py`
using a fake model that encodes each text's own length, so a reordering
bug would be immediately caught). Measured on the real 1,136-chunk
session: **71.4s -> 14.8s, a 4.8x speedup**, byte-identical output.
Combined with also lowering the per-chunk embedding cap from 2000 to 800
characters (a further 16.3s -> ~12s on top of sorting, chosen as a middle
ground rather than exhaustively tuned), the full CLI path on that same
real session went from **103s -> 15.4s** end to end. Still above the
default 8s hook timeout on this particular large session, so the timeout
fallback above remains a real, load-bearing safety net for sessions at or
beyond this size -- not made redundant by this fix, just far less likely
to trigger.

## Known limitations

- Consecutive assistant-only tool calls with no intervening user message
  (e.g. a multi-step autonomous retry loop) don't get relevance propagated
  to them, since propagation currently anchors to the nearest preceding
  user message. This shows up as the one imperfect score in the
  `resolved_errors` benchmark scenario.
- The natural-gap prune cutoff is deliberately conservative: when a truly
  relevant chunk and an irrelevant one score nearly identically (the
  `paraphrase` scenario engineers exactly this via a shared trap word), it
  refuses to draw a cutoff through the near-tie rather than risk a false
  positive. That keeps precision at 1.00 across every benchmark scenario
  (it has never yet wrongly pruned something real) at some cost to recall.
  For a context-management tool, an occasional under-prune is a far
  cheaper mistake than deleting something that mattered, so this tradeoff
  is intentional, not just an artifact.
- 5 hand-built synthetic scenarios is still a small benchmark. It's enough
  to catch real regressions (it already caught two: an equal-weight
  embedding blend, and an over-strength reranker) but not enough to claim
  broad generalization.
- TF-IDF-based relevance is vocabulary-dependent and has no notion of
  synonyms on its own — that gap is what the (dampened) embedding signal
  is meant to cover, and `paraphrase` is the scenario proving it does.
- Token counts are an estimate (`tiktoken`'s `cl100k_base` as a proxy, or a
  char/4 fallback), not Claude's actual tokenizer, since Anthropic doesn't
  expose a free local one.
- Embedding cosine similarity was empirically observed (on `BAAI/bge-small-en-v1.5`)
  to have a high "same-domain" floor for short technical text — unrelated
  programming chatter can still score 0.5+ cosine. Relevance is normalized
  per-transcript (min-max) to compensate, but this is model-specific
  behavior that would need re-validating against a different embedding model.

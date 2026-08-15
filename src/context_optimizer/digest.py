"""Turns scored chunks into a human report, a short model-facing nudge, and
a suggested `/compact` instruction string.

Nothing here calls an LLM. Every output is generated deterministically from
the scores, which keeps this fast enough to run on every prompt submit and
keeps every number in the report traceable back to a concrete cause.
"""
from __future__ import annotations

from dataclasses import dataclass

from .scorer import ScoredChunk

DEFAULT_WINDOW = 200_000
PRUNE_SCORE_THRESHOLD = 0.35
NUDGE_USAGE_THRESHOLD = 0.55  # fraction of window that triggers a hook nudge


@dataclass
class Digest:
    total_tokens: int
    window_size: int
    usage_pct: float
    prune_candidates: list[ScoredChunk]  # sorted worst (lowest score) first
    reclaimable_tokens: int
    top_keep: list[ScoredChunk]  # highest-scoring chunks, for sanity-checking

    def short_nudge(self, max_items: int = 4) -> str:
        """Compact plain-text block meant for hookSpecificOutput.additionalContext.

        Kept short on purpose -- injecting a huge report to solve a context
        problem would be self-defeating.
        """
        if not self.prune_candidates:
            return ""
        lines = [
            f"[context-optimizer] session at {self.usage_pct:.0%} of context "
            f"window ({self.total_tokens:,}/{self.window_size:,} est. tokens). "
            f"Low-relevance-to-current-task content worth summarizing away "
            f"(~{self.reclaimable_tokens:,} tokens reclaimable):"
        ]
        for sc in self.prune_candidates[:max_items]:
            reason = sc.reasons[0] if sc.reasons else "low relevance"
            lines.append(
                f"  - {sc.chunk.label()} (~{sc.chunk.tokens:,} tok, score {sc.score:.2f}): {reason}"
            )
        lines.append(
            "Run `context-optimizer report <transcript>` for the full analysis, "
            "or `/compact` with the suggested instructions it prints."
        )
        return "\n".join(lines)

    def compact_instructions(self, max_items: int = 12) -> str:
        """A ready-to-paste string for `/compact <these instructions>`."""
        superseded_files = sorted(
            {
                sc.chunk.file_path
                for sc in self.prune_candidates
                if sc.chunk.superseded and sc.chunk.file_path
            }
        )
        resolved_errors = list(
            dict.fromkeys(
                sc.chunk.label() for sc in self.prune_candidates if sc.chunk.resolved_error
            )
        )
        low_relevance = list(
            dict.fromkeys(
                sc.chunk.label()
                for sc in self.prune_candidates
                if not sc.chunk.superseded and not sc.chunk.resolved_error
            )
        )

        keep_labels = list(
            dict.fromkeys(sc.chunk.label() for sc in self.top_keep[:max_items])
        )

        parts = [
            "Preserve in full: the current task/goal, all code changes actually made, "
            "unresolved errors, and decisions/constraints stated by the user.",
        ]
        if keep_labels:
            parts.append("Especially keep detail from: " + "; ".join(keep_labels) + ".")
        if superseded_files:
            parts.append(
                "Safe to drop entirely (file contents were read but later overwritten, "
                "so the read is stale): " + ", ".join(superseded_files[:max_items]) + "."
            )
        if resolved_errors:
            parts.append(
                "Safe to compress to one line each (errors that were later fixed): "
                + ", ".join(resolved_errors[:max_items]) + "."
            )
        if low_relevance:
            parts.append(
                "Safe to summarize heavily (low relevance to the current task): "
                + ", ".join(low_relevance[:max_items]) + "."
            )
        return " ".join(parts)


def _natural_prune_cutoff(scored: list[ScoredChunk], ceiling: float) -> set:
    """Pick which below-ceiling chunks to prune by finding the largest gap
    in their sorted scores, instead of a single global magic threshold.

    A fixed cutoff (e.g. "prune everything under 0.35") looks reasonable in
    isolation but breaks across transcripts with different vocabulary sizes,
    where absolute TF-IDF-style scores aren't comparable: in one session
    "clearly still relevant" might score 0.15, in another "clearly noise"
    might score 0.30. What's stable is the SHAPE of the distribution within
    a single transcript -- there is usually a real cluster of low-relevance
    chunks separated by a gap from the borderline-but-still-relevant ones.
    We cut at that gap. If the distribution is too uniform to have a real
    gap, we fall back to everything under the ceiling (the old behavior).
    """
    below = sorted((sc for sc in scored if sc.score < ceiling), key=lambda sc: sc.score)
    if len(below) < 2:
        return {sc.chunk.index for sc in below}

    scores = [sc.score for sc in below]
    span = scores[-1] - scores[0]
    if span <= 1e-9:
        return {sc.chunk.index for sc in below}  # all tied, nothing to split on

    gaps = [(scores[i + 1] - scores[i], i) for i in range(len(scores) - 1)]
    best_gap, best_i = max(gaps, key=lambda g: g[0])

    # Only treat it as a real cluster boundary if the gap is a meaningfully
    # large chunk of the observed range -- otherwise noise in near-uniform
    # scores could carve off an arbitrary single chunk.
    if best_gap < 0.25 * span:
        return {sc.chunk.index for sc in below}

    return {sc.chunk.index for sc in below[: best_i + 1]}


def build_digest(
    scored: list[ScoredChunk],
    window_size: int = DEFAULT_WINDOW,
    prune_threshold: float = PRUNE_SCORE_THRESHOLD,
    top_keep_n: int = 8,
) -> Digest:
    total_tokens = sum(sc.chunk.tokens for sc in scored)
    usage_pct = total_tokens / window_size if window_size else 0.0

    cutoff_indices = _natural_prune_cutoff(scored, prune_threshold)
    # Rank the *display* by token impact, not score: every candidate here
    # already cleared the relevance bar for "safe to summarize away", so the
    # useful ordering is which ones actually free up the most space.
    prune_candidates = sorted(
        (sc for sc in scored if sc.chunk.index in cutoff_indices),
        key=lambda sc: -sc.chunk.tokens,
    )
    reclaimable = sum(sc.chunk.tokens for sc in prune_candidates)

    top_keep = sorted(scored, key=lambda sc: sc.score, reverse=True)[:top_keep_n]

    return Digest(
        total_tokens=total_tokens,
        window_size=window_size,
        usage_pct=usage_pct,
        prune_candidates=prune_candidates,
        reclaimable_tokens=reclaimable,
        top_keep=top_keep,
    )

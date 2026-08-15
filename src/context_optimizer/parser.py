"""Parses a Claude Code session transcript (JSONL) into scorable Chunks.

Claude Code's on-disk transcript schema is undocumented and has shifted
across versions, so every field access here is defensive: unknown or
missing fields degrade to sensible defaults instead of raising, and a
single malformed line never aborts the whole parse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .tokenizer import count_tokens

# Tools whose `input` commonly names a file path, in priority order.
_FILE_PATH_KEYS = ("file_path", "path", "notebook_path")


@dataclass
class Chunk:
    index: int
    uuid: str
    kind: str  # user_message | assistant_message | thinking | tool_call | tool_result
    role: str  # user | assistant | tool
    text: str  # extracted text used for relevance scoring
    tokens: int
    timestamp: Optional[str] = None
    tool_name: Optional[str] = None
    tool_use_id: Optional[str] = None
    file_path: Optional[str] = None
    is_error: bool = False
    superseded: bool = False  # set by mark_superseded_reads()
    resolved_error: bool = False  # set by mark_resolved_errors()
    raw_char_len: int = 0

    def label(self) -> str:
        if self.tool_name and self.file_path:
            return f"{self.tool_name}({self.file_path})"
        if self.tool_name:
            return self.tool_name
        return self.kind


def _extract_file_path(tool_input: dict) -> Optional[str]:
    for key in _FILE_PATH_KEYS:
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _iter_content_blocks(message: dict):
    content = message.get("content")
    if isinstance(content, str):
        yield {"type": "text", "text": content}
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block


def parse_transcript(path: str | Path) -> list[Chunk]:
    """Parse a Claude Code .jsonl transcript into an ordered list of Chunks."""
    path = Path(path)
    chunks: list[Chunk] = []
    if not path.exists():
        return chunks

    # tool_use_id -> tool_name/file_path, so tool_result chunks inherit context
    # from the tool_call that produced them.
    pending_tool_calls: dict[str, tuple[str, Optional[str]]] = {}

    with path.open("r", errors="replace") as f:
        for line_no, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            rtype = record.get("type")
            if rtype not in ("user", "assistant"):
                continue  # skip summary/system/meta lines

            message = record.get("message")
            if not isinstance(message, dict):
                continue

            role = message.get("role", rtype)
            timestamp = record.get("timestamp")
            uuid = record.get("uuid", f"line-{line_no}")

            for block in _iter_content_blocks(message):
                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "")
                    if not text.strip():
                        continue
                    kind = "user_message" if role == "user" else "assistant_message"
                    chunks.append(
                        Chunk(
                            index=len(chunks),
                            uuid=uuid,
                            kind=kind,
                            role=role,
                            text=text,
                            tokens=count_tokens(text),
                            timestamp=timestamp,
                            raw_char_len=len(text),
                        )
                    )

                elif btype == "thinking":
                    text = block.get("thinking", "")
                    if not text.strip():
                        continue
                    chunks.append(
                        Chunk(
                            index=len(chunks),
                            uuid=uuid,
                            kind="thinking",
                            role=role,
                            text=text,
                            tokens=count_tokens(text),
                            timestamp=timestamp,
                            raw_char_len=len(text),
                        )
                    )

                elif btype == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    tool_input = block.get("input") or {}
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    file_path = _extract_file_path(tool_input)
                    tool_use_id = block.get("id")
                    summary_text = f"{tool_name} {json.dumps(tool_input)[:2000]}"
                    if tool_use_id:
                        pending_tool_calls[tool_use_id] = (tool_name, file_path)
                    chunks.append(
                        Chunk(
                            index=len(chunks),
                            uuid=uuid,
                            kind="tool_call",
                            role=role,
                            text=summary_text,
                            tokens=count_tokens(summary_text),
                            timestamp=timestamp,
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            file_path=file_path,
                            raw_char_len=len(summary_text),
                        )
                    )

                elif btype == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    tool_name, file_path = pending_tool_calls.get(
                        tool_use_id, (None, None)
                    )
                    text = _stringify_tool_result_content(block.get("content"))
                    is_error = bool(block.get("is_error", False))
                    chunks.append(
                        Chunk(
                            index=len(chunks),
                            uuid=uuid,
                            kind="tool_result",
                            role="tool",
                            text=text,
                            tokens=count_tokens(text),
                            timestamp=timestamp,
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            file_path=file_path,
                            is_error=is_error,
                            raw_char_len=len(text),
                        )
                    )

    mark_superseded_reads(chunks)
    mark_resolved_errors(chunks)
    return chunks


_READ_TOOLS = {"Read", "Grep", "Glob", "NotebookEdit"}
_WRITE_TOOLS = {"Edit", "Write", "NotebookEdit"}


def mark_superseded_reads(chunks: list[Chunk]) -> None:
    """Flag tool_result chunks for file reads that a later Edit/Write made stale.

    A Read on path X followed later by an Edit/Write on the same X means the
    read's cached content no longer reflects reality -- it's a strong, cheap
    pruning signal independent of semantic relevance.
    """
    last_write_index: dict[str, int] = {}
    for c in chunks:
        if c.kind == "tool_call" and c.tool_name in _WRITE_TOOLS and c.file_path:
            last_write_index[c.file_path] = c.index

    for c in chunks:
        if (
            c.kind == "tool_result"
            and c.tool_name in _READ_TOOLS
            and c.file_path
            and c.file_path in last_write_index
        ):
            # Superseded only if the write happened AFTER this read's own tool_call
            # (tool_result index is always right after its tool_call, so compare
            # against this result's index).
            if last_write_index[c.file_path] > c.index:
                c.superseded = True


def mark_resolved_errors(chunks: list[Chunk]) -> None:
    """Flag failed tool_result chunks that a later successful call superseded.

    Heuristic: an is_error tool_result on (tool_name, file_path) is resolved
    if a later tool_result for the same (tool_name, file_path) succeeded.
    Tools with no file_path are matched on tool_name alone, which is coarser
    but still useful (e.g. a failing Bash command retried and fixed).
    """
    last_success_index: dict[tuple[str, Optional[str]], int] = {}
    for c in chunks:
        if c.kind == "tool_result" and not c.is_error and c.tool_name:
            last_success_index[(c.tool_name, c.file_path)] = c.index

    for c in chunks:
        if c.kind == "tool_result" and c.is_error and c.tool_name:
            key = (c.tool_name, c.file_path)
            later_success = last_success_index.get(key)
            if later_success is not None and later_success > c.index:
                c.resolved_error = True

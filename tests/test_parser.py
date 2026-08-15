import json
import tempfile
from pathlib import Path

from context_optimizer.parser import parse_transcript


def _write_transcript(records: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def test_parses_text_messages():
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "timestamp": "t1",
            "message": {"role": "user", "content": "fix the login bug"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "timestamp": "t2",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Looking into it."}],
            },
        },
    ]
    path = _write_transcript(records)
    chunks = parse_transcript(path)
    assert len(chunks) == 2
    assert chunks[0].kind == "user_message"
    assert chunks[0].text == "fix the login bug"
    assert chunks[1].kind == "assistant_message"


def test_marks_superseded_file_reads():
    records = [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "read auth.py"}},
        {
            "type": "assistant",
            "uuid": "a1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "auth.py"}}
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "def login(): pass"}
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Edit", "input": {"file_path": "auth.py"}}
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u3",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "edited"}],
            },
        },
    ]
    path = _write_transcript(records)
    chunks = parse_transcript(path)
    read_result = next(c for c in chunks if c.kind == "tool_result" and c.tool_name == "Read")
    assert read_result.superseded is True


def test_marks_resolved_errors():
    records = [
        {
            "type": "assistant",
            "uuid": "a1",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "error: bad syntax", "is_error": True}
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
            },
        },
    ]
    path = _write_transcript(records)
    chunks = parse_transcript(path)
    failed = next(c for c in chunks if c.kind == "tool_result" and c.is_error)
    assert failed.resolved_error is True


def test_ignores_malformed_lines():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write("not json\n")
    f.write(json.dumps({"type": "user", "uuid": "u1", "message": {"role": "user", "content": "hi"}}) + "\n")
    f.close()
    chunks = parse_transcript(f.name)
    assert len(chunks) == 1

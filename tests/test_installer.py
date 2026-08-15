import json

from context_optimizer import installer


def test_merge_hooks_into_empty_settings():
    merged = installer.merge_hooks({})
    assert "UserPromptSubmit" in merged["hooks"]
    assert "PreCompact" in merged["hooks"]


def test_merge_hooks_preserves_existing_unrelated_hooks():
    existing = {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "some-other-tool"}]}],
            "SessionStart": [{"hooks": [{"type": "command", "command": "totally-unrelated"}]}],
        },
        "otherSetting": "keep-me",
    }
    merged = installer.merge_hooks(existing)

    commands = [h["command"] for entry in merged["hooks"]["UserPromptSubmit"] for h in entry["hooks"]]
    assert "some-other-tool" in commands
    assert "context-optimizer hook-user-prompt-submit" in commands
    assert merged["hooks"]["SessionStart"] == existing["hooks"]["SessionStart"]
    assert merged["otherSetting"] == "keep-me"


def test_merge_hooks_is_idempotent():
    once = installer.merge_hooks({})
    twice = installer.merge_hooks(once)
    # Re-installing must not duplicate our own entries.
    commands = [h["command"] for entry in twice["hooks"]["UserPromptSubmit"] for h in entry["hooks"]]
    assert commands.count("context-optimizer hook-user-prompt-submit") == 1


def test_remove_hooks_strips_only_our_entries():
    existing = installer.merge_hooks(
        {
            "hooks": {
                "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "some-other-tool"}]}],
            }
        }
    )
    cleaned = installer.remove_hooks(existing)
    commands = [h["command"] for entry in cleaned["hooks"]["UserPromptSubmit"] for h in entry["hooks"]]
    assert "some-other-tool" in commands
    assert "context-optimizer hook-user-prompt-submit" not in commands
    assert "PreCompact" not in cleaned["hooks"]  # nothing else was in there, so the key is dropped


def test_remove_hooks_drops_hooks_key_entirely_when_empty():
    installed = installer.merge_hooks({})
    cleaned = installer.remove_hooks(installed)
    assert "hooks" not in cleaned


def test_install_creates_backup_on_reinstall(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "settings_path", lambda project=False: tmp_path / "settings.json")
    path1 = installer.install()
    assert path1.exists()
    assert not path1.with_suffix(".json.bak").exists()  # nothing to back up on first install

    path2 = installer.install()  # reinstall
    assert path2.with_suffix(".json.bak").exists()

    data = json.loads(path2.read_text())
    assert "UserPromptSubmit" in data["hooks"]


def test_uninstall_after_install_removes_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "settings_path", lambda project=False: tmp_path / "settings.json")
    installer.install()
    path = installer.uninstall()
    data = json.loads(path.read_text())
    assert "hooks" not in data

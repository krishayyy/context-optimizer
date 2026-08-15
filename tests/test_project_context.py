import subprocess
import tempfile

from context_optimizer import project_context


def _init_repo() -> str:
    d = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    return d


def test_gather_detects_modified_files():
    d = _init_repo()
    with open(f"{d}/foo.py", "w") as f:
        f.write("x = 1\n")
    subprocess.run(["git", "add", "foo.py"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True)

    with open(f"{d}/foo.py", "w") as f:
        f.write("x = 2\n")

    ctx = project_context.gather(d)
    assert "foo.py" in ctx.modified_files


def test_gather_detects_branch_name():
    d = _init_repo()
    with open(f"{d}/a.txt", "w") as f:
        f.write("a")
    subprocess.run(["git", "add", "a.txt"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "fix/rate-limit-419"], cwd=d, check=True)

    ctx = project_context.gather(d)
    assert ctx.branch_name == "fix/rate-limit-419"
    assert "rate" in ctx.branch_tokens_text()
    assert "limit" in ctx.branch_tokens_text()


def test_gather_nonexistent_dir_returns_empty_context():
    ctx = project_context.gather("/definitely/not/a/real/path/xyz")
    assert ctx.modified_files == set()
    assert ctx.branch_name == ""


def test_gather_empty_cwd_returns_empty_context():
    ctx = project_context.gather("")
    assert ctx.modified_files == set()

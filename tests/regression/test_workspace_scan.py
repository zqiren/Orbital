import os

from agent_os.agent.workspace_scan import scan_workspace


def _touch(path: str, size: int = 0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x" * size)


def test_lists_files_with_sizes(tmp_path):
    _touch(str(tmp_path / "README.md"), 100)
    _touch(str(tmp_path / "src" / "main.py"), 250)
    out = scan_workspace(str(tmp_path))
    assert "README.md" in out
    assert "src/main.py" in out or os.path.join("src", "main.py") in out
    assert "100" in out and "250" in out


def test_respects_gitignore(tmp_path):
    with open(tmp_path / ".gitignore", "w") as f:
        f.write("ignored/\n*.log\n")
    _touch(str(tmp_path / "keep.py"), 10)
    _touch(str(tmp_path / "ignored" / "secret.py"), 10)
    _touch(str(tmp_path / "debug.log"), 10)
    out = scan_workspace(str(tmp_path))
    assert "keep.py" in out
    assert "secret.py" not in out
    assert "debug.log" not in out


def test_self_bounds_large_tree(tmp_path):
    for i in range(50):
        _touch(str(tmp_path / f"f{i}.txt"), i)
    out = scan_workspace(str(tmp_path), max_files=10)
    # Truncation notice present; output does not list all 50 files.
    assert "more" in out.lower() or "truncat" in out.lower()
    assert out.count("\n") < 50


def test_empty_workspace_returns_notice(tmp_path):
    out = scan_workspace(str(tmp_path))
    assert out.strip() != ""
    assert "no files" in out.lower() or "empty" in out.lower()

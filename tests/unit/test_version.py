"""Runtime version resolution (spec 046 §7).

The app must know its own version at runtime. Resolution order:
  1. agent_os/_version.py — generated at build time by the build scripts
     (authoritative in packaged installs).
  2. pyproject.toml [project] version — dev-checkout fallback.
  3. importlib.metadata — last resort (may be stale egg-info; better than nothing).
  4. "0.0.0" — never raises.
"""

from pathlib import Path

import pytest

from agent_os import version as version_mod


def test_get_version_matches_pyproject_in_dev_checkout():
    # In this checkout no _version.py exists, so the pyproject fallback is the
    # authoritative path — and it must NOT return the stale egg-info value.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    expected = None
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version"):
            expected = line.split("=", 1)[1].strip().strip('"')
            break
    assert expected, "pyproject.toml must carry a [project] version"
    assert version_mod.get_version() == expected


def test_generated_version_module_wins(monkeypatch):
    monkeypatch.setattr(version_mod, "_read_generated", lambda: "9.9.9")
    assert version_mod.get_version() == "9.9.9"


def test_pyproject_parse_from_tmp(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    assert version_mod._read_pyproject(tmp_path) == "1.2.3"


def test_missing_everything_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(version_mod, "_read_generated", lambda: None)
    monkeypatch.setattr(version_mod, "_read_pyproject", lambda root=None: None)
    monkeypatch.setattr(version_mod, "_read_metadata", lambda: None)
    assert version_mod.get_version() == "0.0.0"


def test_never_raises_on_broken_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("not [ valid toml", encoding="utf-8")
    assert version_mod._read_pyproject(tmp_path) is None

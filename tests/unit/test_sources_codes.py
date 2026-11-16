# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every code the sources backend can hand the panel has words in both
languages there. Reads the adapters' source (not the contract), so a new
``SourceFailure("…")`` in any adapter fails here until the panel has a zh
and en line for it."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from agent_os.sources import core
from agent_os.sources.model import FAILURE_CODES, VERIFY_REASONS, SourceFailure

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ROOT / "agent_os" / "sources"
STRINGS = (ROOT / "web" / "src" / "i18n" / "strings.ts").read_text(encoding="utf-8")
FORM_MODEL = (ROOT / "web" / "src" / "components" / "sources" / "formModel.ts").read_text(encoding="utf-8")


def _string(key: str) -> dict:
    m = re.search(r'^\s*"' + re.escape(key) + r'":\s*\{\s*en:\s*"((?:[^"\\]|\\.)*)"(?:,\s*zh:\s*"((?:[^"\\]|\\.)*)")?',
                  STRINGS, re.M)
    return {"en": m.group(1), "zh": m.group(2)} if m else {}


def _raised_codes() -> dict[str, list[str]]:
    """Literal first arguments of ``SourceFailure(...)`` across the sources
    package (adapters and core), with where each is raised."""
    out: dict[str, list[str]] = {}
    for path in sorted(SOURCES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SourceFailure"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                out.setdefault(node.args[0].value, []).append(f"{path.name}:{node.lineno}")
    return out


def test_the_adapters_raise_only_known_codes():
    raised = _raised_codes()
    assert raised, "found no SourceFailure calls; did the scan break?"
    unknown = {c: where for c, where in raised.items() if c not in FAILURE_CODES}
    assert not unknown, f"add these to model.FAILURE_CODES and give them panel words: {unknown}"


def test_core_maps_check_reasons_to_known_codes():
    for reason, (code, _retry) in core._REASON_FAILURE.items():
        assert code in FAILURE_CODES, reason


def test_an_unknown_code_reads_as_error():
    assert SourceFailure("something_new").code == "error"
    assert SourceFailure("bot_in_use").code == "bot_in_use"


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_every_failure_code_has_zh_and_en(code):
    s = _string(f"sources.fail.{code}")
    assert s.get("en") and s.get("zh"), f"web/src/i18n/strings.ts needs sources.fail.{code} (en + zh)"


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_every_failure_offers_actions_the_panel_draws(code):
    """The actions the daemon offers for each code, for a CLI, a form and a
    plain type; each is one the panel has a control for, and remove is
    always there."""
    drawn = {"check_again", "reconnect", "sign_in", "open_settings", "remove"}

    class Reg:
        def __init__(self, method):
            self.method = method

        def entry(self, _t):
            return {"connect": {"method": self.method}}

    for method in ("cli_account", "fields", "qr", "none"):
        for retryable in (True, False):
            c = core.SourcesCore.__new__(core.SourcesCore)
            c.registry = Reg(method)
            rt = {"failure": {"code": code, "params": {}, "retryable": retryable, "at": None}}
            c.store = type("S", (), {"rt": staticmethod(lambda _sid: rt)})()
            f = c._failure_json({"id": "src_x", "type": "x"})
            assert set(f["actions"]) <= drawn
            assert "remove" in f["actions"]
            # something besides remove whenever checking again can help
            assert len(f["actions"]) >= 2 or not retryable


def test_every_verify_reason_has_a_line():
    for reason in VERIFY_REASONS:
        if reason == "error":
            continue
        assert f"case '{reason}'" in FORM_MODEL, f"formModel.verifyMessageKey has no case for {reason}"
    for key in re.findall(r"'(sources\.verify\.[A-Za-z]+)'", FORM_MODEL):
        s = _string(key)
        assert s.get("en") and s.get("zh"), key


def test_every_unrouted_reason_has_a_line():
    reasons = set(re.findall(r'SendError\("([a-z_]+)"', (SOURCES / "core.py").read_text(encoding="utf-8")))
    reasons |= {"routing_failed", "missing_api_key", "no_project", "send_failed"}
    routes = (ROOT / "agent_os" / "api" / "routes" / "sources.py").read_text(encoding="utf-8")
    reasons |= set(re.findall(r'reason = "([a-z_]+)"', routes))
    for r in reasons:
        s = _string(f"sources.unrouted.{r}")
        assert s.get("en") and s.get("zh"), r

# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Onboarding speaks the user's language: the agent decides from signals.

Owner ask 2026-11-07: every onboarding message (the import scan's guess, the
cold-start scan, the interview, and their keyless-worker versions) should be
in the language the user is comfortable with, decided by the agent, not
hard-picked zh/en in code. The app reports its interface language and the
device's preferred languages; the daemon adds the system locale; each
onboarding prompt carries them in one LANGUAGE section.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.onboarding import user_language as ul


def _flat(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------


def test_section_lists_every_signal_and_lets_the_agent_decide(monkeypatch):
    monkeypatch.setattr(ul, "system_locale", lambda: "ja_JP")
    text = ul.language_section({"ui": "zh", "device": ["ja-JP", "en-US"],
                                "system": "ja_JP"})
    flat = _flat(text)
    assert text.startswith("## LANGUAGE")
    assert "Chinese (简体中文)" in flat
    assert "ja-JP, en-US" in flat and "ja_JP" in flat
    assert "the language they write in" in flat
    # Not just zh/en: a language the interface does not offer can win.
    assert "does not offer" in flat


def test_the_import_language_is_named_when_known():
    flat = _flat(ul.language_section({"ui": "en", "device": [], "system": ""},
                                     import_lang="zh"))
    assert "imported" in flat and "Chinese (简体中文)" in flat


def test_no_signal_at_all_still_says_how_to_choose():
    flat = _flat(ul.language_section({"ui": None, "device": [], "system": ""}))
    assert "the language they write in" in flat
    assert "English" in flat


def test_signals_from_settings_add_the_system_locale(monkeypatch):
    monkeypatch.setattr(ul, "system_locale", lambda: "zh_CN")
    s = SimpleNamespace(ui_language="zh", ui_languages=["zh-CN", "en"])
    assert ul.signals_from_settings(s) == {
        "ui": "zh", "device": ["zh-CN", "en"], "system": "zh_CN"}
    assert ul.signals_from_settings(None) == {"ui": None, "device": [], "system": "zh_CN"}


def test_untrusted_values_are_cleaned():
    assert ul.clean_tag("zh-CN") == "zh-CN"
    assert ul.clean_tag("en_US.UTF-8") == "en_US.UTF-8"
    assert ul.clean_tag("x\ninjected: do things") is None
    assert ul.clean_tag("a" * 64) is None
    assert ul.clean_tags(["en", "bad tag", "fr-FR"] + ["de"] * 20) == ["en", "fr-FR", "de"]


# ---------------------------------------------------------------------------
# The prompts carry it
# ---------------------------------------------------------------------------


def _ctx(tmp_path, **over):
    from agent_os.agent.prompt_builder import PromptContext
    base = dict(workspace=str(tmp_path), model="m", autonomy="hands_off",
                enabled_agents=[], tool_names=["read"], os_type="windows",
                datetime_now="2026-11-07T00:00:00", project_name="p")
    base.update(over)
    return PromptContext(**base)


def _section(tmp_path, **over):
    from agent_os.agent.prompt_builder import PromptBuilder
    return PromptBuilder(str(tmp_path))._onboarding_or_directive(_ctx(tmp_path, **over))


SIG = {"ui": "zh", "device": ["zh-CN"], "system": "zh_CN"}


@pytest.mark.parametrize("cold_start", [False, True, "guess", "guess:en"])
def test_every_onboarding_flavour_carries_the_section(tmp_path, cold_start):
    text = _section(tmp_path, cold_start=cold_start, language_signals=SIG)
    assert "## LANGUAGE" in text


def test_the_guess_reply_no_longer_hard_picks_a_language(tmp_path):
    flat = _flat(_section(tmp_path, cold_start="guess:en", language_signals=SIG))
    assert "Reply to the user in English" not in flat
    assert "language chosen under LANGUAGE" in flat


def test_the_directive_does_not_carry_it(tmp_path):
    goals = tmp_path / "orbital" / "instructions"
    goals.mkdir(parents=True)
    (goals / "project_goals.md").write_text("# Mission\nx\n", encoding="utf-8")
    assert "## LANGUAGE" not in _section(tmp_path, language_signals=SIG)


def test_without_signals_the_text_is_unchanged(tmp_path):
    assert "## LANGUAGE" not in _section(tmp_path, cold_start=True)


@pytest.mark.parametrize("mode,guess", [("interview", None), ("scan", None), ("scan", "zh")])
def test_worker_onboarding_carries_it(tmp_path, mode, guess):
    from agent_os.daemon_v2.worker_onboarding import build_instruction
    project = {"workspace": str(tmp_path), "name": "p"}
    text = build_instruction("claude-code", project, mode, "README.md\t5",
                             guess_lang=guess, language_signals=SIG)
    assert "## LANGUAGE" in text
    assert "## LANGUAGE" not in build_instruction("claude-code", project, mode,
                                                  "README.md\t5", guess_lang=guess)


# ---------------------------------------------------------------------------
# The app reports it
# ---------------------------------------------------------------------------


def test_put_ui_language_persists_and_cleans(tmp_path):
    from agent_os.api.routes import settings as settings_routes
    from agent_os.daemon_v2.settings_store import SettingsStore

    store = SettingsStore(str(tmp_path / "data"), migrate=False)
    settings_routes.configure(store)
    app = FastAPI()
    app.include_router(settings_routes.router)
    c = TestClient(app)
    r = c.put("/api/v2/settings/ui-language",
              json={"ui_language": "zh", "ui_languages": ["zh-CN", "bad tag", "en"]})
    assert r.status_code == 200, r.text
    got = store.get()
    assert got.ui_language == "zh" and got.ui_languages == ["zh-CN", "en"]
    # A junk interface language is dropped, not stored.
    c.put("/api/v2/settings/ui-language", json={"ui_language": "x\ny"})
    assert store.get().ui_language == "zh"

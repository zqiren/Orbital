# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Locale-based search-engine auto-pick for the browser tool.

In mainland China, www.google.com is unreachable without a VPN, so the
browser tool's `search` action must auto-pick a China-accessible engine
(Bing China) when the detected browser locale is mainland Chinese, while
leaving Google as the default for everyone else (no behavior change).
"""

import pytest

from agent_os.agent.tools.browser import _pick_search_engine, _is_china_locale


@pytest.mark.parametrize(
    "locale", ["zh-CN", "zh_CN", "zh-Hans", "zh-Hans-CN", "zh", "ZH-CN"]
)
def test_mainland_china_locales_pick_bing_cn(locale):
    eng = _pick_search_engine(locale)
    assert eng.name == "bing-cn"
    # www.bing.com (not cn.bing.com): renders results when tested from outside
    # China, and inside China www.bing.com redirects to cn.bing.com/search —
    # so it works in both. setmkt forces the China market.
    assert "bing.com" in eng.url_template
    assert "setmkt=zh-CN" in eng.url_template
    assert eng.extract_js  # must carry an extractor for its own DOM
    # Bing hydrates results via JS slightly after networkidle; the search flow
    # waits for this selector so extraction doesn't fire on an empty container.
    assert eng.results_selector == "li.b_algo"


@pytest.mark.parametrize(
    "locale", ["en-US", "zh-TW", "zh-HK", "ja-JP", "fr-FR", "", None]
)
def test_non_mainland_locales_pick_google(locale):
    eng = _pick_search_engine(locale)
    assert eng.name == "google"
    assert "google.com" in eng.url_template
    assert eng.extract_js
    # Google path keeps its existing flow unchanged: no pre-extraction wait.
    assert eng.results_selector is None


def test_is_china_locale_matches_only_mainland():
    for yes in ("zh-CN", "zh_CN", "zh-Hans-CN", "zh-Hans", "zh", "ZH-cn"):
        assert _is_china_locale(yes), yes
    for no in ("zh-TW", "zh-HK", "en-US", "ja-JP", "", None):
        assert not _is_china_locale(no), no


def test_url_template_builds_a_valid_https_query():
    for locale in ("zh-CN", "en-US"):
        eng = _pick_search_engine(locale)
        built = eng.url_template.format(q="hello%20world")
        assert built.startswith("https://")
        assert "hello%20world" in built


def test_unknown_locale_defaults_to_google_safely():
    # A garbage locale must never strand a non-China user on a China engine.
    eng = _pick_search_engine("xx-YY-garbage")
    assert eng.name == "google"

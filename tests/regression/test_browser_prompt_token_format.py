# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the browser prompt must teach a RESOLVABLE token format.

DIAGNOSIS-request-credential-login-failures.md: every browser prompt variant
taught a flat token format (`<secret:gmail_password>`) that the central
resolver categorically rejects — it requires `name.field`
(registry.py `_make_resolver`: no dot → None → ValueError). Agents that
followed the prompt verbatim could never type a credential.

These tests render every browser prompt variant and assert:
  1. a dotted `<secret:name.field>` example is present;
  2. NO flat `<secret:word>` token (a key without a dot) appears anywhere;
  3. the prompt teaches "use exactly the tokens returned by
     request_credential" rather than constructing token names.
"""

from __future__ import annotations

import re

import pytest

from agent_os.agent.prompt_builder import (
    _BROWSER_USAGE_PROMPT,
    _BROWSER_USAGE_PROMPT_TEXT_ONLY,
    _BROWSER_USAGE_PROMPT_VISION,
    Autonomy,
    PromptBuilder,
    PromptContext,
)

_DOTTED_EXAMPLE = re.compile(r"<secret:[A-Za-z0-9_]+\.[A-Za-z0-9_]+>")
_ANY_TOKEN = re.compile(r"<secret:([^>]+)>")

_VARIANTS = {
    "plain": _BROWSER_USAGE_PROMPT,
    "vision": _BROWSER_USAGE_PROMPT_VISION,
    "text_only": _BROWSER_USAGE_PROMPT_TEXT_ONLY,
}


def _ctx(vision: bool) -> PromptContext:
    return PromptContext(
        workspace="/ws", model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["browser"], os_type="linux",
        datetime_now="2026-06-12T00:00:00", vision_enabled=vision,
    )


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_prompt_contains_dotted_token_example(variant):
    assert _DOTTED_EXAMPLE.search(_VARIANTS[variant]), (
        f"browser prompt variant '{variant}' has no dotted "
        "<secret:name.field> example"
    )


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_prompt_contains_no_flat_token(variant):
    for m in _ANY_TOKEN.finditer(_VARIANTS[variant]):
        assert "." in m.group(1), (
            f"browser prompt variant '{variant}' teaches the flat token "
            f"<secret:{m.group(1)}> which the resolver rejects (needs name.field)"
        )


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_prompt_teaches_tokens_come_from_request_credential(variant):
    assert "request_credential" in _VARIANTS[variant], (
        f"browser prompt variant '{variant}' must tell the agent to use "
        "exactly the tokens returned by request_credential"
    )


@pytest.mark.parametrize("vision", [True, False])
def test_rendered_browser_section_uses_resolvable_tokens(vision):
    """Belt and braces: assert on the section the builder actually renders."""
    section = PromptBuilder()._browser_section(_ctx(vision))
    assert section is not None
    assert _DOTTED_EXAMPLE.search(section)
    for m in _ANY_TOKEN.finditer(section):
        assert "." in m.group(1)

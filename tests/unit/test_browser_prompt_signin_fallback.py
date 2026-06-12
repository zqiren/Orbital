# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every browser prompt variant must teach the manual sign-in fallback.

TASK-browser-signin-fallback.md: when a sign-in cannot be completed (CAPTCHA,
bot detection, passkey/hardware-key requirement, repeated login failure) the
agent must not retry indefinitely — it must tell the user plainly that the
site blocks automated sign-in and point them to Settings → Browser Sign-In.
"""

import pytest

from agent_os.agent.prompt_builder import (
    _BROWSER_USAGE_PROMPT,
    _BROWSER_USAGE_PROMPT_TEXT_ONLY,
    _BROWSER_USAGE_PROMPT_VISION,
    Autonomy,
    PromptBuilder,
    PromptContext,
)

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
def test_variant_contains_signin_fallback_guidance(variant):
    text = _VARIANTS[variant]
    assert "Settings → Browser Sign-In" in text, (
        f"browser prompt variant '{variant}' must point the user to "
        "Settings → Browser Sign-In when automated sign-in fails"
    )
    assert "CAPTCHA" in text, (
        f"browser prompt variant '{variant}' must name CAPTCHA as a "
        "blocked-sign-in trigger"
    )
    assert "passkey" in text.lower(), (
        f"browser prompt variant '{variant}' must name passkeys as a "
        "blocked-sign-in trigger"
    )
    assert "blocks automated sign-in" in text, (
        f"browser prompt variant '{variant}' must tell the agent to say "
        "plainly that the site blocks automated sign-in"
    )
    assert "do not keep retrying" in text.lower(), (
        f"browser prompt variant '{variant}' must forbid indefinite retries"
    )


@pytest.mark.parametrize("vision", [True, False])
def test_rendered_browser_section_contains_signin_fallback(vision):
    """Assert on the section the builder actually renders."""
    section = PromptBuilder()._browser_section(_ctx(vision))
    assert section is not None
    assert "Settings → Browser Sign-In" in section
    assert "blocks automated sign-in" in section

# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a custom/self-hosted Anthropic-format router (e.g. co.yes.vg)
must actually be reached when sdk="anthropic".

The bug: LLMProvider built the Anthropic SDK client as
``anthropic.AsyncAnthropic(api_key=...)`` with NO ``base_url``, even though the
constructor received one and stored it on ``self.base_url``. The OpenAI branch
one line below threaded ``base_url`` correctly; the Anthropic branch dropped it.

Effect: every request went to the default ``https://api.anthropic.com`` carrying
the *router's* API key. api.anthropic.com rejected that key with 401, surfaced
to the user as "Invalid API key" — even though the key was valid for the router.

These tests pin that the configured base_url reaches the underlying client on
both construction and the hot-swap (update_api_key) path.
"""

from agent_os.agent.providers.openai_compat import LLMProvider

ROUTER_URL = "https://co.yes.vg"


def test_anthropic_provider_honors_custom_base_url():
    """sdk="anthropic" + a custom base_url must configure the SDK client to hit
    the router, not the default api.anthropic.com."""
    provider = LLMProvider(
        model="claude-sonnet-5",
        api_key="router-key",
        base_url=ROUTER_URL,
        sdk="anthropic",
    )
    assert str(provider._anthropic_client.base_url).rstrip("/") == ROUTER_URL


def test_anthropic_provider_default_base_url_unchanged():
    """No base_url given → the SDK keeps its default endpoint (control case)."""
    provider = LLMProvider(
        model="claude-sonnet-5",
        api_key="k",
        base_url=None,
        sdk="anthropic",
    )
    assert "api.anthropic.com" in str(provider._anthropic_client.base_url)


def test_anthropic_update_api_key_preserves_base_url():
    """Hot-swapping the API key must not silently reset the router URL back to
    api.anthropic.com (the update_api_key path had the same dropped-base_url bug)."""
    provider = LLMProvider(
        model="claude-sonnet-5",
        api_key="old-key",
        base_url=ROUTER_URL,
        sdk="anthropic",
    )
    provider.update_api_key("new-key")
    assert provider.api_key == "new-key"
    assert str(provider._anthropic_client.base_url).rstrip("/") == ROUTER_URL

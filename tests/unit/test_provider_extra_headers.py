# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 47 Tier 1 — TokenDance provider registry entry + attribution headers.

Covers:
- LLMProvider threads registry `extra_headers` into both SDK clients
  (default_headers) and preserves them across update_api_key.
- The bundled tokendance registry entry is well-formed.
- The /providers/models chat-protocol filter drops non-chat router entries
  (image/video/TTS) while keeping every provider without the field intact.
"""

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.api.routes.agents_v2 import _chat_model_ids
from agent_os.config.provider_registry import ProviderRegistry

HEADERS = {"X-App-Name": "Orbital", "X-Site-URL": "https://github.com/zqiren/Orbital"}


# ---- LLMProvider header threading ----

def test_openai_client_carries_extra_headers():
    p = LLMProvider("some-model", "sk-test", "https://example.com/v1",
                    sdk="openai", extra_headers=HEADERS)
    assert p._openai_client.default_headers["X-App-Name"] == "Orbital"
    assert p._openai_client.default_headers["X-Site-URL"] == HEADERS["X-Site-URL"]


def test_anthropic_client_carries_extra_headers():
    p = LLMProvider("some-model", "sk-test", "https://example.com",
                    sdk="anthropic", extra_headers=HEADERS)
    assert p._anthropic_client.default_headers["X-App-Name"] == "Orbital"


def test_no_extra_headers_is_default_and_clean():
    p = LLMProvider("some-model", "sk-test", "https://example.com/v1", sdk="openai")
    assert p.extra_headers is None
    assert "X-App-Name" not in p._openai_client.default_headers


def test_update_api_key_preserves_extra_headers():
    p = LLMProvider("some-model", "sk-test", "https://example.com/v1",
                    sdk="openai", extra_headers=HEADERS)
    p.update_api_key("sk-new")
    assert p._openai_client.default_headers["X-App-Name"] == "Orbital"
    assert p.api_key == "sk-new"


def test_update_api_key_preserves_extra_headers_anthropic():
    p = LLMProvider("some-model", "sk-test", "https://example.com",
                    sdk="anthropic", extra_headers=HEADERS)
    p.update_api_key("sk-new")
    assert p._anthropic_client.default_headers["X-App-Name"] == "Orbital"


def test_extra_headers_copied_not_aliased():
    src = dict(HEADERS)
    p = LLMProvider("m", "k", None, sdk="openai", extra_headers=src)
    src["X-App-Name"] = "mutated"
    assert p.extra_headers["X-App-Name"] == "Orbital"


# ---- Bundled tokendance registry entry ----

@pytest.fixture
def tokendance():
    return ProviderRegistry().get_provider_data("tokendance")


def test_tokendance_entry_shape(tokendance):
    assert tokendance["base_url"] == "https://tokendance.space/gateway/v1"
    assert tokendance["sdk"] == "openai"
    assert tokendance["china_only"] is True
    assert tokendance["supports_model_list"] is True
    assert tokendance["currency"] == "CNY"
    assert tokendance["suggested_models"], "suggested_models must be non-empty"


def test_tokendance_attribution_headers(tokendance):
    headers = tokendance["extra_headers"]
    # Their app-attribution doc: localhost apps MUST send X-App-Name to be tracked.
    assert headers["X-App-Name"] == "Orbital"
    assert headers["X-Site-URL"].startswith("https://")


def test_tokendance_default_pricing_present(tokendance):
    # Placeholder until partner machine-readable pricing (Spec 47 R7); without a
    # _default the ledger would silently bill at the 3.0/15.0 USD global fallback.
    default = tokendance["pricing"]["_default"]
    assert default["input_per_1m"] > 0
    assert default["output_per_1m"] > 0
    assert default["cached_input_per_1m"] < default["input_per_1m"]


def test_tokendance_suggested_models_have_metadata(tokendance):
    for model_id in tokendance["suggested_models"]:
        assert model_id in tokendance["models"], f"{model_id} missing models entry"


# ---- /providers/models chat-protocol filter ----

def test_chat_filter_keeps_chat_and_drops_nonchat():
    data = {"data": [
        {"id": "deepseek-v4-pro", "supported_protocols": ["openai:chat-completions", "anthropic:messages"]},
        {"id": "kimi-k3", "supported_protocols": ["openai:chat-completions"]},
        {"id": "claude-ish", "supported_protocols": ["anthropic:messages"]},
        {"id": "seedream-5.0-pro", "supported_protocols": ["ark:image-generations"]},
        {"id": "kling-3.0", "supported_protocols": ["kling:text2video", "kling:image2video"]},
        {"id": "minimax-speech-2.8-hd", "supported_protocols": ["minimax:t2a_v2"]},
    ]}
    assert _chat_model_ids(data) == ["deepseek-v4-pro", "kimi-k3", "claude-ish"]


def test_chat_filter_keeps_entries_without_protocol_field():
    # Every non-router provider (OpenAI, DeepSeek, ...) has no supported_protocols.
    data = {"data": [{"id": "gpt-5.5"}, {"id": "deepseek-chat", "supported_protocols": []}]}
    assert _chat_model_ids(data) == ["gpt-5.5", "deepseek-chat"]


def test_chat_filter_skips_missing_ids_and_bad_shapes():
    data = {"data": [{"id": ""}, {"no_id": True}, {"id": "ok", "supported_protocols": "weird"}]}
    assert _chat_model_ids(data) == ["ok"]

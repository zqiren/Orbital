# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenCode Go per-conversation routing header.

Since 2026-09 the Go gateway answers 400 ``MissingSessionID`` to any request
without ``x-opencode-session`` (verified on the wire on both the
/chat/completions and the Anthropic /messages tier paths) and asks clients
to identify with their own User-Agent. The registry declares the header
name (``session_header``); LLMProvider sends it per request with the
conversation id the loop binds; unbound clients (Test Connection, utility
calls outside a loop) send a stable per-instance id so it is never missing.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers import openai_compat
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.config.provider_registry import ProviderRegistry

HDR = "x-opencode-session"


# ---- registry ----

def test_opencode_go_entry_declares_session_header_and_user_agent():
    go = ProviderRegistry().get_provider_data("opencode-go")
    assert go["session_header"] == HDR
    assert go["extra_headers"]["User-Agent"].startswith("Orbital/")


def test_no_other_bundled_provider_gained_a_session_header_by_accident():
    reg = ProviderRegistry()
    with_header = {k for k, v in reg.all_providers().items() if v.get("session_header")}
    assert with_header == {"opencode-go"}


# ---- header value resolution ----

def test_provider_without_session_header_sends_no_per_request_headers():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai")
    assert p.session_header is None
    assert p._request_headers() is None


def test_unbound_provider_uses_a_stable_per_instance_id():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    first = p._request_headers()
    assert set(first) == {HDR} and first[HDR]
    assert p._request_headers() == first, "must be stable across requests"
    q = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    assert q._request_headers()[HDR] != first[HDR], "distinct clients are distinct conversations"


def test_bound_getter_is_read_live_on_every_request():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    box = {"sid": "sess-A"}
    p.bind_session_id(lambda: box["sid"])
    assert p._request_headers() == {HDR: "sess-A"}
    box["sid"] = "sess-B"
    assert p._request_headers() == {HDR: "sess-B"}


def test_empty_or_failing_getter_falls_back_to_instance_id():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    p.bind_session_id(lambda: None)
    assert p._request_headers()[HDR] == p._instance_session_id

    def boom():
        raise RuntimeError("session gone")
    p.bind_session_id(boom)
    assert p._request_headers()[HDR] == p._instance_session_id


def test_user_agent_version_token_is_expanded(monkeypatch):
    import agent_os.version as v
    monkeypatch.setattr(v, "get_version", lambda: "9.9.9")
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai",
                    extra_headers={"User-Agent": "Orbital/{version}", "X-Static": "s"})
    assert p.extra_headers == {"User-Agent": "Orbital/9.9.9", "X-Static": "s"}
    assert p._openai_client.default_headers["User-Agent"] == "Orbital/9.9.9"


# ---- the header reaches the SDK call, on every path ----

def _openai_response():
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_openai_complete_passes_extra_headers():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    p.bind_session_id(lambda: "sess-1")
    create = AsyncMock(return_value=_openai_response())
    p._openai_client.chat.completions.create = create
    asyncio.run(p.complete([{"role": "user", "content": "hi"}]))
    assert create.call_args.kwargs["extra_headers"] == {HDR: "sess-1"}


def test_openai_complete_omits_extra_headers_key_when_unconfigured():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai")
    create = AsyncMock(return_value=_openai_response())
    p._openai_client.chat.completions.create = create
    asyncio.run(p.complete([{"role": "user", "content": "hi"}]))
    assert "extra_headers" not in create.call_args.kwargs


def test_openai_stream_passes_extra_headers():
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    p.bind_session_id(lambda: "sess-2")

    async def _aiter():
        if False:
            yield None
    create = AsyncMock(return_value=_aiter())
    p._openai_client.chat.completions.create = create

    async def drain():
        async for _ in p.stream([{"role": "user", "content": "hi"}]):
            pass
    asyncio.run(drain())
    assert create.call_args.kwargs["extra_headers"] == {HDR: "sess-2"}


def _anthropic_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
        model="m",
    )


def test_anthropic_complete_passes_extra_headers():
    p = LLMProvider("m", "k", "https://example.com", sdk="anthropic", session_header=HDR)
    p.bind_session_id(lambda: "sess-3")
    create = AsyncMock(return_value=_anthropic_response())
    p._anthropic_client.messages.create = create
    asyncio.run(p.complete([{"role": "user", "content": "hi"}]))
    assert create.call_args.kwargs["extra_headers"] == {HDR: "sess-3"}


def test_anthropic_stream_passes_extra_headers():
    p = LLMProvider("m", "k", "https://example.com", sdk="anthropic", session_header=HDR)
    p.bind_session_id(lambda: "sess-4")

    async def _aiter():
        if False:
            yield None
    create = AsyncMock(return_value=_aiter())
    p._anthropic_client.messages.create = create

    async def drain():
        async for _ in p.stream([{"role": "user", "content": "hi"}]):
            pass
    asyncio.run(drain())
    assert create.call_args.kwargs["extra_headers"] == {HDR: "sess-4"}


# ---- the loop binds its live session to every provider it owns ----

def _loop_with(session, **providers):
    return AgentLoop(session, providers["provider"], MagicMock(), MagicMock(),
                     utility_provider=providers.get("utility"),
                     fallback_providers=providers.get("fallbacks"),
                     auth_fallback_provider=providers.get("auth"))


def test_loop_binds_session_uuid_to_all_providers_and_follows_a_swap():
    mk = lambda: LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    primary, util, fb, auth = mk(), mk(), mk(), mk()
    session = SimpleNamespace(session_id="chat-1", session_uuid="uuid-1")
    loop = _loop_with(session, provider=primary, utility=util, fallbacks=[fb], auth=auth)
    for p in (primary, util, fb, auth):
        assert p._request_headers() == {HDR: "uuid-1"}
    # the manager swaps the session object on a hot resume
    loop._session = SimpleNamespace(session_id="chat-2", session_uuid="uuid-2")
    for p in (primary, util, fb, auth):
        assert p._request_headers() == {HDR: "uuid-2"}


def test_loop_falls_back_to_f1_chat_id_without_a_uuid():
    primary = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    _loop_with(SimpleNamespace(session_id="chat-9"), provider=primary)
    assert primary._request_headers() == {HDR: "chat-9"}


def test_loop_tolerates_providers_without_bind(monkeypatch):
    # Test doubles / older provider objects have no bind_session_id.
    _loop_with(SimpleNamespace(session_id="s"), provider=MagicMock(spec=[]))


def test_rebind_after_provider_rebuild():
    session = SimpleNamespace(session_id="chat-1", session_uuid="uuid-1")
    loop = _loop_with(session, provider=LLMProvider("m", "k", None, sdk="openai", session_header=HDR))
    fresh = LLMProvider("m2", "k", None, sdk="openai", session_header=HDR)
    assert fresh._request_headers()[HDR] != "uuid-1"
    loop._provider = fresh
    loop.bind_session_headers()
    assert fresh._request_headers() == {HDR: "uuid-1"}

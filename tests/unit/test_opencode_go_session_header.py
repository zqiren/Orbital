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


# ---- spec 140: project-scoped routing key for trigger-started sessions ----
#
# The header is a ROUTING key: same value = same upstream machine = warm
# prompt cache. A registry ``session_scope: "project"`` makes trigger-started
# manager sessions send one stable ``project:<id>`` value so every fire of a
# trigger lands on the warm route. Chat sessions (and everything else) keep
# the per-session uuid. Providers without the field are unchanged.

def _go(**kw):
    return LLMProvider("m", "k", "https://example.com/v1", sdk="openai",
                       session_header=HDR, **kw)


def _trigger_loop(session, provider, project_id="proj-1", **providers):
    return AgentLoop(session, provider, MagicMock(), MagicMock(),
                     utility_provider=providers.get("utility"),
                     fallback_providers=providers.get("fallbacks"),
                     auth_fallback_provider=providers.get("auth"),
                     routing_project_id=project_id)


def test_registry_opencode_go_is_project_scoped():
    go = ProviderRegistry().get_provider_data("opencode-go")
    assert go["session_scope"] == "project"
    assert "session_scope" in go["notes"]


def test_session_scope_only_on_providers_that_have_a_session_header():
    reg = ProviderRegistry()
    scoped = {k: v for k, v in reg.all_providers().items() if v.get("session_scope")}
    assert set(scoped) == {"opencode-go"}
    for v in scoped.values():
        assert v["session_scope"] == "project" and v.get("session_header")


def test_trigger_loop_on_project_scoped_provider_sends_the_project_key():
    primary, util, fb, auth = (_go(session_scope="project") for _ in range(4))
    session = SimpleNamespace(session_id="chat-1", session_uuid="uuid-1")
    _trigger_loop(session, primary, utility=util, fallbacks=[fb], auth=auth)
    for p in (primary, util, fb, auth):
        assert p._request_headers() == {HDR: "project:proj-1"}
        assert p.request_session_id() == "project:proj-1"


def test_chat_loop_on_project_scoped_provider_still_sends_the_session_uuid():
    primary = _go(session_scope="project")
    _loop_with(SimpleNamespace(session_id="chat-1", session_uuid="uuid-1"), provider=primary)
    assert primary._request_headers() == {HDR: "uuid-1"}


def test_trigger_loop_on_provider_without_session_scope_sends_the_session_uuid():
    primary = _go()
    assert primary.session_scope is None
    _trigger_loop(SimpleNamespace(session_id="chat-1", session_uuid="uuid-1"), primary)
    assert primary._request_headers() == {HDR: "uuid-1"}


def test_unknown_session_scope_value_is_per_conversation():
    primary = _go(session_scope="galaxy")
    _trigger_loop(SimpleNamespace(session_id="chat-1", session_uuid="uuid-1"), primary)
    assert primary._request_headers() == {HDR: "uuid-1"}


def test_trigger_loop_keeps_project_key_across_a_session_swap_and_hot_resume():
    session = SimpleNamespace(session_id="chat-1", session_uuid="uuid-1")
    loop = _trigger_loop(session, _go(session_scope="project"))
    # a later user message into the same trigger session: same loop, same key
    loop._session = SimpleNamespace(session_id="chat-1", session_uuid="uuid-1")
    assert loop._provider._request_headers() == {HDR: "project:proj-1"}
    # hot resume rebuilds the clients (fresh, unbound) and rebinds them
    fresh, fresh_util = _go(session_scope="project"), _go(session_scope="project")
    assert fresh._request_headers()[HDR] != "project:proj-1"
    loop._provider, loop._utility_provider = fresh, fresh_util
    loop.bind_session_headers()
    assert fresh._request_headers() == {HDR: "project:proj-1"}
    assert fresh_util._request_headers() == {HDR: "project:proj-1"}


def test_hot_resume_onto_a_provider_without_scope_falls_back_to_the_uuid():
    loop = _trigger_loop(SimpleNamespace(session_id="c", session_uuid="uuid-7"),
                         _go(session_scope="project"))
    fresh = _go()
    loop._provider = fresh
    loop.bind_session_headers()
    assert fresh._request_headers() == {HDR: "uuid-7"}


def test_chat_loop_never_clears_a_shared_providers_project_key():
    # Fan-out workers share the parent's provider object; a loop without a
    # routing project must not unbind the parent's project key.
    shared = _go(session_scope="project")
    _trigger_loop(SimpleNamespace(session_id="c", session_uuid="parent-uuid"), shared)
    shared_getter = shared._session_id_getter
    from agent_os.daemon_v2.native_worker import _WorkerProviderView
    _loop_with(SimpleNamespace(session_id="w", session_uuid="worker-uuid"),
               provider=_WorkerProviderView(shared, lambda: None))
    assert shared._session_id_getter is shared_getter
    assert shared._request_headers() == {HDR: "project:proj-1"}


def test_project_key_is_unmistakably_not_a_uuid():
    p = _go(session_scope="project")
    p.bind_routing_project("1f0c2a")
    assert p.request_session_id() == "project:1f0c2a"
    p.bind_routing_project(None)
    assert p.request_session_id() == p._instance_session_id


def test_project_key_reaches_the_wire_and_the_dump(tmp_path, monkeypatch):
    monkeypatch.setenv(openai_compat.DUMP_ENV, str(tmp_path))
    p = _go(session_scope="project")
    _trigger_loop(SimpleNamespace(session_id="c", session_uuid="uuid-1"), p, project_id="pid9")
    create = AsyncMock(return_value=_openai_response())
    p._openai_client.chat.completions.create = create
    asyncio.run(p.complete([{"role": "user", "content": "hi"}]))
    assert create.call_args.kwargs["extra_headers"] == {HDR: "project:pid9"}
    dumps = list(tmp_path.glob("*.json"))
    assert [d.name for d in dumps] == ["project_pid9_1.json"]


# ---- the registry field reaches every LLMProvider construction ----

SCOPED_JSON = {
    "providers": {
        "go": {
            "display_name": "Scoped gateway",
            "base_url": "https://go.example/v1",
            "sdk": "openai",
            "session_header": HDR,
            "session_scope": "project",
            "suggested_models": ["m1"],
            "models": {"m1": {"context_window": 100, "max_output": 10},
                       "_default": {"context_window": 50, "max_output": 5}},
        },
        "plain": {
            "display_name": "Plain",
            "base_url": "https://plain.example/v1",
            "sdk": "openai",
            "session_header": HDR,
            "suggested_models": ["p1"],
            "models": {"_default": {"context_window": 50, "max_output": 5}},
        },
    },
    "defaults": {"unknown_model": {"context_window": 10}},
}


class _RecordingCtor:
    def __init__(self):
        self.calls = []

    def __call__(self, model, api_key=None, base_url=None, **kw):
        self.calls.append(dict(kw, model=model))
        inst = MagicMock()
        inst.complete = AsyncMock(return_value=MagicMock())
        return inst


@pytest.fixture
def scoped_registry(tmp_path):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(SCOPED_JSON), encoding="utf-8")
    return ProviderRegistry(str(path))


def test_agent_manager_threads_session_scope_into_every_provider(scoped_registry, monkeypatch):
    from agent_os.daemon_v2 import agent_manager as am_mod
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.daemon_v2.models import AgentConfig, FallbackModelEntry
    ctor = _RecordingCtor()
    monkeypatch.setattr(am_mod, "LLMProvider", ctor)
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
                       process_manager=MagicMock(), provider_registry=scoped_registry)
    cfg = AgentConfig(workspace="/tmp/ws", model="m1", api_key="sk-test",
                      base_url="https://go.example/v1", sdk="openai", provider="go",
                      utility_model="m1",
                      llm_fallback_models=[FallbackModelEntry(model="p1", provider="plain")])
    mgr._build_llm_providers(cfg)
    scopes = [(c["model"], c.get("session_scope")) for c in ctor.calls]
    assert scopes == [("m1", "project"), ("p1", None), ("m1", "project")]
    assert all(c["session_header"] == HDR for c in ctor.calls)


def test_auth_fallback_provider_gets_session_scope(scoped_registry, monkeypatch):
    from agent_os.daemon_v2 import agent_manager as am_mod
    from agent_os.daemon_v2.agent_manager import AgentManager
    ctor = _RecordingCtor()
    monkeypatch.setattr(am_mod, "LLMProvider", ctor)
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
                       process_manager=MagicMock(), provider_registry=scoped_registry)
    af = SimpleNamespace(provider="go", model="m1", api_key="k", base_url=None, sdk="openai")
    mgr._build_auth_fallback_provider(SimpleNamespace(auth_fallback=af))
    assert ctor.calls[-1]["session_scope"] == "project"


def test_connection_test_threads_session_scope(scoped_registry, monkeypatch):
    import agent_os.agent.providers.openai_compat as oc
    from agent_os.api.routes import agents_v2
    ctor = _RecordingCtor()
    monkeypatch.setattr(oc, "LLMProvider", ctor)
    monkeypatch.setattr(agents_v2, "_provider_registry", scoped_registry)
    out = asyncio.run(agents_v2.run_connection_test("go", "m1", "sk-x", None, "openai"))
    assert out["ok"] is True
    assert ctor.calls[-1]["session_scope"] == "project"
    assert ctor.calls[-1]["session_header"] == HDR


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger_source,expected", [("schedule", "p1"), (None, None)])
async def test_start_agent_gives_only_trigger_loops_the_routing_project(
        tmp_path, trigger_source, expected):
    from unittest.mock import patch
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.daemon_v2.models import AgentConfig
    projects = {"p1": {"project_id": "p1", "name": "demo", "workspace": str(tmp_path),
                       "is_scratch": False}}
    store = MagicMock()
    store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    store.list_projects = MagicMock(return_value=list(projects.values()))
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[])
    sub.stop = AsyncMock()
    sub.stop_all = AsyncMock()
    mgr = AgentManager(project_store=store, ws_manager=MagicMock(), sub_agent_manager=sub,
                       activity_translator=MagicMock(), process_manager=MagicMock())
    session = Session.new("demo_trig00001", str(tmp_path))
    config = AgentConfig(workspace=str(tmp_path), model="gpt-4", api_key="sk-test")
    with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
         patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
         patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
         patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
         patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
         patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):
        MockReg.return_value.tool_names.return_value = ["read"]
        MockLoop.return_value.run = AsyncMock()
        await mgr.start_agent("p1", config, session=session, trigger_source=trigger_source)
        assert MockLoop.call_args.kwargs["routing_project_id"] == expected


# ---- non-ASCII conversation ids (GitHub #61) ----

def _bound(sid):
    p = LLMProvider("m", "k", "https://example.com/v1", sdk="openai", session_header=HDR)
    p.bind_session_id(lambda: sid)
    return p


def test_a_chinese_project_session_id_becomes_a_legal_header_value():
    """Session ids start with the project name, and httpx refuses a non-ASCII
    header value before any network call ('ascii' codec can't encode)."""
    value = _bound("新项目_1a2b3c4d")._request_headers()[HDR]
    assert value.isascii()
    value.encode("ascii")  # what httpx does


def test_the_encoded_id_is_stable_and_distinct_per_conversation():
    """The gateway routes (and caches) on this value."""
    a = _bound("新项目_1a2b3c4d")
    assert a._request_headers() == a._request_headers()
    assert a._request_headers() != _bound("新项目_ffffffff")._request_headers()
    assert a._request_headers() != _bound("旧项目_1a2b3c4d")._request_headers()


def test_an_ascii_session_id_is_sent_byte_for_byte():
    """Existing conversations keep their routing key across the upgrade."""
    assert _bound("my-proj_1a2b3c4d")._request_headers()[HDR] == "my-proj_1a2b3c4d"

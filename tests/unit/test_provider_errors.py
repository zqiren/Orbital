# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Credential/provider error classification (silent-error surfacing).

Covers the taxonomy that lets the frontend show actionable messages instead
of swallowed 500s / log-only failures:
  - ProviderConfigError carries a stable machine code
  - classify_llm_error maps SDK/runtime exceptions -> (code, message)
  - _build_llm_providers raises missing_api_key BEFORE constructing an SDK
    client, so every start path (scan/start/inject/trigger) gets a typed error
"""
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from agent_os.daemon_v2.models import AgentConfig
from agent_os.daemon_v2.provider_errors import (
    ProviderConfigError,
    classify_llm_error,
)


class _FakeStatusError(Exception):
    """Stand-in for openai/anthropic APIStatusError (both expose status_code)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def test_provider_config_error_carries_code_and_message():
    e = ProviderConfigError("missing_api_key", "No API key configured")
    assert e.code == "missing_api_key"
    assert str(e) == "No API key configured"


def test_classify_passes_provider_config_error_through():
    code, msg = classify_llm_error(
        ProviderConfigError("missing_api_key", "No API key configured")
    )
    assert code == "missing_api_key"
    assert msg == "No API key configured"


@pytest.mark.parametrize("status", [401, 403])
def test_classify_auth_statuses_as_invalid_api_key(status):
    code, msg = classify_llm_error(_FakeStatusError("nope", status))
    assert code == "invalid_api_key"
    assert "nope" in msg


def test_classify_404_as_model_not_found():
    code, _ = classify_llm_error(_FakeStatusError("model X not found", 404))
    assert code == "model_not_found"


def test_classify_real_openai_connection_error_as_unreachable():
    exc = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.example.com/v1")
    )
    code, _ = classify_llm_error(exc)
    assert code == "provider_unreachable"


def test_classify_unknown_exception_as_provider_error():
    code, msg = classify_llm_error(RuntimeError("boom"))
    assert code == "provider_error"
    assert "boom" in msg


def _make_manager():
    from agent_os.daemon_v2.agent_manager import AgentManager

    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        settings_store=MagicMock(),
        credential_store=MagicMock(),
        browser_manager=MagicMock(),
        provider_registry=MagicMock(),
    )


def test_build_llm_providers_raises_missing_api_key_on_empty_key():
    manager = _make_manager()
    config = AgentConfig(
        workspace="/tmp",
        model="MiniMax-M3",
        api_key="",
        base_url="https://api.example.com/v1",
    )
    with pytest.raises(ProviderConfigError) as ei:
        manager._build_llm_providers(config)
    assert ei.value.code == "missing_api_key"


def test_build_llm_providers_succeeds_with_key_present():
    manager = _make_manager()
    config = AgentConfig(
        workspace="/tmp",
        model="MiniMax-M3",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
    )
    provider, fallbacks, utility, model_info = manager._build_llm_providers(config)
    assert provider is not None
    assert fallbacks == []


def test_record_loop_error_classifies_stores_and_broadcasts():
    manager = _make_manager()
    manager._record_loop_error(
        "proj_x", "sess_1",
        ProviderConfigError("missing_api_key", "No LLM API key configured"),
    )
    ev = manager.get_last_terminal_event("proj_x", session_id="sess_1")
    assert ev["type"] == "error"
    assert ev["error_code"] == "missing_api_key"
    assert ev["details"] == "No LLM API key configured"

    payload = manager._ws.broadcast.call_args[0][1]
    assert payload["type"] == "agent.status"
    assert payload["status"] == "error"
    assert payload["error_code"] == "missing_api_key"
    assert payload["reason"] == "No LLM API key configured"


def test_record_loop_error_maps_401_to_invalid_api_key():
    manager = _make_manager()
    manager._record_loop_error("proj_x", "sess_1", _FakeStatusError("Unauthorized", 401))
    ev = manager.get_last_terminal_event("proj_x", session_id="sess_1")
    assert ev["error_code"] == "invalid_api_key"


def test_terminal_event_without_error_code_keeps_legacy_shape():
    manager = _make_manager()
    manager._set_last_terminal_event("proj_x", "sess_1", "stopped")
    ev = manager.get_last_terminal_event("proj_x", session_id="sess_1")
    assert "error_code" not in ev


# ---------------------------------------------------------------------------
# Mid-run LLM failures (ABORT 401/403/400, retries exhausted) are handled
# INSIDE the loop — it appends a session system row and ends normally, so
# _on_loop_done sees no exception. The loop must expose the terminal error
# and the manager must broadcast it, or the UI shows nothing (the original
# invalid-key silent failure).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_exposes_terminal_llm_error(tmp_path):
    from agent_os.agent.loop import AgentLoop
    from agent_os.agent.session import Session
    from agent_os.agent.context import ContextManager
    from agent_os.agent.prompt_builder import Autonomy, PromptContext
    from agent_os.agent.providers.types import LLMError

    session = Session.new("llmfail", str(tmp_path))

    class AbortProvider:
        model = "test-model"
        sdk = "openai"
        provider = "test"
        reasoning = None
        capabilities = None

        async def stream(self, messages, tools=None):
            raise LLMError("Incorrect API key provided", status_code=401)
            yield  # pragma: no cover — makes this an async generator

    class EmptyRegistry:
        def schemas(self):
            return []

        def is_async(self, name):
            return False

        def execute(self, name, arguments):
            raise RuntimeError("no tools")

        def tool_names(self):
            return []

        def reset_run_state(self):
            pass

    class Builder:
        def build(self, context):
            return ("prefix", "suffix", "runtime")

    ctx = PromptContext(
        workspace=str(tmp_path), model="test-model",
        autonomy=Autonomy.HANDS_OFF, enabled_agents=[], tool_names=[],
        os_type="linux", datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )
    loop = AgentLoop(session, AbortProvider(), EmptyRegistry(),
                     ContextManager(session, Builder(), ctx))
    await loop.run(initial_message="hi")

    err = loop.last_llm_error
    assert err is not None
    assert err.status_code == 401
    code, _ = classify_llm_error(err)
    assert code == "invalid_api_key"


def test_on_loop_done_broadcasts_llm_error_swallowed_by_loop():
    from agent_os.agent.providers.types import LLMError
    from agent_os.daemon_v2.agent_manager import make_session_key

    manager = _make_manager()
    handle = MagicMock()
    handle.loop.last_llm_error = LLMError("Incorrect API key", status_code=401)
    handle.session.is_stopped.return_value = False
    handle.session.pop_deferred_messages.return_value = []
    handle.session._paused_for_approval = False
    handle.session.pop_queued_messages.return_value = []
    manager._handles[make_session_key("proj_x", "sess_1")] = handle
    manager._sub_agent_manager.list_active.return_value = []

    task = MagicMock()
    task.cancelled.return_value = False
    task.exception.return_value = None
    manager._on_loop_done("proj_x", session_id="sess_1")(task)

    ev = manager.get_last_terminal_event("proj_x", session_id="sess_1")
    assert ev is not None and ev["type"] == "error"
    assert ev["error_code"] == "invalid_api_key"
    payloads = [c[0][1] for c in manager._ws.broadcast.call_args_list]
    err_payload = next(p for p in payloads if p.get("status") == "error")
    assert err_payload["error_code"] == "invalid_api_key"

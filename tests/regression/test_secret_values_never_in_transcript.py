# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SECRECY regression: credential VALUES never reach the transcript.

TASK-credential-contract-fixes 1e — the absolute invariant: the LLM provider
must never see credential values. Values live in the (fake) keychain and are
substituted into tool args at execution time only. Everything the loop writes
back — session transcript, every tool result, log output — may contain the
PLACEHOLDER `<secret:name.field>`, never the value.

Full round trip with the REAL wiring (the production shape from
agent_manager._build_registry: ToolRegistry(user_credential_store=store) AND
BrowserTool(user_credential_store=store)):
  1. agent calls request_credential → pending → loop pauses;
  2. user submits the modal (sentinel value lands in the store);
  3. resume → agent types the token via the browser `type` action
     (mocked Playwright page — no real browser);
  4. scan the session JSONL, every in-memory message, and captured logs:
     the sentinel appears NOWHERE; the page locator (execution side)
     received the real value.

This test FAILS on the pre-fix code: central substitution in the registry
resolved the token BEFORE BrowserTool._action_type computed its display text,
so the tool result echoed `Typed '<REAL VALUE>' into element` into the
transcript.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMResponse, StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.browser_refs import RefEntry
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.request_credential import RequestCredentialTool

SENTINEL = "SENTINEL-hunter2-zQ81xKfv-NEVER-IN-TRANSCRIPT"


# --- harness (mirrors tests/regression/test_yield_turn.py) ------------------

def _ctx(ws: str) -> PromptContext:
    return PromptContext(
        workspace=ws, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["request_credential", "browser"],
        os_type="linux", datetime_now="2026-06-12T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("sys", "suffix", "runtime")


def _text(text: str) -> LLMResponse:
    return LLMResponse(raw_message={"role": "assistant", "content": text},
                       text=text, tool_calls=[], has_tool_calls=False,
                       finish_reason="stop", status_text=None,
                       usage=TokenUsage(10, 5))


def _tool(tc_id: str, name: str, args: str = "{}") -> LLMResponse:
    tc = {"id": tc_id, "type": "function",
          "function": {"name": name, "arguments": args}}
    return LLMResponse(raw_message={"role": "assistant", "content": None,
                                    "tool_calls": [tc]},
                       text="", tool_calls=[tc], has_tool_calls=True,
                       finish_reason="tool_calls", status_text=None,
                       usage=TokenUsage(10, 5))


class _Provider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.stream_call_count = 0

    async def stream(self, messages, tools=None):
        self.stream_call_count += 1
        idx = self._idx
        self._idx += 1
        if idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            for i, tc in enumerate(resp.tool_calls):
                d = dict(tc)
                d["index"] = i
                yield StreamChunk(tool_calls_delta=[d])
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True, usage=TokenUsage(1, 1))


class _FakeCredStore:
    def __init__(self):
        self._values: dict[tuple[str, str], str] = {}
        self._meta: dict[str, dict] = {}

    def get_metadata(self, name):
        return self._meta.get(name)

    def get_value(self, name, field):
        return self._values.get((name, field))

    def store(self, name, domain, fields: dict):
        self._meta[name] = {"domain": domain, "fields": list(fields)}
        for f, v in fields.items():
            self._values[(name, f)] = v


def _make_mock_page():
    page = AsyncMock()
    page.url = "https://cloud.dify.ai/signin"
    page.title = AsyncMock(return_value="Sign in")
    page.is_closed = MagicMock(return_value=False)
    page.wait_for_load_state = AsyncMock()
    return page


def _make_browser_manager(page):
    bm = MagicMock()
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/ws/orbital/output/screenshots/step_0001.png")
    bm.get_ref_map = MagicMock(return_value={"e5": RefEntry(role="textbox", name="Password", nth=0)})
    bm.store_ref_map = MagicMock()
    bm.clear_ref_map = MagicMock()
    return bm


@pytest.mark.asyncio
async def test_secret_values_never_in_transcript_or_logs(tmp_path, caplog):
    store = _FakeCredStore()
    page = _make_mock_page()
    bm = _make_browser_manager(page)

    # Production wiring shape: store goes to BOTH registry and BrowserTool.
    registry = ToolRegistry(user_credential_store=store)
    registry.register(RequestCredentialTool(credential_store=store))
    registry.register(BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace=str(tmp_path),
        autonomy_preset="hands_off",
        screenshot_namespace="default",
        user_credential_store=store,
    ))

    session = Session.new("secrecy-roundtrip", str(tmp_path))

    provider = _Provider([
        _tool("tc-cred-1", "request_credential", json.dumps({
            "name": "dify", "domain": "cloud.dify.ai",
            "fields": ["password"], "reason": "login",
        })),
        _tool("tc-type-1", "browser", json.dumps({
            "action": "type", "ref": "e5",
            "text": "<secret:dify.password>",
        })),
        _text("done"),
    ])

    loop = AgentLoop(session, provider, registry,
                     ContextManager(session, _Builder(), _ctx(str(tmp_path))))

    mock_locator = AsyncMock()
    mock_locator.fill = AsyncMock()

    with caplog.at_level(logging.DEBUG):
        # request → pause
        persist_user_row(loop._session, "log into dify")
        await loop.run()
        assert session.is_paused()

        # user submits the secure modal
        store.store("dify", "cloud.dify.ai", {"password": SENTINEL})

        # resume → agent types the token via browser `type`
        session.resume()
        with patch("agent_os.agent.tools.browser.resolve_ref",
                   new_callable=AsyncMock, return_value=mock_locator):
            await loop.run()

    # Execution side DID receive the real value (substitution worked) ...
    mock_locator.fill.assert_awaited_once_with(SENTINEL)

    # ... but the value appears NOWHERE the LLM (or a human reading the
    # transcript) can see:
    # 1. every in-memory message, incl. every tool result
    all_messages = json.dumps(session.get_messages())
    assert SENTINEL not in all_messages

    # 2. the session transcript on disk
    transcript = "\n".join(
        p.read_text(encoding="utf-8") for p in Path(tmp_path).rglob("*.jsonl")
    )
    assert SENTINEL not in transcript

    # 3. captured logs (all loggers, DEBUG and up)
    assert SENTINEL not in caplog.text

    # The placeholder MAY appear — that is the whole point of tokens.
    assert "<secret:dify.password>" in transcript

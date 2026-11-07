# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A thinking-off request the model rejects falls back to thinking on.

Smoke 2026-11-07: OpenCode Go now serves ``glm-5.2`` with GLM-5.3, which is
thinking-only. The catalog still says the model is hybrid
(``param:thinking.type=enabled``), so the Workbench router's
``complete(disable_reasoning=True)`` sends ``thinking: disabled`` and the
gateway answers 400 "GLM-5.3 is a thinking-only model; disabling thinking
(reasoning_effort='none') is not supported." on every attempt: every
Workbench send failed to route.

The provider now retries such a call once without the off-switch and stops
sending it for that model until the daemon restarts.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.providers.types import LLMError

REJECT = ("Upstream request failed: [400] GLM-5.3 is a thinking-only model; "
          "disabling thinking (reasoning_effort='none') is not supported.")


def _bad_request(message: str) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
    resp = httpx.Response(400, request=req, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=resp,
                                  body={"error": {"message": message}})


def _ok():
    msg = MagicMock()
    msg.content = '{"project_id": "p1"}'
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": msg.content}
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=None)


def _provider(create):
    reasoning = SimpleNamespace(supported=True, field="reasoning_content", echo_back="optional",
                                enable="param:thinking.type=enabled")
    p = LLMProvider(model="glm-5.2", api_key="k", base_url="http://x",
                    provider="opencode-go", reasoning=reasoning)
    p._openai_client = MagicMock()
    p._openai_client.chat.completions.create = create
    return p


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_memory():
    from agent_os.agent.providers import openai_compat
    openai_compat._REASONING_OFF_REJECTED.clear()
    yield
    openai_compat._REASONING_OFF_REJECTED.clear()


def test_rejected_off_switch_retries_with_thinking_on_and_remembers():
    create = AsyncMock(side_effect=[_bad_request(REJECT), _ok(), _ok()])
    p = _provider(create)
    out = _run(p.complete([{"role": "user", "content": "route"}], disable_reasoning=True))
    assert out.text == '{"project_id": "p1"}'
    first, second = create.call_args_list[0].kwargs, create.call_args_list[1].kwargs
    assert first["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in second
    # The next call does not try the off-switch again, even from a new
    # provider object (the Workbench router builds one per call).
    p2 = _provider(create)
    _run(p2.complete([{"role": "user", "content": "again"}], disable_reasoning=True))
    assert create.call_count == 3
    assert "extra_body" not in create.call_args_list[2].kwargs


def test_an_unrelated_400_is_not_retried():
    create = AsyncMock(side_effect=[_bad_request("messages: field required")])
    p = _provider(create)
    with pytest.raises(LLMError):
        _run(p.complete([{"role": "user", "content": "x"}], disable_reasoning=True))
    assert create.call_count == 1


def test_a_thinking_400_without_the_off_switch_is_not_retried():
    create = AsyncMock(side_effect=[_bad_request(REJECT)])
    p = _provider(create)
    with pytest.raises(LLMError):
        _run(p.complete([{"role": "user", "content": "x"}]))
    assert create.call_count == 1

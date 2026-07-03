# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: NativeWorkerAdapter + worker loop construction.

Exercises the adapter against a real ``AgentLoop``/``Session`` (not mocked)
with a stub provider and a stub tool registry, mirroring the real-loop test
style in ``test_loop_ledger_emission.py``. Covers the duck-type contract
``SubAgentManager._dispatch_async``'s ``_background_send`` fallback relies on
(sub_agent_manager.py:939-1036): ``_transport is None``, ``send()`` blocks for
one turn and never raises for task-level failures, ``_last_response`` carries
the outcome.
"""

import pytest

from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.native_worker import (
    NativeWorkerAdapter,
    WorkerDeps,
    make_worker_handle,
)


def test_make_worker_handle():
    assert make_worker_handle("a1b2c3d4", 0) == "worker:a1b2c3d4-0"
    assert make_worker_handle("a1b2c3d4", 2) == "worker:a1b2c3d4-2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubProvider:
    """Single-turn provider with provider/model/sdk identity attributes,
    matching the shape AgentLoop reads (see TextProvider in
    test_loop_ledger_emission.py)."""

    def __init__(self, text: str):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"
        self._text = text

    async def stream(self, messages, tools=None):
        yield StreamChunk(text=self._text)
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=10, output_tokens=5))


class _RaisingProvider:
    """Provider whose stream() raises mid-turn — simulates a task-level
    failure (network error, API error, etc.) inside the worker's loop."""

    def __init__(self):
        self.provider = "stub"
        self.model = "stub-model"
        self.sdk = "stub-sdk"

    async def stream(self, messages, tools=None):
        raise RuntimeError("boom: provider unreachable")
        yield  # pragma: no cover — makes this an async generator


class _StubToolRegistry:
    """Minimal ToolRegistryLike stub — no tools registered, just the shape
    AgentLoop's run() loop calls unconditionally every turn."""

    def schemas(self) -> list[dict]:
        return []

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return []

    def reset_run_state(self) -> None:
        pass


def _make_registry_factory(registry):
    def _make_tool_registry(allowed_paths, forbidden_paths):
        return registry
    return _make_tool_registry


@pytest.fixture
def stub_worker_deps(tmp_path):
    return WorkerDeps(
        provider=_StubProvider("done: 42"),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )


@pytest.fixture
def stub_worker_deps_err(tmp_path):
    return WorkerDeps(
        provider=_RaisingProvider(),
        workspace=str(tmp_path),
        project_id="proj-1",
        parent_session_id="parent-sess-1",
        make_tool_registry=_make_registry_factory(_StubToolRegistry()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_sets_last_response(tmp_path, stub_worker_deps):
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    assert a._transport is None
    assert a._idle is True
    await a.send("compute the answer")
    assert a._last_response == "done: 42"
    assert a._idle is True          # idle restored after turn
    assert a.is_running() is False


@pytest.mark.asyncio
async def test_send_failure_encoded_not_raised(tmp_path, stub_worker_deps_err):
    a = NativeWorkerAdapter(deps=stub_worker_deps_err, handle="worker:x-1",
                            display_name="t1", allowed_paths=None,
                            forbidden_paths=None)
    await a.send("boom")            # provider raises inside the loop
    assert a._last_response.startswith("Error")
    assert a._broken is True
    assert a.is_running() is False


@pytest.mark.asyncio
async def test_worker_session_tagged(tmp_path, stub_worker_deps):
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    # session JSONL exists and carries the kind meta row
    text = open(a.session_path).read()
    assert '"session_kind"' in text and '"worker"' in text


@pytest.mark.asyncio
async def test_make_tool_registry_receives_path_scopes(tmp_path, stub_worker_deps):
    """The adapter forwards allowed_paths/forbidden_paths verbatim to
    deps.make_tool_registry — Task 2's spawner-provided factory is the one
    that turns those into an enforced restricted registry; this task only
    guarantees the call happens with the right arguments."""
    seen = {}

    def _make_tool_registry(allowed_paths, forbidden_paths):
        seen["allowed"] = allowed_paths
        seen["forbidden"] = forbidden_paths
        return _StubToolRegistry()

    stub_worker_deps.make_tool_registry = _make_tool_registry
    NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                        display_name="t0",
                        allowed_paths=["src/"], forbidden_paths=["secrets/"])
    assert seen == {"allowed": ["src/"], "forbidden": ["secrets/"]}


@pytest.mark.asyncio
async def test_stop_cancels_turn(tmp_path, stub_worker_deps):
    """stop() delegates to loop.cancel_turn() and leaves the adapter in a
    non-running, idle state — safe to call even with no turn in flight."""
    a = NativeWorkerAdapter(deps=stub_worker_deps, handle="worker:x-0",
                            display_name="t0", allowed_paths=None,
                            forbidden_paths=None)
    await a.stop()
    assert a.is_running() is False
    assert a._idle is True

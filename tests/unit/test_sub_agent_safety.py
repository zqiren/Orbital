# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for sub-agent routing safety fixes.

Covers:
- Fix 1: Send serialization (asyncio.Lock on CLIAdapter.send)
- Fix 2: agent_message depth limit
- Fix 3: Cancel race (lifecycle lock on SubAgentManager)
- Fix 5: Sub-agent-aware idle status
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.agent.tools.agent_message import AgentMessageTool


def _make_mock_adapter(send_delay: float = 0):
    """CLIAdapter-like mock with controllable send delay."""
    adapter = MagicMock(spec=CLIAdapter)
    adapter._last_response = None
    adapter.is_alive = MagicMock(return_value=True)
    adapter.is_idle = MagicMock(return_value=False)

    call_log = []

    async def mock_send(msg):
        call_log.append(("enter", msg, asyncio.get_event_loop().time()))
        if send_delay > 0:
            await asyncio.sleep(send_delay)
        call_log.append(("exit", msg, asyncio.get_event_loop().time()))

    adapter.send = mock_send
    adapter.stop = AsyncMock()
    adapter.start = AsyncMock()
    adapter._call_log = call_log
    return adapter


def _make_sub_agent_manager(**kwargs):
    """SubAgentManager with mock process_manager."""
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    return SubAgentManager(process_manager=pm, **kwargs)


def _register_adapter(mgr, project_id, handle, adapter):
    """Directly inject an adapter into SubAgentManager for testing."""
    if project_id not in mgr._adapters:
        mgr._adapters[project_id] = {}
    mgr._adapters[project_id][handle] = adapter


# ---------------------------------------------------------------------------
# Fix 1: Send serialization
# ---------------------------------------------------------------------------

class TestSendSerialization:
    """Fix 1: Concurrent sends to the same adapter must not interleave.

    The lock lives on CLIAdapter._send_lock.  We test with a real CLIAdapter
    whose transport is mocked so no process is spawned.
    """

    @pytest.mark.asyncio
    async def test_concurrent_sends_serialized(self):
        """Two concurrent send() calls must not overlap execution."""
        call_log = []

        async def slow_transport_send(msg):
            call_log.append(("enter", msg, asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            call_log.append(("exit", msg, asyncio.get_event_loop().time()))
            return f"response to {msg}"

        transport = MagicMock()
        transport.send = slow_transport_send
        transport.is_alive = MagicMock(return_value=True)

        adapter = CLIAdapter(handle="a", display_name="A", transport=transport)

        await asyncio.gather(
            adapter.send("message-1"),
            adapter.send("message-2"),
        )

        enters = [(a, m, t) for a, m, t in call_log if a == "enter"]
        exits = [(a, m, t) for a, m, t in call_log if a == "exit"]

        assert len(enters) == 2
        assert len(exits) == 2

        # First send must exit before second send enters
        first_exit = min(exits, key=lambda x: x[2])
        second_enter = max(enters, key=lambda x: x[2])
        assert first_exit[2] <= second_enter[2], (
            "Sends overlapped — lock is not working"
        )

    @pytest.mark.asyncio
    async def test_sends_to_different_adapters_parallel(self):
        """Sends to different adapters should NOT block each other.

        SubAgentManager.send() is now non-blocking — it dispatches via
        background task. Both sends should return near-instantly and the
        background adapter.send() calls should be independent.
        """

        async def slow_transport_send(msg):
            await asyncio.sleep(0.05)
            return "ok"

        # Use spec to restrict transport to known interface (no dispatch)
        from agent_os.agent.transports.base import AgentTransport
        transport_a = MagicMock(spec=["send", "read_stream", "start", "stop", "is_alive"])
        transport_a.send = slow_transport_send
        transport_a.is_alive = MagicMock(return_value=True)

        transport_b = MagicMock(spec=["send", "read_stream", "start", "stop", "is_alive"])
        transport_b.send = slow_transport_send
        transport_b.is_alive = MagicMock(return_value=True)

        adapter_a = CLIAdapter(handle="a", display_name="A", transport=transport_a)
        adapter_b = CLIAdapter(handle="b", display_name="B", transport=transport_b)

        mgr = _make_sub_agent_manager()
        _register_adapter(mgr, "proj1", "agent-a", adapter_a)
        _register_adapter(mgr, "proj1", "agent-b", adapter_b)

        t0 = asyncio.get_event_loop().time()
        await asyncio.gather(
            mgr.send("proj1", "agent-a", "msg-a"),
            mgr.send("proj1", "agent-b", "msg-b"),
        )
        elapsed = asyncio.get_event_loop().time() - t0

        # Non-blocking dispatch returns immediately (< 0.05s)
        assert elapsed < 0.08, (
            f"Sends to different adapters blocked each other ({elapsed:.3f}s)"
        )

    @pytest.mark.asyncio
    async def test_send_to_stopped_adapter_returns_error(self):
        """send() after adapter removed returns error, no lock issues."""
        mgr = _make_sub_agent_manager()
        result = await mgr.send("proj1", "ghost", "hello")
        assert "Error" in result or "not running" in result


# ---------------------------------------------------------------------------
# Fix 2: agent_message depth limit
# ---------------------------------------------------------------------------

class TestAgentMessageDepthLimit:
    """Fix 2: agent_message tool must cap send calls per run."""

    def _make_tool(self, max_sends=3):
        mock_mgr = MagicMock()
        mock_mgr.send = AsyncMock(return_value="sent ok")
        mock_mgr.start = AsyncMock(return_value="started")
        mock_mgr.stop = AsyncMock(return_value="stopped")
        mock_mgr.list_active = MagicMock(return_value=[])
        mock_mgr.status = MagicMock(return_value="running")
        tool = AgentMessageTool(
            sub_agent_manager=mock_mgr,
            project_id="proj1",
            max_sends_per_run=max_sends,
        )
        return tool, mock_mgr

    @pytest.mark.asyncio
    async def test_sends_succeed_within_limit(self):
        """Sends under the limit succeed normally."""
        tool, mgr = self._make_tool(max_sends=3)
        for i in range(3):
            result = await tool.execute(action="send", agent="a", message=f"msg-{i}")
            assert "Error" not in result.content
            assert "limit" not in result.content.lower()
        assert mgr.send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_blocked_after_limit(self):
        """Send beyond the limit returns error, does NOT call sub_agent_manager."""
        tool, mgr = self._make_tool(max_sends=3)
        for i in range(3):
            await tool.execute(action="send", agent="a", message=f"msg-{i}")
        # 4th send should be blocked
        result = await tool.execute(action="send", agent="a", message="one-too-many")
        assert "limit" in result.content.lower() or "error" in result.content.lower()
        assert mgr.send.call_count == 3

    @pytest.mark.asyncio
    async def test_reset_clears_counter(self):
        """on_run_start() resets the counter, allowing sends again."""
        tool, mgr = self._make_tool(max_sends=3)
        for i in range(3):
            await tool.execute(action="send", agent="a", message=f"msg-{i}")
        tool.on_run_start()
        result = await tool.execute(action="send", agent="a", message="after-reset")
        assert "Error" not in result.content
        assert mgr.send.call_count == 4

    @pytest.mark.asyncio
    async def test_start_stop_not_counted(self):
        """start and stop actions do not increment the send counter."""
        tool, mgr = self._make_tool(max_sends=2)

        await tool.execute(action="start", agent="a", message="")
        await tool.execute(action="stop", agent="a", message="")
        await tool.execute(action="start", agent="b", message="")

        # Counter should still be 0, sends should still work
        result = await tool.execute(action="send", agent="a", message="msg1")
        assert "Error" not in result.content
        result = await tool.execute(action="send", agent="a", message="msg2")
        assert "Error" not in result.content
        # 3rd send should be blocked (limit=2)
        result = await tool.execute(action="send", agent="a", message="msg3")
        assert "limit" in result.content.lower() or "error" in result.content.lower()


# ---------------------------------------------------------------------------
# Fix 3: Cancel race condition
# ---------------------------------------------------------------------------

class TestCancelRace:
    """Fix 3: start/stop lifecycle must not leave orphan adapters."""

    @pytest.mark.asyncio
    async def test_stop_during_slow_start_no_orphan(self):
        """If stop_all fires while start is awaiting adapter.start(),
        the adapter must still be stopped — no orphan process."""
        mgr = _make_sub_agent_manager()

        slow_adapter = _make_mock_adapter()

        async def slow_start(*a, **kw):
            await asyncio.sleep(0.1)

        slow_adapter.start = slow_start

        config = MagicMock()
        mgr._adapter_configs["agent-a"] = config

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter",
                    return_value=slow_adapter):
            start_task = asyncio.create_task(mgr.start("proj1", "agent-a"))
            await asyncio.sleep(0.02)  # let start begin but not finish
            await mgr.stop_all("proj1")
            start_result = await start_task

        # Adapter was stopped (stop_all found it after start registered it)
        # OR start was rejected because stopping flag was set.
        # Either way: no orphan.
        assert (
            slow_adapter.stop.called
            or "shutting down" in start_result.lower()
        )

    @pytest.mark.asyncio
    async def test_start_during_stop_all_rejected(self):
        """start() called while stop_all is running should be rejected."""
        mgr = _make_sub_agent_manager()
        adapter = _make_mock_adapter()
        _register_adapter(mgr, "proj1", "agent-a", adapter)

        async def slow_stop():
            await asyncio.sleep(0.1)
        adapter.stop = slow_stop

        stop_task = asyncio.create_task(mgr.stop_all("proj1"))
        await asyncio.sleep(0.02)  # let stop_all begin

        result = await mgr.start("proj1", "agent-b")
        await stop_task

        assert ("shutting down" in result.lower() or
                "agent-b" not in mgr._adapters.get("proj1", {}))

    @pytest.mark.asyncio
    async def test_concurrent_start_stop_no_deadlock(self):
        """start + stop_all concurrently must complete within timeout."""
        mgr = _make_sub_agent_manager()
        adapter = _make_mock_adapter()
        _register_adapter(mgr, "proj1", "agent-a", adapter)

        async def run_both():
            await asyncio.gather(
                mgr.stop_all("proj1"),
                mgr.start("proj1", "agent-b"),
            )

        await asyncio.wait_for(run_both(), timeout=5.0)


# ---------------------------------------------------------------------------
# Fix 5: Sub-agent-aware idle status
# ---------------------------------------------------------------------------

class TestIdleStatus:
    """Fix 5: idle broadcast must consider active sub-agents."""

    def _make_agent_manager(self, active_sub_agents=None):
        from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle

        ws = MagicMock()
        ws.broadcast = MagicMock()

        sub_mgr = MagicMock()
        sub_mgr.list_active = MagicMock(
            return_value=active_sub_agents or []
        )
        sub_mgr.stop = AsyncMock()
        sub_mgr.stop_all = AsyncMock()

        mgr = AgentManager.__new__(AgentManager)
        mgr._ws = ws
        mgr._sub_agent_manager = sub_mgr
        mgr._idle_poll_tasks = {}
        mgr._handles = {}
        mgr._sleep_handle = None
        mgr._platform_provider = None
        return mgr, ws, sub_mgr

    def _install_handle(self, mgr, project_id, session_stopped=False, task_done=True):
        from agent_os.daemon_v2.agent_manager import ProjectHandle
        session = MagicMock()
        session.is_stopped = MagicMock(return_value=session_stopped)
        session.pop_queued_messages = MagicMock(return_value=[])
        session._paused_for_approval = False
        handle = ProjectHandle(
            session=session,
            loop=MagicMock(),
            provider=MagicMock(),
            registry=MagicMock(),
            context_manager=MagicMock(),
            interceptor=MagicMock(),
            task=MagicMock(done=MagicMock(return_value=task_done)),
        )
        mgr._handles[project_id] = handle
        return handle

    @pytest.mark.asyncio
    async def test_idle_when_no_sub_agents(self):
        """With no active sub-agents, broadcast idle immediately."""
        mgr, ws, _ = self._make_agent_manager(active_sub_agents=[])
        self._install_handle(mgr, "proj1")

        mock_task = MagicMock()
        mock_task.exception = MagicMock(return_value=None)
        callback = mgr._on_loop_done("proj1")
        callback(mock_task)

        calls = ws.broadcast.call_args_list
        assert len(calls) >= 1
        payload = calls[-1][0][1]
        assert payload["status"] == "idle"

    @pytest.mark.asyncio
    async def test_waiting_when_sub_agents_active(self):
        """With active sub-agents, broadcast waiting instead of idle."""
        mgr, ws, _ = self._make_agent_manager(
            active_sub_agents=[{"handle": "claude-code", "status": "running"}]
        )
        self._install_handle(mgr, "proj1")

        mock_task = MagicMock()
        mock_task.exception = MagicMock(return_value=None)
        callback = mgr._on_loop_done("proj1")
        mock_future = MagicMock()
        with patch("asyncio.ensure_future", return_value=mock_future) as mock_ef:
            callback(mock_task)
            coro = mock_ef.call_args[0][0]
            coro.close()

        calls = ws.broadcast.call_args_list
        assert len(calls) >= 1
        payload = calls[-1][0][1]
        assert payload["status"] == "waiting"

    @pytest.mark.asyncio
    async def test_waiting_transitions_to_idle(self):
        """Polling loop broadcasts idle once sub-agents finish."""
        mgr, ws, sub_mgr = self._make_agent_manager()
        self._install_handle(mgr, "proj1")

        # First poll: active. Second poll: empty.
        sub_mgr.list_active = MagicMock(side_effect=[
            [{"handle": "claude-code"}],
            [],
        ])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj1")

        calls = ws.broadcast.call_args_list
        statuses = [c[0][1]["status"] for c in calls]
        assert statuses[-1] == "idle"

    @pytest.mark.asyncio
    async def test_poll_terminates_on_max_polls(self):
        """Polling must stop after max iterations, not run forever."""
        mgr, ws, sub_mgr = self._make_agent_manager(
            active_sub_agents=[{"handle": "stuck-agent"}]
        )
        self._install_handle(mgr, "proj1")

        # Override to small cap for fast test
        mgr._MAX_IDLE_POLLS = 3

        sleep_count = [0]
        async def counting_sleep(duration):
            sleep_count[0] += 1

        with patch("asyncio.sleep", side_effect=counting_sleep):
            await mgr._check_sub_agents_done("proj1")

        assert sleep_count[0] == 3
        calls = ws.broadcast.call_args_list
        final_status = calls[-1][0][1]["status"]
        assert final_status == "idle", (
            f"Polling didn't terminate — last status was '{final_status}'"
        )

    @pytest.mark.asyncio
    async def test_new_loop_stops_polling(self):
        """If a new agent loop starts, polling should exit silently."""
        mgr, ws, sub_mgr = self._make_agent_manager(
            active_sub_agents=[{"handle": "agent-a"}]
        )
        # task.done() returns False = new loop is running
        self._install_handle(mgr, "proj1", task_done=False)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj1")

        idle_calls = [
            c for c in ws.broadcast.call_args_list
            if c[0][1].get("status") == "idle"
        ]
        assert len(idle_calls) == 0, (
            "Poller broadcast idle even though a new loop is running"
        )


# ---------------------------------------------------------------------------
# Registry lifecycle hook
# ---------------------------------------------------------------------------

class TestRegistryLifecycleHook:
    """Fix 2 (supporting): ToolRegistry.reset_run_state() calls on_run_start on tools."""

    def test_reset_run_state_calls_hooks(self):
        from agent_os.agent.tools.registry import ToolRegistry

        tool_with_hook = MagicMock()
        tool_with_hook.name = "agent_message"
        tool_with_hook.on_run_start = MagicMock()

        tool_without_hook = MagicMock(spec=["name", "execute"])
        tool_without_hook.name = "read"

        registry = ToolRegistry()
        registry.register(tool_with_hook)
        registry.register(tool_without_hook)

        registry.reset_run_state()

        tool_with_hook.on_run_start.assert_called_once()
        # tool_without_hook should not cause error (no on_run_start)


# ---------------------------------------------------------------------------
# Fix 1 (is_idle): pending_response flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_idle_false_while_pending_response():
    """is_idle() must return False while transport.send() is in flight."""
    adapter = CLIAdapter(handle="test", display_name="Test")

    # Create a mock transport where send() blocks until we release it
    send_started = asyncio.Event()
    send_release = asyncio.Event()

    class MockTransport:
        def is_alive(self):
            return True
        async def send(self, msg):
            send_started.set()
            await send_release.wait()
            return "response"
        async def start(self, *args, **kwargs):
            pass
        async def stop(self):
            pass

    adapter._transport = MockTransport()

    # Before send: idle is False (initial state)
    assert adapter.is_idle() is False

    # Start send in background
    task = asyncio.create_task(adapter.send("hello"))
    await send_started.wait()

    # During send: must NOT be idle (pending_response is True)
    assert adapter.is_idle() is False

    # Release send
    send_release.set()
    await task

    # After send completes: should be idle
    assert adapter.is_idle() is True


@pytest.mark.asyncio
async def test_is_idle_true_after_response_received():
    """is_idle() returns True only after transport response is fully received."""
    adapter = CLIAdapter(handle="test", display_name="Test")

    class MockTransport:
        def is_alive(self):
            return True
        async def send(self, msg):
            return "done"
        async def start(self, *args, **kwargs):
            pass
        async def stop(self):
            pass

    adapter._transport = MockTransport()

    await adapter.send("hello")
    assert adapter.is_idle() is True
    assert adapter._pending_response is False

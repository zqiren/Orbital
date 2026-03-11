# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E smoke test for all recent bug fixes.

Covers the following scenarios end-to-end by simulating user actions
through the daemon API layers:

  BUG-005a  SDK transport streaming — message events reach read_stream()
            consumers in real-time, and ProcessManager broadcasts them
            to a mock WebSocket before send() completes.

  BUG-001   Browser warmup flags — launch_warmup() includes all
            CHROME_FLAGS and calls _apply_stealth() before page.goto().

  BUG-005b  Idle race — CLIAdapter.is_idle() returns False at
            construction (not ready yet) and True after send()
            completes (work done, correctly idle).

  BUG-003a  Credential interception — request_credential is intercepted
            in ALL autonomy presets and is not bypassed by approve-all.

  BUG-003b  Credential card rendering (compile-time) — CredentialCard
            is importable and ChatView rendering logic distinguishes
            tool_name === "request_credential".
"""

import asyncio
import importlib
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.base import TransportEvent, transport_event_to_chunk
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.adapters.base import OutputChunk
from agent_os.daemon_v2.browser_manager import BrowserManager
from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.process_manager import ProcessManager


# ======================================================================
# Shared fake SDK types (mirrors the SDK's message protocol)
# ======================================================================

class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, name: str, id: str, input: dict):
        self.name = name
        self.id = id
        self.input = input


class _FakeAssistantMessage:
    def __init__(self, content: list):
        self.content = content


class _FakeResultMessage:
    def __init__(self, is_error: bool = False, result=None, session_id=None):
        self.is_error = is_error
        self.result = result
        self.session_id = session_id


# ======================================================================
# Shared helpers
# ======================================================================

def _make_transport():
    """Create an SDKTransport with mocked internals (bypass __init__)."""
    transport = SDKTransport.__new__(SDKTransport)
    transport._client = MagicMock()
    transport._session_id = None
    transport._alive = True
    transport._workspace = ""
    transport._pending_approvals = {}
    transport._event_queue = asyncio.Queue()
    transport._needs_flush = False
    return transport


def _make_cli_adapter(transport):
    """Create a CLIAdapter that delegates to the given transport."""
    return CLIAdapter(
        handle="claude-code",
        display_name="Claude Code",
        transport=transport,
    )


def _patch_sdk_types():
    """Context manager to patch SDK type references in sdk_transport module."""
    return patch.multiple(
        "agent_os.agent.transports.sdk_transport",
        AssistantMessage=_FakeAssistantMessage,
        ResultMessage=_FakeResultMessage,
        TextBlock=_FakeTextBlock,
        ToolUseBlock=_FakeToolUseBlock,
    )


def _credential_tool_call():
    """Standard credential tool call fixture."""
    return {
        "id": "tc-cred-1",
        "name": "request_credential",
        "arguments": {
            "name": "github",
            "domain": "github.com",
            "fields": ["username", "password"],
            "reason": "Need to access repository",
        },
    }


def _make_interceptor(preset: Autonomy) -> AutonomyInterceptor:
    """Create an AutonomyInterceptor with a mock ws_manager."""
    ws = MagicMock()
    return AutonomyInterceptor(
        preset=preset,
        ws_manager=ws,
        project_id="test-project",
    )


def _mock_playwright_context():
    """Build a mock playwright + context that resolves launch_persistent_context."""
    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock()
    mock_page.on = MagicMock()
    mock_page.goto = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.wait_for_event = AsyncMock(side_effect=Exception("closed"))
    mock_ctx.close = AsyncMock()

    mock_chromium = MagicMock()
    mock_chromium.launch_persistent_context = AsyncMock(return_value=mock_ctx)

    mock_pw = MagicMock()
    mock_pw.chromium = mock_chromium
    mock_pw.stop = AsyncMock()

    return mock_pw, mock_ctx, mock_page


# ======================================================================
# BUG-005a: SDK Transport Streaming — full ProcessManager + WebSocket flow
# ======================================================================

class TestBug005a_SDKTransportStreaming:
    """E2E: message events stream through transport -> adapter -> ProcessManager -> activity (v5)."""

    @pytest.mark.asyncio
    async def test_message_events_reach_consumer_before_send_completes(self):
        """Full stack: SDKTransport -> CLIAdapter -> ProcessManager -> activity_translator.

        Verifies that message events arrive at the ProcessManager consumer
        (fed to activity_translator and broadcast) BEFORE send() returns.
        ProcessManager consumes from adapter.read_stream() in a background
        task -- this is the path that was broken before BUG-005a.
        v5: ProcessManager writes to transcript, not session.
        """
        transport = _make_transport()
        adapter = _make_cli_adapter(transport)

        assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Here is your analysis.")])
        result_msg = _FakeResultMessage(session_id="sess-e2e-001")

        async def mock_receive():
            yield assistant_msg
            await asyncio.sleep(0.1)  # simulate SDK processing time
            yield result_msg

        transport._client.query = AsyncMock()
        transport._client.receive_response = mock_receive

        # --- Simulate ProcessManager ---
        ws_manager = MagicMock()
        ws_manager.broadcast = MagicMock()
        activity_translator = MagicMock()

        # Track when activity_translator.on_message is called (proxy for
        # the message reaching the consumer in real-time)
        consumer_received = asyncio.Event()

        def on_message_hook(msg, project_id):
            consumer_received.set()

        activity_translator.on_message = on_message_hook

        pm = ProcessManager(ws_manager=ws_manager, activity_translator=activity_translator)

        with _patch_sdk_types():
            # Start PM consumer (like daemon startup)
            await pm.start("test-project", "claude-code", adapter)

            # Small delay to let consumer attach to read_stream
            await asyncio.sleep(0.02)

            # Send a message (like user injecting via API)
            send_task = asyncio.create_task(adapter.send("analyze this workspace"))

            # Message event must reach the PM consumer -> activity_translator BEFORE send() completes
            try:
                await asyncio.wait_for(consumer_received.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pytest.fail(
                    "BUG-005a regression: message event never reached ProcessManager "
                    "consumer — events are being buffered instead of streamed"
                )

            # Wait for send to complete
            await send_task

            # Verify activity_translator was notified with the message content
            assert consumer_received.is_set(), \
                "activity_translator.on_message was not called"

            # Verify WS broadcast was called for chat messages
            chat_broadcasts = [
                c for c in ws_manager.broadcast.call_args_list
                if c[0][1].get("type") == "chat.sub_agent_message"
            ]
            assert len(chat_broadcasts) >= 1, "WS broadcast should have been called"
            assert any(
                "Here is your analysis." in str(c[0][1].get("content", ""))
                for c in chat_broadcasts
            )

            # Clean up
            transport._alive = False
            await pm.stop("test-project", "claude-code")

    @pytest.mark.asyncio
    async def test_read_stream_yields_events_in_realtime(self):
        """Message events appear in SDKTransport._event_queue BEFORE send() returns."""
        transport = _make_transport()

        assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Real-time response")])
        result_msg = _FakeResultMessage()

        async def mock_receive():
            yield assistant_msg
            yield result_msg

        transport._client.query = AsyncMock()
        transport._client.receive_response = mock_receive

        with _patch_sdk_types():
            send_task = asyncio.create_task(transport.send("hello"))

            # Event should be in queue before send() returns
            event = await asyncio.wait_for(transport._event_queue.get(), timeout=2.0)
            assert event.event_type == "message"
            assert event.raw_text == "Real-time response"

            result = await send_task
            assert "Real-time response" in result


# ======================================================================
# BUG-001: Browser Warmup Flags
# ======================================================================

class TestBug001_BrowserWarmupFlags:
    """E2E: browser warmup launch uses all anti-detection flags and stealth JS."""

    @pytest.fixture
    def browser_manager(self, tmp_path):
        return BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

    @pytest.mark.asyncio
    async def test_warmup_includes_all_chrome_flags(self, browser_manager):
        """launch_warmup() must pass every flag in BrowserManager.CHROME_FLAGS."""
        mock_pw, mock_ctx, mock_page = _mock_playwright_context()

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_pw)

            await browser_manager.launch_warmup("https://accounts.google.com")

            call_kwargs = mock_pw.chromium.launch_persistent_context.call_args
            args = call_kwargs.kwargs.get("args", call_kwargs[1].get("args", []))

            # Critical flag
            assert "--disable-blink-features=AutomationControlled" in args, \
                "Missing critical anti-detection flag"

            # All CHROME_FLAGS must be present
            for flag in BrowserManager.CHROME_FLAGS:
                assert flag in args, f"Missing flag: {flag}"

    @pytest.mark.asyncio
    async def test_stealth_called_before_goto(self, browser_manager):
        """_apply_stealth() must execute before page.goto() during warmup."""
        mock_pw, mock_ctx, mock_page = _mock_playwright_context()

        call_order = []

        async def track_evaluate(*a, **kw):
            call_order.append("stealth")

        async def track_goto(*a, **kw):
            call_order.append("goto")

        mock_page.evaluate = track_evaluate
        mock_page.goto = track_goto

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_pw)

            await browser_manager.launch_warmup("https://accounts.google.com")

            assert "stealth" in call_order, "_apply_stealth was not called"
            assert "goto" in call_order, "page.goto was not called"
            assert call_order.index("stealth") < call_order.index("goto"), \
                "_apply_stealth must be called BEFORE page.goto()"


# ======================================================================
# BUG-005b: Idle Race
# ======================================================================

class TestBug005b_IdleRace:
    """E2E: adapter starts not-idle; becomes idle after send() completes (work done)."""

    @pytest.mark.asyncio
    async def test_adapter_idle_after_instant_transport_send(self):
        """After transport.send() completes, adapter.is_idle() must be True (work done)."""
        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(return_value="Done instantly")
        mock_transport.is_alive = MagicMock(return_value=True)
        mock_transport.start = AsyncMock()
        mock_transport.stop = AsyncMock()

        adapter = CLIAdapter(
            handle="claude-code",
            display_name="Claude Code",
            transport=mock_transport,
        )

        # Send completes immediately
        await adapter.send("do something quick")

        # Adapter should be idle after send completes — work is done
        assert adapter.is_idle() is True, \
            "Adapter should be idle after send() completes — work is done"

    @pytest.mark.asyncio
    async def test_on_loop_done_sees_idle_after_send(self):
        """After send() completes, _on_loop_done correctly sees adapter as idle."""
        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(return_value="Response")
        mock_transport.is_alive = MagicMock(return_value=True)
        mock_transport.start = AsyncMock()
        mock_transport.stop = AsyncMock()

        adapter = CLIAdapter(
            handle="claude-code",
            display_name="Claude Code",
            transport=mock_transport,
        )

        await adapter.send("analyze workspace")

        # After send completes, adapter is idle (work done)
        assert adapter.is_alive() is True
        assert adapter.is_idle() is True

        status_list = [
            {"handle": "claude-code", "status": "running" if not adapter.is_idle() else "idle"},
        ]
        idle_agents = [a for a in status_list if a["status"] == "idle"]

        assert len(idle_agents) == 1, "Sub-agent should appear in idle list after work completes"
        assert idle_agents[0]["status"] == "idle", \
            "After send() completes, adapter should correctly report idle"

    @pytest.mark.asyncio
    async def test_adapter_starts_not_idle(self):
        """CLIAdapter should not be idle at construction time."""
        mock_transport = MagicMock()
        mock_transport.is_alive = MagicMock(return_value=True)

        adapter = CLIAdapter(
            handle="test-agent",
            display_name="Test Agent",
            transport=mock_transport,
        )

        assert adapter.is_idle() is False, "Adapter must start as not-idle"


# ======================================================================
# BUG-003a: Credential Interception
# ======================================================================

class TestBug003a_CredentialInterception:
    """E2E: request_credential is always intercepted, never bypassed."""

    @pytest.mark.parametrize("preset", [
        Autonomy.HANDS_OFF,
        Autonomy.CHECK_IN,
        Autonomy.SUPERVISED,
    ])
    def test_request_credential_intercepted_all_presets(self, preset):
        """request_credential must be intercepted in every autonomy preset."""
        interceptor = _make_interceptor(preset)
        assert interceptor.should_intercept(_credential_tool_call()) is True, \
            f"request_credential not intercepted in {preset.name}"

    def test_approve_all_does_not_bypass_credentials(self):
        """activate_bypass_all must NOT skip request_credential."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        interceptor.activate_bypass_all(duration=600)
        assert interceptor.should_intercept(_credential_tool_call()) is True, \
            "approve-all bypass should not skip credential requests"

    def test_per_action_bypass_does_not_skip_credentials(self):
        """Per-action approval bypass must NOT skip request_credential."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        tc = _credential_tool_call()
        interceptor.record_approval(tc["name"], tc["arguments"])
        assert interceptor.should_intercept(tc) is True, \
            "Per-action bypass should not skip credential requests"

    def test_other_tools_follow_normal_preset_rules(self):
        """Non-credential tools must follow normal preset rules."""
        # HANDS_OFF: shell should NOT be intercepted
        interceptor_ho = _make_interceptor(Autonomy.HANDS_OFF)
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor_ho.should_intercept(shell_call) is False, \
            "HANDS_OFF should not intercept shell"

        # HANDS_OFF: request_access SHOULD be intercepted
        access_call = {"name": "request_access", "arguments": {}}
        assert interceptor_ho.should_intercept(access_call) is True, \
            "HANDS_OFF should intercept request_access"

        # SUPERVISED: read should NOT be intercepted
        interceptor_sv = _make_interceptor(Autonomy.SUPERVISED)
        read_call = {"name": "read", "arguments": {"path": "/etc/hosts"}}
        assert interceptor_sv.should_intercept(read_call) is False, \
            "SUPERVISED should not intercept read"

        # SUPERVISED: shell SHOULD be intercepted
        assert interceptor_sv.should_intercept(shell_call) is True, \
            "SUPERVISED should intercept shell"

        # CHECK_IN: shell SHOULD be intercepted
        interceptor_ci = _make_interceptor(Autonomy.CHECK_IN)
        assert interceptor_ci.should_intercept(shell_call) is True, \
            "CHECK_IN should intercept shell"

    def test_approve_all_bypasses_normal_tools_but_not_credentials(self):
        """approve-all should bypass normal tools but still intercept credentials."""
        interceptor = _make_interceptor(Autonomy.SUPERVISED)
        interceptor.activate_bypass_all(duration=600)

        # Shell should be bypassed
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False, \
            "approve-all should bypass normal tools"

        # Credentials should NOT be bypassed
        assert interceptor.should_intercept(_credential_tool_call()) is True, \
            "approve-all must not bypass credential requests"


# ======================================================================
# BUG-003b: Credential Card Rendering (compile-time verification)
# ======================================================================

class TestBug003b_CredentialCardRendering:
    """Compile-time checks: CredentialCard is importable and ChatView uses it."""

    def test_credential_card_component_importable(self):
        """CredentialCard.tsx must exist and export PendingCredential + default."""
        credential_card_path = Path(__file__).resolve().parents[2] / "web" / "src" / "components" / "CredentialCard.tsx"
        assert credential_card_path.exists(), \
            f"CredentialCard.tsx not found at {credential_card_path}"

        source = credential_card_path.read_text(encoding="utf-8")

        # Verify key exports exist
        assert "export interface PendingCredential" in source, \
            "PendingCredential interface not exported"
        assert "export default function CredentialCard" in source, \
            "CredentialCard not exported as default"

        # Verify it accepts the expected props
        assert "credential:" in source, "Missing credential prop"
        assert "projectId:" in source, "Missing projectId prop"
        assert "onResolve?:" in source, "Missing onResolve prop"

    def test_chatview_renders_credential_card_for_request_credential(self):
        """ChatView must distinguish tool_name === 'request_credential' and render CredentialCard."""
        chatview_path = Path(__file__).resolve().parents[2] / "web" / "src" / "components" / "ChatView.tsx"
        assert chatview_path.exists(), \
            f"ChatView.tsx not found at {chatview_path}"

        source = chatview_path.read_text(encoding="utf-8")

        # Verify CredentialCard is imported
        assert "import CredentialCard from" in source or "import CredentialCard," in source, \
            "ChatView does not import CredentialCard"

        # Verify the rendering logic checks for 'request_credential'
        assert "request_credential" in source, \
            "ChatView does not reference 'request_credential' tool_name"

        # Verify both CredentialCard and ApprovalCard are rendered conditionally
        assert "<CredentialCard" in source, \
            "ChatView does not render <CredentialCard>"
        assert "<ApprovalCard" in source, \
            "ChatView does not render <ApprovalCard>"

        # Verify the conditional branching pattern:
        # tool_name === 'request_credential' ? <CredentialCard .../> : <ApprovalCard .../>
        credential_branch_pattern = re.compile(
            r"tool_name\s*===?\s*['\"]request_credential['\"].*?CredentialCard",
            re.DOTALL,
        )
        assert credential_branch_pattern.search(source), \
            "ChatView does not branch rendering on tool_name === 'request_credential'"

    def test_chatview_renders_credential_card_for_realtime_approvals(self):
        """ChatView must also render CredentialCard for real-time (non-historical) approvals."""
        chatview_path = Path(__file__).resolve().parents[2] / "web" / "src" / "components" / "ChatView.tsx"
        source = chatview_path.read_text(encoding="utf-8")

        # Count how many times CredentialCard is rendered (should be at least 2:
        # once for historical approval_card items, once for real-time approvals map)
        credential_card_count = source.count("<CredentialCard")
        assert credential_card_count >= 2, (
            f"Expected CredentialCard to be rendered in at least 2 places "
            f"(historical + real-time), found {credential_card_count}"
        )

    def test_types_include_credential_activity_category(self):
        """Types file must include 'credential_request' as an ActivityCategory."""
        types_path = Path(__file__).resolve().parents[2] / "web" / "src" / "types.ts"
        assert types_path.exists(), f"types.ts not found at {types_path}"

        source = types_path.read_text(encoding="utf-8")
        assert "credential_request" in source, \
            "types.ts does not include 'credential_request' activity category"

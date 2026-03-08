# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: approval/denial history corruption in check_in mode.

Bug: In check_in mode, when a user message is injected while a tool call is
pending approval, the message appears between the assistant's tool_calls and
one of its tool results in the session JSONL.  The context manager's
_validate_tool_results Pass 2 then drops the tool result as "orphaned"
(valid_tool_ids was reset on the user message), causing strict OpenAI-compatible
providers (Kimi K2.5, etc.) to reject the malformed history with HTTP 400:

    "assistant message with 'tool_calls' must be followed by tool messages"

Fix A: _validate_tool_results Pass 2 no longer resets valid_tool_ids on
user/system messages.  valid_tool_ids is only reset when a new assistant
message with tool_calls is encountered, matching Pass 1's look-ahead behavior.

Fix B: _on_loop_done checks _paused_for_approval BEFORE draining the queue.
inject_message queues instead of direct-appending when paused for approval.

See evidence/approval-denial-investigation.md for full investigation.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.context import ContextManager
from agent_os.daemon_v2.agent_manager import AgentManager


# ---------------------------------------------------------------------------
# Fix A: _validate_tool_results — tool results after user messages
# ---------------------------------------------------------------------------


class TestToolResultAfterUserMessage:
    """Pass 2 must NOT strip tool results separated from their assistant
    message by an intervening user/system message."""

    def test_tool_result_after_user_message_preserved(self):
        """The core bug: tool result for write:16 appears after a user message
        but belongs to the preceding assistant's tool_calls.  Must be kept."""
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "edit", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            {"role": "user", "content": "next task"},  # interleaved user message
            {"role": "tool", "tool_call_id": "tc_2", "content": "ok"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
        assert "tc_1" in tool_ids
        assert "tc_2" in tool_ids, "tc_2 must NOT be stripped by Pass 2"

    def test_tool_result_after_system_message_preserved(self):
        """System messages (e.g., error notes) should not break the pairing."""
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "file content"},
            {"role": "system", "content": "Budget warning: 80% used"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "written"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
        assert "tc_2" in tool_ids

    def test_tool_result_after_text_only_assistant_preserved(self):
        """A text-only assistant message between tool_calls and result
        should not break the pairing."""
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "done"},
            {"role": "assistant", "content": "Processing..."},
            {"role": "tool", "tool_call_id": "tc_2", "content": "DENIED by user"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
        assert "tc_2" in tool_ids

    def test_duplicate_result_after_user_message_still_stripped(self):
        """A DUPLICATE tool result (same tc_id already resolved) after a user
        message must still be dropped — it's a true orphan."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            {"role": "user", "content": "next question"},
            {"role": "tool", "tool_call_id": "tc_1", "content": "CANCELLED: duplicate"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "ok"

    def test_orphan_after_new_assistant_batch_still_stripped(self):
        """Tool result from batch 1 appearing after batch 2 is a true orphan."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}, {"id": "tc_2"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            # tc_2 missing — Pass 1 injects synthetic
            {"role": "assistant", "tool_calls": [{"id": "tc_3"}]},
            {"role": "tool", "tool_call_id": "tc_3", "content": "ok"},
            # tc_2 appears here — truly orphaned (after new batch)
            {"role": "tool", "tool_call_id": "tc_2", "content": "CANCELLED"},
        ]
        result = ContextManager._validate_tool_results(messages)
        # tc_2 should have a synthetic result (from Pass 1), the late CANCELLED dropped
        tc2_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "tc_2"]
        assert len(tc2_results) == 1
        assert "CANCELLED" not in tc2_results[0]["content"]

    def test_batch_denied_and_cancelled_with_interleaved_user(self):
        """Realistic check_in scenario: [edit, write] batch, edit auto-approved,
        write intercepted and denied, user message injected in between."""
        messages = [
            {"role": "user", "content": "Create a file and update the index"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_edit", "type": "function", "function": {"name": "edit", "arguments": "{}"}},
                {"id": "tc_write", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc_edit", "content": '{"status": "success"}'},
            {"role": "user", "content": "Actually, create press-kit.md instead"},
            {"role": "tool", "tool_call_id": "tc_write", "content": "DENIED by user. Reason: wrong file"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
        assert "tc_edit" in tool_ids
        assert "tc_write" in tool_ids, "DENIED result must be preserved"
        assert len(result) == 5, "No messages should be dropped"


# ---------------------------------------------------------------------------
# Fix B: inject_message queues during pending approval
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create an AgentManager with minimal mocks."""
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr


class TestInjectMessageDuringApproval:
    """inject_message must queue (not direct-append) when paused for approval."""

    @pytest.mark.asyncio
    async def test_inject_queues_when_paused_for_approval(self, manager):
        """When session._paused_for_approval is True, inject_message must
        queue the message instead of appending it directly."""
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = True
        session.queue_message = MagicMock()
        session.append = MagicMock()

        handle = MagicMock()
        handle.session = session
        handle.interceptor = MagicMock()

        task_mock = MagicMock()
        task_mock.done.return_value = True  # loop is done
        task_mock.exception.return_value = None
        handle.task = task_mock

        manager._handles["proj_test"] = handle

        result = await manager.inject_message("proj_test", "next task please")

        assert result == "queued"
        session.queue_message.assert_called_once_with("next task please", nonce=None)
        session.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_direct_appends_when_not_paused(self, manager):
        """Normal case: no approval pause → direct append and resume."""
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = False
        session.queue_message = MagicMock()
        session.append = MagicMock()
        session.pop_queued_messages = MagicMock(return_value=[])

        handle = MagicMock()
        handle.session = session
        handle.interceptor = MagicMock()

        task_mock = MagicMock()
        task_mock.done.return_value = True
        task_mock.exception.return_value = None
        handle.task = task_mock

        manager._handles["proj_test"] = handle
        manager._start_loop = AsyncMock()

        result = await manager.inject_message("proj_test", "hello")

        assert result == "delivered"
        session.append.assert_called()
        appended = session.append.call_args[0][0]
        assert appended["role"] == "user"
        assert appended["content"] == "hello"


class TestOnLoopDoneApprovalBeforeQueueDrain:
    """_on_loop_done must check _paused_for_approval BEFORE draining the queue."""

    def test_queued_messages_not_drained_when_paused_for_approval(self, manager):
        """If paused for approval, queued messages stay in the queue."""
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = True
        session.pop_queued_messages = MagicMock(return_value=["pending msg"])
        session.append = MagicMock()

        handle = MagicMock()
        handle.session = session

        task = MagicMock()
        task.exception.return_value = None
        handle.task = task

        manager._handles["proj_test"] = handle

        callback = manager._on_loop_done("proj_test")
        callback(task)

        # pop_queued_messages should NOT have been called — the approval
        # check returns early before reaching the queue drain.
        session.pop_queued_messages.assert_not_called()

        # Should broadcast pending_approval, not idle
        manager._ws.broadcast.assert_called()
        payload = manager._ws.broadcast.call_args[0][1]
        assert payload["status"] == "pending_approval"

    def test_queued_messages_drained_after_approval_resolved(self, manager):
        """After approval resolves (not paused), queue is drained normally."""
        session = MagicMock()
        session.is_stopped.return_value = False
        session._paused_for_approval = False
        session.pop_queued_messages.return_value = ["queued msg"]
        session.append = MagicMock()

        handle = MagicMock()
        handle.session = session

        task = MagicMock()
        task.exception.return_value = None
        handle.task = task

        manager._handles["proj_test"] = handle

        callback = manager._on_loop_done("proj_test")
        with patch("asyncio.ensure_future") as mock_ef:
            callback(task)
            if mock_ef.call_args:
                coro = mock_ef.call_args[0][0]
                coro.close()

        session.pop_queued_messages.assert_called_once()
        session.append.assert_called()
        appended = session.append.call_args[0][0]
        assert appended["content"] == "queued msg"


# ---------------------------------------------------------------------------
# Integration: check_in multi-tool batch with interleaved user message
# ---------------------------------------------------------------------------


class TestCheckInMultiToolDenialIntegration:
    """End-to-end scenario: check_in mode, [edit, write] batch, write
    intercepted, user message injected while pending, deny, next LLM call
    must see a valid history."""

    def test_context_valid_after_denial_with_interleaved_user_message(self):
        """Simulates the full JSONL sequence from a check_in denial with
        an interleaved user message, and verifies the context manager
        produces a valid history for the LLM."""
        # This is the exact message sequence from the P-20 bug:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Update the index and create a new file"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_edit_15", "type": "function",
                 "function": {"name": "edit", "arguments": '{"path": "launch-tasks.md"}'}},
                {"id": "tc_write_16", "type": "function",
                 "function": {"name": "write", "arguments": '{"path": "PROJECT_STATE.md"}'}},
            ]},
            # edit auto-approved, result appended immediately
            {"role": "tool", "tool_call_id": "tc_edit_15",
             "content": '{"path": "launch-tasks.md", "status": "success"}'},
            # User message injected while write was pending approval
            {"role": "user", "content": "Create launch/press-kit.md with media assets"},
            # write denied (or approved) — result appended after user message
            {"role": "tool", "tool_call_id": "tc_write_16",
             "content": "DENIED by user. Reason: wrong file target"},
            # Agent acknowledges denial
            {"role": "assistant",
             "content": "I understand. I won't create that file. Let me work on press-kit.md instead."},
        ]

        result = ContextManager._validate_tool_results(messages)

        # Verify ALL tool_calls have matching results
        for msg in result:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc_ids = {tc["id"] for tc in msg["tool_calls"]}
                # Collect all tool results in the entire conversation
                result_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
                missing = tc_ids - result_ids
                assert not missing, (
                    f"tool_calls {missing} have no matching tool results — "
                    f"strict providers (Kimi, OpenAI) will reject this with HTTP 400"
                )

        # Verify message count — nothing dropped
        assert len(result) == len(messages), (
            f"Expected {len(messages)} messages, got {len(result)} — "
            f"Pass 2 incorrectly stripped a message"
        )

    def test_context_valid_with_cancelled_siblings_and_denial(self):
        """Batch [write_A, write_B]: write_A intercepted, write_B cancelled,
        then write_A denied.  CANCELLED appears before DENIED."""
        messages = [
            {"role": "user", "content": "Create two files"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_w1", "type": "function",
                 "function": {"name": "write", "arguments": '{"path": "a.txt"}'}},
                {"id": "tc_w2", "type": "function",
                 "function": {"name": "write", "arguments": '{"path": "b.txt"}'}},
            ]},
            # write_B cancelled (sibling after intercepted write_A)
            {"role": "tool", "tool_call_id": "tc_w2",
             "content": "CANCELLED: This tool call was not executed."},
            # User message injected during approval wait
            {"role": "user", "content": "Nevermind about those files"},
            # write_A denied after user message
            {"role": "tool", "tool_call_id": "tc_w1",
             "content": "DENIED by user. Reason: user cancelled"},
        ]

        result = ContextManager._validate_tool_results(messages)
        tool_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
        assert "tc_w1" in tool_ids, "DENIED result must be preserved"
        assert "tc_w2" in tool_ids, "CANCELLED result must be preserved"
        assert len(result) == len(messages)

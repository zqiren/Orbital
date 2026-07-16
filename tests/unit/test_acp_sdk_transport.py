# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the typed official-SDK ACP transport."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from acp import RequestError
from acp.schema import (
    AgentCapabilities,
    ConfigOptionUpdate,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    SessionCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    ToolCallUpdate,
)
from unittest.mock import AsyncMock

from agent_os.agent.transports.acp_sdk_transport import (
    ACPSDKTransport,
    _interaction_response,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "dummy_acp_sdk_agent.py"


async def _start(tmp_path: Path, **kwargs) -> ACPSDKTransport:
    transport = ACPSDKTransport(**kwargs)
    await transport.start(sys.executable, [str(FIXTURE)], str(tmp_path))
    return transport


def _drain(transport: ACPSDKTransport):
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


async def _next_event(transport: ACPSDKTransport, event_type: str):
    while True:
        event = await asyncio.wait_for(transport._event_queue.get(), timeout=5)
        if event.event_type == event_type:
            return event


def test_internal_flags_are_stripped_and_cursor_model_is_global():
    args, permission_mode, model = ACPSDKTransport._extract_internal_args(
        ["acp", "--model", "grok-4.5", "--orbital-permission-mode", "ask"]
    )
    assert args == ["--model", "grok-4.5", "acp"]
    assert permission_mode == "ask"
    assert model == "grok-4.5"


@pytest.mark.asyncio
async def test_cursor_cli_alias_is_not_resent_over_config_after_canonicalization():
    transport = ACPSDKTransport(model="cursor-grok-4.5-low")
    transport._session_id = "session-1"
    transport._connection = AsyncMock()
    response = NewSessionResponse(
        session_id="session-1",
        config_options=[
            SessionConfigOptionSelect(
                type="select",
                id="model",
                name="Model",
                current_value="grok-4.5[effort=high,fast=true]",
                options=[
                    SessionConfigSelectOption(
                        value="grok-4.5[effort=high,fast=true]", name="Grok 4.5 Fast"
                    )
                ],
            )
        ],
    )
    transport._remember_session_configuration(response)
    await transport._apply_desired_configuration(response)
    transport._connection.set_config_option.assert_not_awaited()
    assert (
        transport._effective_config_value("model")
        == "grok-4.5[effort=high,fast=true]"
    )


@pytest.mark.asyncio
async def test_missing_resume_session_falls_back_fresh_but_reports_reason(tmp_path):
    transport = ACPSDKTransport(resume_record={"session_id": "missing"})
    transport._workspace = str(tmp_path)
    transport._connection = AsyncMock()
    transport._connection.initialize.return_value = InitializeResponse(
        protocol_version=1,
        agent_capabilities=AgentCapabilities(
            load_session=True, session_capabilities=SessionCapabilities()
        ),
    )
    transport._connection.load_session.side_effect = RequestError.resource_not_found(
        "missing"
    )
    transport._connection.new_session.return_value = NewSessionResponse(
        session_id="fresh", config_options=[]
    )
    await transport._initialize_and_open_session()
    assert transport.session_id == "fresh"
    assert transport.resume_outcome == ("fresh", "resume_failed")
    assert transport._resume_session_id is None


@pytest.mark.asyncio
async def test_generic_resume_error_fails_start_instead_of_claiming_fresh(tmp_path):
    transport = ACPSDKTransport(resume_record={"session_id": "existing"})
    transport._workspace = str(tmp_path)
    transport._connection = AsyncMock()
    transport._connection.initialize.return_value = InitializeResponse(
        protocol_version=1,
        agent_capabilities=AgentCapabilities(
            load_session=True, session_capabilities=SessionCapabilities()
        ),
    )
    transport._connection.load_session.side_effect = RequestError.auth_required()
    with pytest.raises(RequestError, match="Authentication required"):
        await transport._initialize_and_open_session()
    transport._connection.new_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_session_updates_do_not_emit_phantom_turn_chunks():
    transport = ACPSDKTransport()
    transport._session_id = "session-1"
    update = ConfigOptionUpdate(
        session_update="config_option_update",
        config_options=[
            SessionConfigOptionSelect(
                type="select",
                id="model",
                name="Model",
                current_value="grok-4.5",
                options=[SessionConfigSelectOption(value="grok-4.5", name="Grok")],
            )
        ],
    )
    await transport.session_update("session-1", update)
    assert transport._event_queue.empty()
    assert transport.session_config_options["model"]["currentValue"] == "grok-4.5"


@pytest.mark.asyncio
async def test_official_sdk_initialize_config_and_prompt_boundary(tmp_path):
    transport = await _start(tmp_path, model="grok-4.5", mode="plan")
    try:
        startup = _drain(transport)
        assert [event.event_type for event in startup] == ["thread_started"]
        assert transport.session_id == "dummy-session"
        assert transport.resume_outcome == ("fresh", None)
        assert transport.session_config_options["model"]["currentValue"] == "grok-4.5"

        await transport.dispatch("hello")
        message = await _next_event(transport, "message")
        boundary = await _next_event(transport, "turn_complete")
        assert "Echo: hello" in message.raw_text
        assert "model=grok-4.5" in message.raw_text
        assert boundary.data == {
            "cause": "success",
            "stop_reason": "end_turn",
            "session_id": "dummy-session",
            "model": "grok-4.5",
        }
    finally:
        await transport.stop()
    assert not transport.is_alive()


@pytest.mark.asyncio
async def test_provider_confirmed_resume_uses_advertised_resume(tmp_path):
    transport = await _start(
        tmp_path, resume_record={"session_id": "existing-session"}
    )
    try:
        assert transport.session_id == "existing-session"
        assert transport.resume_outcome == ("resumed", None)
        assert transport._resume_session_id == "existing-session"
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_auto_permission_selects_allow_once_not_persistent_grant(tmp_path):
    transport = await _start(tmp_path, permission_mode="auto")
    _drain(transport)
    try:
        result = await transport.send("permission")
        outcome = json.loads(result)
        assert outcome == {"outcome": {"optionId": "allow-once", "outcome": "selected"}}
        assert not transport._pending_approvals
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_auto_permission_never_falls_back_to_allow_always():
    transport = ACPSDKTransport(permission_mode="auto")
    response = await transport.request_permission(
        "session",
        ToolCallUpdate(tool_call_id="tool", title="Tool"),
        [
            PermissionOption(
                option_id="allow-always", name="Always", kind="allow_always"
            )
        ],
    )
    assert response.outcome.outcome == "cancelled"


@pytest.mark.asyncio
async def test_boolean_deny_never_falls_back_to_reject_always():
    transport = ACPSDKTransport(permission_mode="ask")
    request = asyncio.create_task(
        transport.request_permission(
            "session",
            ToolCallUpdate(tool_call_id="tool", title="Tool"),
            [
                PermissionOption(
                    option_id="reject-always", name="Always reject", kind="reject_always"
                )
            ],
        )
    )
    event = await _next_event(transport, "permission_request")
    await transport.respond_to_permission(event.data["request_id"], False)
    response = await request
    assert response.outcome.outcome == "cancelled"


@pytest.mark.asyncio
async def test_ask_permission_blocks_until_selected_option(tmp_path):
    transport = await _start(tmp_path, permission_mode="ask")
    _drain(transport)
    try:
        await transport.dispatch("permission")
        event = await _next_event(transport, "permission_request")
        assert event.data["tool_name"] == "Run command"
        assert [item["optionId"] for item in event.data["options"]] == [
            "allow-once",
            "allow-always",
            "reject-once",
        ]
        await transport.respond_to_permission_response(
            event.data["request_id"],
            approved=True,
            reply_text=None,
            temporary_allow_s=600,
            decision="allow-once",
        )
        message = await _next_event(transport, "message")
        assert "allow-once" in message.raw_text
        assert (await _next_event(transport, "turn_complete")).data["cause"] == "success"
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["cancel", "cancelled", "deny-stop"])
async def test_rich_permission_cancel_vocabulary_cancels_request(tmp_path, decision):
    transport = await _start(tmp_path, permission_mode="ask")
    _drain(transport)
    try:
        await transport.dispatch("permission")
        event = await _next_event(transport, "permission_request")
        await transport.respond_to_permission_response(
            event.data["request_id"],
            approved=False,
            reply_text=None,
            temporary_allow_s=None,
            decision=decision,
        )
        message = await _next_event(transport, "message")
        assert '"outcome": "cancelled"' in message.raw_text
        await _next_event(transport, "turn_complete")
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_cursor_question_extension_is_provider_neutral_interaction(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("question")
        event = await _next_event(transport, "interaction_required")
        assert event.data["kind"] == "cursor/ask_question"
        assert event.data["questions"][0]["options"][1]["label"] == "简体中文"
        await transport.resolve_interaction(
            event.data["interaction_id"], selection={"language": "zh"}
        )
        message = await _next_event(transport, "message")
        assert "selectedOptionIds" in message.raw_text
        assert "zh" in message.raw_text
        await _next_event(transport, "turn_complete")
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_cursor_option_question_does_not_encode_text_as_option_id(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("question")
        event = await _next_event(transport, "interaction_required")
        await transport.respond_to_interaction(
            event.data["interaction_id"], text="please choose Chinese"
        )
        message = await _next_event(transport, "message")
        decoded = json.loads(message.raw_text)
        assert decoded == {
            "outcome": {
                "outcome": "skipped",
                "reason": "Option selection required",
            }
        }
        assert "selectedOptionIds" not in message.raw_text
        await _next_event(transport, "turn_complete")
    finally:
        await transport.stop()


def test_explicit_free_text_question_uses_text_answer_shape():
    data = {
        "kind": "cursor/ask_question",
        "params": {
            "questions": [
                {"id": "details", "prompt": "Details?", "answerType": "text"}
            ]
        },
    }
    assert _interaction_response(
        data, text="some guidance", selection=None, accepted=None
    ) == {
        "outcome": {
            "outcome": "answered",
            "answers": [{"questionId": "details", "text": "some guidance"}],
        }
    }


@pytest.mark.asyncio
async def test_cursor_multi_selection_list_targets_first_question(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("question")
        event = await _next_event(transport, "interaction_required")
        await transport.respond_to_interaction(
            event.data["interaction_id"], selection=["en", "zh"]
        )
        message = await _next_event(transport, "message")
        decoded = json.loads(message.raw_text)
        assert decoded["outcome"]["answers"] == [
            {"questionId": "language", "selectedOptionIds": ["en", "zh"]}
        ]
        await _next_event(transport, "turn_complete")
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_cursor_plan_extension_accepts_exact_response(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("plan")
        event = await _next_event(transport, "interaction_required")
        assert event.data["kind"] == "cursor/create_plan"
        await transport.respond_to_interaction(
            event.data["interaction_id"],
            response={"outcome": {"outcome": "accepted"}},
        )
        message = await _next_event(transport, "message")
        assert "accepted" in message.raw_text
        await _next_event(transport, "turn_complete")
    finally:
        await transport.stop()


def test_plan_text_is_not_encoded_as_plan_uri_and_rejection_uses_reason():
    data = {"kind": "cursor/create_plan", "params": {}}
    accepted = _interaction_response(
        data, text="revise phase two", selection=None, accepted=None
    )
    assert accepted == {"outcome": {"outcome": "accepted"}}
    assert "planUri" not in accepted["outcome"]
    rejected = _interaction_response(
        data, text="scope is too broad", selection="reject", accepted=None
    )
    assert rejected == {
        "outcome": {"outcome": "rejected", "reason": "scope is too broad"}
    }


@pytest.mark.asyncio
async def test_multiple_agent_message_deltas_emit_one_complete_turn_message(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("multi-delta")
        message = await _next_event(transport, "message")
        boundary = await _next_event(transport, "turn_complete")
        assert message.raw_text == "first second third"
        assert message.data == {"text": "first second third", "complete": True}
        assert boundary.data["stop_reason"] == "end_turn"
        assert not any(event.event_type == "message" for event in _drain(transport))
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_cancel_reaches_genuine_cancelled_prompt_response(tmp_path):
    transport = await _start(tmp_path)
    _drain(transport)
    try:
        await transport.dispatch("wait")
        await transport.cancel()
        boundary = await _next_event(transport, "turn_complete")
        assert boundary.data["stop_reason"] == "cancelled"
        assert boundary.data["cause"] == "interrupted"
    finally:
        await transport.stop()

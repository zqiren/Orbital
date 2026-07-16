#!/usr/bin/env python3
"""Official-SDK ACP agent fixture for ACPSDKTransport tests."""

from __future__ import annotations

import asyncio
import json

from acp import PROTOCOL_VERSION, run_agent, text_block
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    RequestPermissionResponse,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    ToolCallUpdate,
)


def _config_options(
    model: str = "cursor-small", mode: str = "agent"
) -> list[SessionConfigOptionSelect]:
    return [
        SessionConfigOptionSelect(
            type="select",
            id="model",
            name="Model",
            current_value=model,
            options=[
                SessionConfigSelectOption(value="cursor-small", name="Cursor Small"),
                SessionConfigSelectOption(value="grok-4.5", name="Grok 4.5"),
            ],
        ),
        SessionConfigOptionSelect(
            type="select",
            id="mode",
            name="Mode",
            current_value=mode,
            options=[
                SessionConfigSelectOption(value="agent", name="Agent"),
                SessionConfigSelectOption(value="plan", name="Plan"),
                SessionConfigSelectOption(value="ask", name="Ask"),
            ],
        ),
    ]


class DummyAgent:
    def __init__(self) -> None:
        self.connection = None
        self.model = "cursor-small"
        self.mode = "agent"
        self.cancelled = asyncio.Event()

    def on_connect(self, connection) -> None:
        self.connection = connection

    async def initialize(self, protocol_version, **kwargs) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    resume=SessionResumeCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
            agent_info=Implementation(name="dummy-acp", version="1.0"),
        )

    async def new_session(self, cwd, **kwargs) -> NewSessionResponse:
        return NewSessionResponse(
            session_id="dummy-session",
            config_options=_config_options(self.model, self.mode),
        )

    async def load_session(self, cwd, session_id, **kwargs) -> LoadSessionResponse:
        return LoadSessionResponse(config_options=_config_options(self.model, self.mode))

    async def resume_session(self, session_id, cwd, **kwargs) -> ResumeSessionResponse:
        return ResumeSessionResponse(config_options=_config_options(self.model, self.mode))

    async def close_session(self, session_id, **kwargs):
        return {}

    async def set_config_option(self, config_id, session_id, value, **kwargs):
        if config_id == "model":
            self.model = str(value)
        elif config_id == "mode":
            self.mode = str(value)
        return SetSessionConfigOptionResponse(
            config_options=_config_options(self.model, self.mode)
        )

    async def prompt(self, session_id, prompt, **kwargs) -> PromptResponse:
        text = "".join(getattr(block, "text", "") for block in prompt)
        if text == "permission":
            outcome: RequestPermissionResponse = await self.connection.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id="tool-1",
                    title="Run command",
                    kind="execute",
                    status="pending",
                    raw_input={"command": "echo hello"},
                ),
                options=[
                    PermissionOption(
                        option_id="allow-once", name="Allow once", kind="allow_once"
                    ),
                    PermissionOption(
                        option_id="allow-always", name="Always allow", kind="allow_always"
                    ),
                    PermissionOption(
                        option_id="reject-once", name="Reject", kind="reject_once"
                    ),
                ],
            )
            text = json.dumps(outcome.model_dump(by_alias=True, exclude_none=True))
        elif text == "question":
            answer = await self.connection.ext_method(
                "cursor/ask_question",
                {
                    "sessionId": session_id,
                    "queryId": "query-1",
                    "title": "Choose a language",
                    "questions": [
                        {
                            "id": "language",
                            "prompt": "Language?",
                            "options": [
                                {"id": "en", "label": "English"},
                                {"id": "zh", "label": "简体中文"},
                            ],
                            "allowMultiple": False,
                        }
                    ],
                },
            )
            text = json.dumps(answer, ensure_ascii=False)
        elif text == "plan":
            answer = await self.connection.ext_method(
                "cursor/create_plan",
                {"toolCallId": "plan-1", "name": "Implementation", "plan": "Do it"},
            )
            text = json.dumps(answer)
        elif text == "wait":
            await self.cancelled.wait()
            return PromptResponse(stop_reason="cancelled")
        elif text == "multi-delta":
            for delta in ("first ", "second", " third"):
                await self.connection.session_update(
                    session_id=session_id,
                    update=AgentMessageChunk(
                        session_update="agent_message_chunk",
                        content=text_block(delta),
                        message_id="message-multi",
                    ),
                )
            return PromptResponse(stop_reason="end_turn")
        else:
            text = f"Echo: {text}; model={self.model}; mode={self.mode}"

        await self.connection.session_update(
            session_id=session_id,
            update=AgentMessageChunk(
                session_update="agent_message_chunk",
                content=text_block(text),
                message_id="message-1",
            ),
        )
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs) -> None:
        self.cancelled.set()


if __name__ == "__main__":
    asyncio.run(run_agent(DummyAgent(), use_unstable_protocol=True))

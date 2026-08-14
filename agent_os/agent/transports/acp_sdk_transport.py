# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed Agent Client Protocol transport built on the official Python SDK.

The transport owns protocol mechanics only.  Permission policy, blocked-agent
supervision, prompt queuing, and user-visible copy stay in the orchestration
layer.  Cursor-specific extension payloads are deliberately passed through as
data so the transport does not become the policy owner for questions or plans.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import psutil
from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AuthCapabilities,
    CancelElicitationResponse,
    ClientCapabilities,
    ClientSessionCapabilities,
    ConfigOptionUpdate,
    DeniedOutcome,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    FileSystemCapabilities,
    Implementation,
    PermissionOption,
    PlanCapabilities,
    RequestPermissionResponse,
    SessionConfigOptionsCapabilities,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from agent_os.agent.transports.base import AgentTransport, TransportEvent

logger = logging.getLogger(__name__)

SUPPORTED_ACP_SDK_VERSION = "0.11.0"
_VALID_PERMISSION_MODES = {"auto", "ask"}


class ACPSDKTransport(AgentTransport):
    """Persistent ACP stdio client using ``agent-client-protocol`` 0.11.0.

    ``resume_record`` has the same shape used by the other persistent
    transports: only ``session_id`` is identity.  Model and mode are live
    configuration and are never restored from that record.
    """

    supports_background_status = False

    def __init__(
        self,
        *,
        resume_record: dict[str, Any] | None = None,
        permission_mode: str = "auto",
        model: str | None = None,
        mode: str | None = None,
        config_options: Mapping[str, str | bool] | None = None,
        auth_method_id: str | None = None,
    ) -> None:
        if permission_mode not in _VALID_PERMISSION_MODES:
            raise ValueError(
                f"permission_mode must be one of {sorted(_VALID_PERMISSION_MODES)}"
            )
        record = resume_record or {}
        self._resume_session_id: str | None = record.get("session_id")
        self._permission_mode = permission_mode
        self._model = model
        self._mode = mode
        self._desired_config: dict[str, str | bool] = dict(config_options or {})
        if model is not None:
            self._desired_config.setdefault("model", model)
        if mode is not None:
            self._desired_config.setdefault("mode", mode)
        self._auth_method_id = auth_method_id

        self._workspace = ""
        self._session_id: str | None = None
        self._resume_outcome: tuple[str, str | None] = ("fresh", None)
        self._agent_capabilities: Any = None
        self._available_auth_methods: list[dict[str, Any]] = []
        self._session_config_options: dict[str, dict[str, Any]] = {}

        self._connection: Any = None
        self._stdio_context: Any = None
        self._popen: asyncio.subprocess.Process | None = None
        self._proc: psutil.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._prompt_task: asyncio.Task[None] | None = None
        self._alive = False
        self._stopping = False
        self._turn_open = False
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._turn_text_parts: list[str] = []
        self._turn_message_emitted = False
        # The SDK dispatcher runs notifications concurrently with response
        # handling. Raw-wire observation lets us wait until every preceding
        # session/update handler has finished before treating prompt response
        # as the transcript-facing boundary.
        self._pending_session_updates = 0
        self._session_updates_idle = asyncio.Event()
        self._session_updates_idle.set()

        # Names retained for compatibility with SubAgentManager's approval
        # recovery scan.  Values are typed response futures owned by the SDK
        # callback that is currently blocking the provider.
        self._pending_approvals: dict[str, asyncio.Future[RequestPermissionResponse]] = {}
        self._pending_approval_data: dict[str, dict[str, Any]] = {}
        self._pending_interactions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # process/session startup
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_internal_args(
        args: list[str],
    ) -> tuple[list[str], str | None, str | None]:
        """Strip Orbital-only flags and normalize Cursor's global model flag.

        Cursor requires ``agent --model X acp`` rather than
        ``agent acp --model X``.  We also apply the model through advertised
        session configuration after ``session/new``/``session/load``; keeping
        the reordered flag is the compatibility fallback for ACP agents that
        do not advertise a model option.
        """
        cleaned: list[str] = []
        permission_mode: str | None = None
        model: str | None = None
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--orbital-permission-mode":
                if index + 1 >= len(args):
                    raise ValueError("--orbital-permission-mode requires a value")
                permission_mode = args[index + 1]
                index += 2
                continue
            if arg == "--model":
                if index + 1 >= len(args):
                    raise ValueError("--model requires a value")
                model = args[index + 1]
                index += 2
                continue
            cleaned.append(arg)
            index += 1
        if model is not None:
            try:
                acp_index = cleaned.index("acp")
            except ValueError:
                cleaned.extend(["--model", model])
            else:
                cleaned[acp_index:acp_index] = ["--model", model]
        return cleaned, permission_mode, model

    async def start(
        self,
        command: str,
        args: list[str],
        workspace: str,
        env: dict | None = None,
    ) -> None:
        if self._alive:
            raise RuntimeError("ACP transport is already started")
        cleaned_args, permission_mode, argv_model = self._extract_internal_args(list(args))
        if permission_mode is not None:
            self.update_permission_mode(permission_mode)
        if argv_model is not None:
            self._model = argv_model
            self._desired_config["model"] = argv_model

        self._workspace = os.path.abspath(workspace)
        merged_env = dict(os.environ)
        merged_env.pop("CLAUDECODE", None)
        if env:
            merged_env.update(env)

        self._stdio_context = spawn_agent_process(
            self,
            command,
            *cleaned_args,
            cwd=self._workspace,
            env=merged_env,
            transport_kwargs={"shutdown_timeout": 2.0},
            # session/resume and session/close are advertised typed methods in
            # SDK 0.11 but remain protocol-marked unstable.  We only call them
            # after capability negotiation.
            use_unstable_protocol=True,
            observers=[self._observe_stream],
        )
        try:
            self._connection, self._popen = await self._stdio_context.__aenter__()
            self._alive = True
            try:
                self._proc = psutil.Process(self._popen.pid)
            except Exception:
                logger.warning("ACPSDKTransport: could not capture process pid=%s", self._popen.pid)
            if self._popen.stderr is not None:
                self._stderr_task = asyncio.create_task(
                    self._drain_stderr(), name=f"acp-stderr-{self._popen.pid}"
                )
            await self._initialize_and_open_session()
        except BaseException:
            await self.stop()
            raise

    async def _initialize_and_open_session(self) -> None:
        response = await self._connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
                session=ClientSessionCapabilities(
                    config_options=SessionConfigOptionsCapabilities(boolean={})
                ),
                plan=PlanCapabilities(),
                auth=AuthCapabilities(terminal=False),
                elicitation=ElicitationCapabilities(
                    form=ElicitationFormCapabilities()
                ),
            ),
            client_info=Implementation(
                name="orbital", title="Orbital", version="0.7.5"
            ),
        )
        if response.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                "ACP protocol negotiation failed: "
                f"agent selected {response.protocol_version}, client supports {PROTOCOL_VERSION}"
            )
        self._agent_capabilities = response.agent_capabilities
        self._available_auth_methods = [
            method.model_dump(by_alias=True, exclude_none=True)
            for method in (response.auth_methods or [])
        ]
        if self._auth_method_id is not None:
            await self.authenticate(self._auth_method_id)

        session_response: Any
        requested_resume = self._resume_session_id
        if requested_resume is not None:
            try:
                session_caps = getattr(self._agent_capabilities, "session_capabilities", None)
                if session_caps is not None and session_caps.resume is not None:
                    session_response = await self._connection.resume_session(
                        session_id=requested_resume, cwd=self._workspace
                    )
                elif bool(getattr(self._agent_capabilities, "load_session", False)):
                    session_response = await self._connection.load_session(
                        session_id=requested_resume, cwd=self._workspace
                    )
                else:
                    raise _ResumeUnsupported
            except _ResumeUnsupported:
                self._resume_session_id = None
                self._resume_outcome = ("fresh", "resume_unsupported")
                session_response = await self._connection.new_session(cwd=self._workspace)
                self._session_id = session_response.session_id
            except RequestError as exc:
                if exc.code != -32002:  # ACP resource-not-found
                    raise
                logger.warning(
                    "ACPSDKTransport: session %s could not be resumed; starting fresh: %s",
                    requested_resume,
                    exc,
                )
                self._resume_session_id = None
                self._resume_outcome = ("fresh", "resume_failed")
                session_response = await self._connection.new_session(cwd=self._workspace)
                self._session_id = session_response.session_id
            else:
                self._session_id = requested_resume
                self._resume_outcome = ("resumed", None)
        else:
            session_response = await self._connection.new_session(cwd=self._workspace)
            self._session_id = session_response.session_id
            self._resume_outcome = ("fresh", None)

        self._remember_session_configuration(session_response)
        await self._apply_desired_configuration(session_response)
        await self._wait_for_session_updates()
        event_data = {
            "session_id": self._session_id,
            "resume_status": self._resume_outcome[0],
            "resume_reason": self._resume_outcome[1],
            "model": self._effective_config_value("model") or self._model,
        }
        await self._event_queue.put(TransportEvent("thread_started", event_data))

    async def authenticate(self, method_id: str) -> None:
        if self._connection is None:
            raise RuntimeError("ACP transport is not started")
        advertised = {
            item.get("id") or item.get("methodId") for item in self._available_auth_methods
        }
        if advertised and method_id not in advertised:
            raise ValueError(f"ACP authentication method is not advertised: {method_id}")
        await self._connection.authenticate(method_id=method_id)

    def _remember_session_configuration(self, response: Any) -> None:
        self._session_config_options = {}
        for option in getattr(response, "config_options", None) or []:
            dumped = option.model_dump(by_alias=True, exclude_none=True)
            self._session_config_options[option.id] = dumped

    async def _apply_desired_configuration(self, response: Any) -> None:
        if self._session_id is None:
            return
        options = {
            option.id: option for option in (getattr(response, "config_options", None) or [])
        }
        for config_id, value in self._desired_config.items():
            option = options.get(config_id)
            if option is None or option.current_value == value:
                continue
            if not _config_value_is_advertised(option, value):
                # Cursor accepts friendly CLI aliases (for example
                # ``cursor-grok-4.5-low``) then reports a canonical session
                # value (for example ``grok-4.5[effort=high,fast=true]``).
                # Re-sending the alias through session/set_config_option is
                # invalid.  Trust the provider's canonicalized current value.
                logger.debug(
                    "ACPSDKTransport: desired %s=%r is not an advertised session value; "
                    "keeping provider-selected %r",
                    config_id,
                    value,
                    option.current_value,
                )
                continue
            result = await self._connection.set_config_option(
                config_id=config_id, session_id=self._session_id, value=value
            )
            self._remember_session_configuration(result)

    def _effective_config_value(self, config_id: str) -> str | bool | None:
        option = self._session_config_options.get(config_id) or {}
        return option.get("currentValue")

    # ------------------------------------------------------------------
    # prompt lifecycle and event streaming
    # ------------------------------------------------------------------

    async def dispatch(self, message: str) -> None:
        if not self.is_alive() or self._connection is None or self._session_id is None:
            raise RuntimeError("ACP transport is not initialized")
        if self._prompt_task is not None and not self._prompt_task.done():
            raise RuntimeError("ACP prompt already in progress; queue the follow-up")
        self._turn_text_parts = []
        self._turn_message_emitted = False
        self._turn_open = True
        self._prompt_task = asyncio.create_task(
            self._run_prompt(message), name=f"acp-prompt-{self._session_id}"
        )

    async def send(self, message: str) -> str | None:
        await self.dispatch(message)
        assert self._prompt_task is not None
        await self._prompt_task
        return "".join(self._turn_text_parts) or "(no response)"

    async def _run_prompt(self, message: str) -> None:
        try:
            response = await self._connection.prompt(
                session_id=self._session_id, prompt=[text_block(message)]
            )
            await self._wait_for_session_updates()
            await self._emit_accumulated_message()
            stop_reason = response.stop_reason
            if stop_reason == "cancelled":
                cause = "stopped" if self._stopping else "interrupted"
            elif stop_reason == "refusal":
                cause = "error"
            else:
                cause = "success"
            await self._event_queue.put(
                TransportEvent(
                    "turn_complete",
                    {
                        "cause": cause,
                        "stop_reason": stop_reason,
                        "session_id": self._session_id,
                        "model": self._effective_config_value("model") or self._model,
                    },
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._stopping:
                await self._emit_accumulated_message()
                await self._event_queue.put(
                    TransportEvent("error", {"error": str(exc)}, f"Error: {exc}")
                )
                await self._event_queue.put(
                    TransportEvent(
                        "turn_complete",
                        {
                            "cause": "error",
                            "session_id": self._session_id,
                            "model": self._effective_config_value("model") or self._model,
                        },
                    )
                )
        finally:
            self._turn_open = False

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        while self.is_alive() or not self._event_queue.empty():
            try:
                yield await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if not self.is_alive():
                    break

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        try:
            if session_id != self._session_id:
                logger.debug("ACPSDKTransport: update for inactive session %s", session_id)
                return
            if not self._turn_open:
                # A load/resume may replay history and agents may announce config
                # during startup.  Neither is a user turn; emitting an ordinary
                # chunk here would make ProcessManager open a phantom dispatch.
                if isinstance(update, ConfigOptionUpdate):
                    for option in update.config_options:
                        self._session_config_options[option.id] = option.model_dump(
                            by_alias=True, exclude_none=True
                        )
                return
            if isinstance(update, AgentMessageChunk):
                # ACP agent-message chunks are deltas. ProcessManager's `message`
                # event is a complete transcript/UI message, so retain deltas
                # through the turn and emit exactly once immediately before the
                # genuine session/prompt response boundary.
                self._turn_text_parts.append(_content_text(update.content))
                return
            event = self._session_update_to_event(update)
            if event is not None:
                await self._event_queue.put(event)
        finally:
            if self._pending_session_updates > 0:
                self._pending_session_updates -= 1
                if self._pending_session_updates == 0:
                    self._session_updates_idle.set()

    def _observe_stream(self, event: Any) -> None:
        """Synchronously count incoming session updates before dispatch."""
        message = getattr(event, "message", {})
        direction = getattr(getattr(event, "direction", None), "value", None)
        if direction == "incoming" and message.get("method") == "session/update":
            self._pending_session_updates += 1
            self._session_updates_idle.clear()

    async def _wait_for_session_updates(self) -> None:
        if self._pending_session_updates == 0:
            return
        try:
            await asyncio.wait_for(self._session_updates_idle.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(
                "ACPSDKTransport: timed out draining %s session updates",
                self._pending_session_updates,
            )

    async def _emit_accumulated_message(self) -> None:
        if self._turn_message_emitted or not self._turn_text_parts:
            return
        text = "".join(self._turn_text_parts)
        self._turn_message_emitted = True
        await self._event_queue.put(
            TransportEvent("message", {"text": text, "complete": True}, text)
        )

    def _session_update_to_event(self, update: Any) -> TransportEvent | None:
        dumped = update.model_dump(by_alias=True, exclude_none=True)
        if isinstance(update, AgentMessageChunk):
            text = _content_text(update.content)
            return TransportEvent(
                "message",
                {"text": text, "delta": True, "message_id": update.message_id},
                text,
            )
        if isinstance(update, AgentThoughtChunk):
            text = _content_text(update.content)
            return TransportEvent("status", {"text": text, "kind": "thought"}, text)
        if isinstance(update, (ToolCallStart, ToolCallProgress)):
            return TransportEvent(
                "tool_use",
                {
                    "tool_name": update.title or update.kind or "tool",
                    "tool_call_id": update.tool_call_id,
                    "tool_input": update.raw_input,
                    "status": update.status,
                    "update": dumped,
                },
                # The post-hoc capsule parser recovers tool names from this text
                # and nothing else: transcript entries persist ``content`` with
                # no metadata, so the format IS the contract
                # (sub_agent_transcript.py:17-19). Plain titles produced zero
                # capsule rows for every ACP agent, cursor included.
                f"[Using tool: {update.title or update.kind or 'tool'}]",
            )
        if isinstance(update, (AgentPlanUpdate, AgentPlanContentUpdate, AgentPlanRemovedUpdate)):
            return TransportEvent("status", {"kind": "plan", "plan": dumped}, "Plan updated")
        if isinstance(update, ConfigOptionUpdate):
            for option in update.config_options:
                self._session_config_options[option.id] = option.model_dump(
                    by_alias=True, exclude_none=True
                )
            return TransportEvent(
                "status", {"kind": "config_options", "options": dumped}, ""
            )
        return TransportEvent("status", {"kind": "session_update", "update": dumped}, "")

    # ------------------------------------------------------------------
    # permission requests
    # ------------------------------------------------------------------

    async def request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        if self._permission_mode == "auto":
            option = _choose_permission_option(options, approved=True)
            if option is None:
                return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=option.option_id)
            )

        request_id = str(uuid.uuid4())
        future: asyncio.Future[RequestPermissionResponse] = (
            asyncio.get_running_loop().create_future()
        )
        data = {
            "request_id": request_id,
            "permission_id": request_id,
            "session_id": session_id,
            "tool_name": tool_call.title or tool_call.kind or "tool",
            "tool_input": tool_call.raw_input,
            "tool_call": tool_call.model_dump(by_alias=True, exclude_none=True),
            "options": [
                option.model_dump(by_alias=True, exclude_none=True) for option in options
            ],
        }
        self._pending_approvals[request_id] = future
        self._pending_approval_data[request_id] = data
        await self._event_queue.put(
            TransportEvent(
                "permission_request",
                data,
                f"Permission requested: {data['tool_name']}",
            )
        )
        try:
            return await future
        finally:
            self._pending_approvals.pop(request_id, None)
            self._pending_approval_data.pop(request_id, None)

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        data = self._pending_approval_data.get(permission_id)
        if data is None:
            return
        options = [PermissionOption.model_validate(item) for item in data.get("options", [])]
        selected = _choose_permission_option(options, approved=approved)
        await self.respond_to_permission_option(
            permission_id, selected.option_id if selected is not None else None
        )

    async def respond_to_permission_option(
        self,
        permission_id: str,
        option_id: str | None,
        *,
        outcome: str = "selected",
    ) -> None:
        future = self._pending_approvals.get(permission_id)
        if future is None or future.done():
            return
        valid_ids = {
            option.get("optionId")
            for option in self._pending_approval_data[permission_id].get("options", [])
        }
        if outcome == "cancelled" or option_id is None:
            response = RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )
        else:
            if option_id not in valid_ids:
                raise ValueError(f"unknown ACP permission option: {option_id}")
            response = RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=option_id)
            )
        future.set_result(response)

    async def respond_to_permission_response(
        self,
        permission_id: str,
        *,
        approved: bool,
        reply_text: str | None = None,
        temporary_allow_s: int | None = None,
        decision: str | None = None,
    ) -> None:
        """Rich orchestration seam; text/temporary bypass remain Orbital state."""
        del reply_text, temporary_allow_s
        if decision is not None:
            normalized = decision.lower().replace("_", "-")
            if normalized in {"cancel", "cancelled", "deny-stop", "deny-and-stop"}:
                await self.respond_to_permission_option(
                    permission_id, None, outcome="cancelled"
                )
            elif normalized in {"decline", "deny", "reject"}:
                await self.respond_to_permission(permission_id, False)
            elif normalized in {"accept", "approve", "allow"}:
                await self.respond_to_permission(permission_id, True)
            else:
                # Provider-native option ids remain available to richer cards.
                await self.respond_to_permission_option(permission_id, decision)
        else:
            await self.respond_to_permission(permission_id, approved)

    # ------------------------------------------------------------------
    # provider-neutral blocking interactions and ACP extensions
    # ------------------------------------------------------------------

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._request_interaction(method, params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._event_queue.put(
            TransportEvent(
                "status",
                {"kind": "extension_notification", "method": method, "params": params},
                "",
            )
        )

    async def create_elicitation(
        self, message: str, mode: Any, **kwargs: Any
    ) -> AcceptElicitationResponse | CancelElicitationResponse:
        payload = {
            "message": message,
            "mode": mode.model_dump(by_alias=True, exclude_none=True),
            **kwargs,
        }
        response = await self._request_interaction("elicitation/create", payload)
        action = response.get("action")
        if action == "accept":
            return AcceptElicitationResponse(
                action="accept", content=response.get("content")
            )
        return CancelElicitationResponse(action="cancel")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        del elicitation_id, kwargs

    async def _request_interaction(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        interaction_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        kind = method.removeprefix("_")
        data = {
            "interaction_id": interaction_id,
            "kind": kind,
            "method": kind,
            "params": params,
            "prompt": params.get("title") or params.get("message") or kind,
        }
        if "questions" in params:
            data["questions"] = params["questions"]
            data["options"] = params["questions"]
        if kind.endswith("create_plan"):
            data["plan"] = params
        self._pending_interactions[interaction_id] = {
            "future": future,
            "data": data,
        }
        await self._event_queue.put(
            TransportEvent("interaction_required", data, str(data["prompt"]))
        )
        try:
            return await future
        finally:
            self._pending_interactions.pop(interaction_id, None)

    async def resolve_interaction(
        self,
        interaction_id: str,
        response: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        selection: Any = None,
        accepted: bool | None = None,
    ) -> None:
        pending = self._pending_interactions.get(interaction_id)
        if pending is None:
            raise KeyError(f"unknown ACP interaction: {interaction_id}")
        future: asyncio.Future[dict[str, Any]] = pending["future"]
        if future.done():
            return
        if response is None:
            response = _interaction_response(
                pending["data"], text=text, selection=selection, accepted=accepted
            )
        future.set_result(response)

    async def respond_to_interaction(self, interaction_id: str, **kwargs: Any) -> None:
        """Alias used by orchestration code that names the action ``respond``."""
        await self.resolve_interaction(interaction_id, **kwargs)

    # ------------------------------------------------------------------
    # teardown/introspection
    # ------------------------------------------------------------------

    async def cancel(self) -> None:
        if self._connection is not None and self._session_id is not None and self._turn_open:
            await self._connection.cancel(session_id=self._session_id)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            with contextlib.suppress(Exception):
                await self.cancel()

            for future in list(self._pending_approvals.values()):
                if not future.done():
                    future.set_result(
                        RequestPermissionResponse(
                            outcome=DeniedOutcome(outcome="cancelled")
                        )
                    )
            for pending in list(self._pending_interactions.values()):
                future = pending["future"]
                if not future.done():
                    future.set_result(_cancelled_interaction_response(pending["data"]))

            if self._prompt_task is not None and not self._prompt_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._prompt_task), timeout=2.0)
                except Exception:
                    self._prompt_task.cancel()
                    await asyncio.gather(self._prompt_task, return_exceptions=True)
            self._prompt_task = None

            session_caps = getattr(self._agent_capabilities, "session_capabilities", None)
            if (
                self._connection is not None
                and self._session_id is not None
                and session_caps is not None
                and session_caps.close is not None
            ):
                with contextlib.suppress(Exception):
                    await self._connection.close_session(session_id=self._session_id)

            if self._stderr_task is not None and not self._stderr_task.done():
                self._stderr_task.cancel()
                await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None

            if self._proc is not None:
                try:
                    from agent_os.agent.transports.process_kill import kill_process_tree

                    outcome = await kill_process_tree(self._proc, label="acp-sdk")
                    if not outcome.parent_dead:
                        logger.error("ACPSDKTransport: agent process survived teardown")
                except Exception:
                    logger.exception("ACPSDKTransport: process-tree teardown failed")
                    if self._popen is not None and self._popen.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            self._popen.kill()
            elif self._popen is not None and self._popen.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._popen.kill()

            if self._stdio_context is not None:
                with contextlib.suppress(Exception):
                    await self._stdio_context.__aexit__(None, None, None)
        finally:
            self._alive = False
            self._turn_open = False
            self._connection = None
            self._stdio_context = None
            self._popen = None
            self._proc = None
            self._session_id = None

    async def _drain_stderr(self) -> None:
        assert self._popen is not None and self._popen.stderr is not None
        while True:
            line = await self._popen.stderr.readline()
            if not line:
                return
            logger.debug("ACP agent stderr: %s", line.decode(errors="replace").rstrip())

    def is_alive(self) -> bool:
        return bool(
            self._alive and self._popen is not None and self._popen.returncode is None
        )

    def update_permission_mode(self, mode: str) -> None:
        if mode not in _VALID_PERMISSION_MODES:
            raise ValueError(f"permission mode must be one of {sorted(_VALID_PERMISSION_MODES)}")
        self._permission_mode = mode

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def resume_outcome(self) -> tuple[str, str | None]:
        return self._resume_outcome

    @property
    def available_auth_methods(self) -> list[dict[str, Any]]:
        return list(self._available_auth_methods)

    @property
    def session_config_options(self) -> dict[str, dict[str, Any]]:
        return dict(self._session_config_options)


class _ResumeUnsupported(Exception):
    pass


def _content_text(content: Any) -> str:
    return content.text if isinstance(content, TextContentBlock) else ""


def _choose_permission_option(
    options: list[PermissionOption], *, approved: bool
) -> PermissionOption | None:
    # Orbital's automatic/boolean policy is deliberately session-local. Never
    # translate it into a provider-persistent grant or rejection. A richer
    # permission card may still select a provider option id explicitly via
    # respond_to_permission_option().
    kind = "allow_once" if approved else "reject_once"
    for option in options:
        if option.kind == kind:
            return option
    return None


def _config_value_is_advertised(option: Any, value: str | bool) -> bool:
    """Whether ``value`` can safely be sent for an advertised config option."""
    if isinstance(value, bool):
        return getattr(option, "type", None) == "boolean"
    advertised: set[str] = set()
    for item in getattr(option, "options", None) or []:
        direct = getattr(item, "value", None)
        if direct is not None:
            advertised.add(direct)
        for nested in getattr(item, "options", None) or []:
            nested_value = getattr(nested, "value", None)
            if nested_value is not None:
                advertised.add(nested_value)
    return value in advertised


def _interaction_response(
    data: dict[str, Any],
    *,
    text: str | None,
    selection: Any,
    accepted: bool | None,
) -> dict[str, Any]:
    kind = data["kind"]
    if kind.endswith("ask_question"):
        if selection is None and text is None:
            return {"outcome": {"outcome": "cancelled"}}
        questions = data.get("params", {}).get("questions") or []
        if isinstance(selection, list) and all(
            isinstance(item, dict) for item in selection
        ):
            answers = selection
        elif isinstance(selection, list):
            question_id = questions[0].get("id") if questions else "question"
            answers = [
                {"questionId": question_id, "selectedOptionIds": selection}
            ]
        elif isinstance(selection, dict):
            answers = [
                {
                    "questionId": question_id,
                    "selectedOptionIds": (
                        option_ids if isinstance(option_ids, list) else [option_ids]
                    ),
                }
                for question_id, option_ids in selection.items()
            ]
        elif selection is not None:
            question_id = questions[0].get("id") if questions else "question"
            answers = [{"questionId": question_id, "selectedOptionIds": [selection]}]
        else:
            # Cursor's current option-question extension accepts option ids,
            # not arbitrary text. Only use a text answer shape when the
            # provider explicitly marks the question as free-text capable.
            question = questions[0] if questions else {}
            supports_text = bool(
                question.get("allowFreeText")
                or question.get("freeText")
                or question.get("answerType") == "text"
                or question.get("type") == "text"
            )
            if not supports_text:
                return {
                    "outcome": {
                        "outcome": "skipped",
                        "reason": "Option selection required",
                    }
                }
            answers = [{"questionId": question.get("id", "question"), "text": text}]
        return {"outcome": {"outcome": "answered", "answers": answers}}
    if kind.endswith("create_plan"):
        normalized = selection.lower() if isinstance(selection, str) else None
        if normalized in {"cancel", "cancelled"}:
            return {"outcome": {"outcome": "cancelled"}}
        if accepted is False or normalized in {"reject", "rejected", "deny"}:
            return {"outcome": {"outcome": "rejected", "reason": text or "Rejected"}}
        # `text` is management guidance/reason, never a filesystem URI. An
        # exact provider response (including an actual planUri) can still be
        # supplied through resolve_interaction(response=...).
        return {"outcome": {"outcome": "accepted"}}
    if kind == "elicitation/create":
        if accepted is False:
            return {"action": "cancel"}
        return {"action": "accept", "content": selection if isinstance(selection, dict) else {}}
    return {"text": text, "selection": selection, "accepted": accepted}


def _cancelled_interaction_response(data: dict[str, Any]) -> dict[str, Any]:
    kind = data["kind"]
    if kind.endswith("ask_question"):
        return {"outcome": {"outcome": "cancelled"}}
    if kind.endswith("create_plan"):
        return {"outcome": {"outcome": "cancelled"}}
    if kind == "elicitation/create":
        return {"action": "cancel"}
    return {"cancelled": True}

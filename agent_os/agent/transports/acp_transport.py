# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ACP transport -- JSON-RPC 2.0 over stdio for Agent Client Protocol."""
import asyncio
import json
import logging
import os
import subprocess
import sys

from agent_os.agent.transports.base import AgentTransport, TransportEvent

logger = logging.getLogger(__name__)

# Map ACP update kinds to TransportEvent types
_ACP_KIND_MAP = {
    "text": "message", "message": "message", "agentMessage": "message",
    "toolCall": "tool_use", "toolCallResult": "status",
    "permissionRequest": "permission_request",
    "thought": "status", "agentThought": "status",
    "error": "error",
}


class ACPTransport(AgentTransport):
    """ACP transport: JSON-RPC 2.0 over stdio for Agent Client Protocol."""

    def __init__(self):
        self._process = None
        self._session_id = None
        self._request_id = 0
        self._event_queue = asyncio.Queue()

    def _next_id(self):
        self._request_id += 1
        return self._request_id

    def _write(self, msg):
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode() if isinstance(data, str) else data)
        self._process.stdin.flush()

    def _read_response(self):
        """Read lines until we get a response (has 'id' + 'result'/'error') or EOF."""
        while True:
            line = self._process.stdout.readline()
            if not line:
                return None
            try:
                msg = json.loads(line.decode("utf-8", errors="replace").strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                return msg
            elif msg.get("method") == "session/update":
                params = msg.get("params", {})
                update = params.get("update", {})
                # MVP: auto-approve permission requests
                if update.get("kind") == "permissionRequest":
                    perm_id = update.get("permissionId", "")
                    logger.warning("ACP auto-approving permission request: %s (tool: %s)",
                                   perm_id, update.get("tool", {}).get("name", "unknown"))
                    self._auto_approve(perm_id)
                event = self._parse_session_update(params)
                self._event_queue.put_nowait(event)

    def _auto_approve(self, permission_id):
        """Auto-approve a permission request (MVP — full approval UI is follow-up)."""
        resp_id = self._next_id()
        self._write({
            "jsonrpc": "2.0", "id": resp_id, "method": "session/permissionResponse",
            "params": {
                "sessionId": self._session_id,
                "permissionId": permission_id,
                "granted": True,
            },
        })

    def _parse_session_update(self, params):
        update = params.get("update", {})
        kind = update.get("kind", "")
        event_type = _ACP_KIND_MAP.get(kind, "status")

        data = {"text": update.get("text", "")}
        raw_text = update.get("text", "")

        if kind == "permissionRequest":
            tool = update.get("tool", {})
            data = {
                "permission_id": update.get("permissionId", ""),
                "tool_name": tool.get("name", ""),
                "tool_args": tool.get("params", {}),
                "reason": update.get("reason", ""),
            }
            raw_text = f"{tool.get('name', '')}: {json.dumps(tool.get('params', {}))}"
        elif kind == "toolCall":
            tool = update.get("tool", update)
            data = {"tool_name": tool.get("name", ""), "text": raw_text}
        elif kind == "error":
            data = {"text": update.get("message", update.get("text", ""))}
            raw_text = data["text"]

        return TransportEvent(event_type=event_type, data=data, raw_text=raw_text)

    async def start(self, command, args, workspace, env=None):
        merged_env = os.environ.copy()
        merged_env.pop("CLAUDECODE", None)
        if env:
            merged_env.update(env)

        cmd = [command] + list(args)
        use_shell = sys.platform == "win32" and command.lower().endswith((".cmd", ".bat"))

        self._process = subprocess.Popen(
            " ".join(cmd) if use_shell else cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=workspace, env=merged_env,
            shell=use_shell,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Send initialize
        init_id = self._next_id()
        self._write({
            "jsonrpc": "2.0", "id": init_id, "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientInfo": {"name": "agent-os", "version": "1.0.0"},
                "clientCapabilities": {},
            },
        })
        init_resp = await asyncio.to_thread(self._read_response)
        if init_resp is None or "error" in init_resp:
            raise RuntimeError(f"ACP initialize failed: {init_resp}")

        # Send session/new
        new_id = self._next_id()
        self._write({
            "jsonrpc": "2.0", "id": new_id, "method": "session/new", "params": {},
        })
        new_resp = await asyncio.to_thread(self._read_response)
        if new_resp is None or "error" in new_resp:
            raise RuntimeError(f"ACP session/new failed: {new_resp}")

        self._session_id = new_resp.get("result", {}).get("sessionId")
        await self._event_queue.put(TransportEvent(
            event_type="session_created",
            data={"session_id": self._session_id},
        ))

    async def send(self, message):
        prompt_id = self._next_id()
        self._write({
            "jsonrpc": "2.0", "id": prompt_id, "method": "session/prompt",
            "params": {
                "sessionId": self._session_id,
                "prompt": [{"kind": "text", "text": message}],
            },
        })
        # Read response (blocking, but processes session/update notifications along the way)
        await asyncio.to_thread(self._read_response)
        # Collect accumulated text from events
        texts = []
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                if event.event_type == "message":
                    texts.append(event.data.get("text", event.raw_text))
            except asyncio.QueueEmpty:
                break
        return "\n".join(texts) if texts else "(no response)"

    async def read_stream(self):
        while self.is_alive() or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                if not self.is_alive():
                    break

    async def stop(self):
        if self._process and self._process.poll() is None:
            try:
                shutdown_id = self._next_id()
                self._write({
                    "jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown", "params": {},
                })
                await asyncio.to_thread(self._process.wait, timeout=5)
            except Exception:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except Exception:
                        self._process.kill()
        self._process = None

    def is_alive(self):
        return self._process is not None and self._process.poll() is None

    @property
    def session_id(self):
        return self._session_id

    async def respond_to_permission(self, permission_id, approved):
        if not self.is_alive():
            return
        resp_id = self._next_id()
        self._write({
            "jsonrpc": "2.0", "id": resp_id, "method": "session/permissionResponse",
            "params": {
                "sessionId": self._session_id,
                "permissionId": permission_id,
                "granted": approved,
            },
        })

# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PTY/PIPE transport: persistent subprocess with streaming output."""
import asyncio
import os
import subprocess
import sys

try:
    import pty
    _HAS_PTY = True
except ImportError:
    _HAS_PTY = False

from agent_os.agent.adapters.cli_adapter import strip_ansi
from agent_os.agent.adapters.output_parser import OutputParser
from agent_os.agent.transports.base import AgentTransport, TransportEvent


# Map OutputChunk.chunk_type to TransportEvent.event_type
_CHUNK_TO_EVENT = {
    "response": "message",
    "tool_activity": "tool_use",
    "approval_request": "permission_request",
    "status": "status",
}


class PTYTransport(AgentTransport):
    """Interactive transport: persistent process via PTY (Unix) or PIPE (Windows)."""

    def __init__(self, approval_patterns: list[str] | None = None):
        self._process: subprocess.Popen | None = None
        self._master_fd: int | None = None
        self._parser: OutputParser | None = None
        self._approval_patterns = approval_patterns or []

    async def start(self, command: str, args: list[str], workspace: str, env: dict | None = None) -> None:
        merged_env = os.environ.copy()
        merged_env.pop("CLAUDECODE", None)
        if env:
            merged_env.update(env)

        cmd = command
        if args:
            args_str = " ".join(f'"{a}"' if " " in a else a for a in args)
            cmd = f"{cmd} {args_str}"

        if _HAS_PTY:
            master_fd, slave_fd = pty.openpty()
            self._process = subprocess.Popen(
                cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                cwd=workspace, env=merged_env, shell=True,
            )
            os.close(slave_fd)
            self._master_fd = master_fd
            import fcntl
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        else:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workspace, env=merged_env, shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            self._master_fd = None

        await asyncio.sleep(0.05)
        if self._process.poll() is not None:
            self._process = None
            from agent_os.agent.adapters.base import AdapterError
            raise AdapterError("Process exited immediately")

        self._parser = OutputParser(self._approval_patterns)

    async def send(self, message: str) -> str | None:
        data = (message + "\n").encode("utf-8")
        if _HAS_PTY and self._master_fd is not None:
            os.write(self._master_fd, data)
        elif self._process and self._process.stdin:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        return None

    async def read_stream(self):
        loop = asyncio.get_event_loop()
        while self.is_alive():
            try:
                if _HAS_PTY and self._master_fd is not None:
                    raw = await loop.run_in_executor(None, self._read_pty)
                else:
                    raw = await loop.run_in_executor(None, self._read_pipe)

                if raw is None:
                    continue

                text = raw.decode("utf-8", errors="replace")
                text = strip_ansi(text)
                chunk = self._parser.parse(text)
                yield TransportEvent(
                    event_type=_CHUNK_TO_EVENT.get(chunk.chunk_type, "message"),
                    data={"text": chunk.text, "timestamp": chunk.timestamp},
                    raw_text=chunk.text,
                )
            except OSError:
                break

        if not self.is_alive() and self._parser:
            chunk = self._parser.parse("")
            yield TransportEvent(
                event_type=_CHUNK_TO_EVENT.get(chunk.chunk_type, "message"),
                data={"text": chunk.text, "timestamp": chunk.timestamp},
                raw_text=chunk.text,
            )

    def _read_pty(self) -> bytes | None:
        try:
            return os.read(self._master_fd, 4096)
        except OSError:
            return None

    def _read_pipe(self) -> bytes | None:
        if self._process and self._process.stdout:
            try:
                data = self._process.stdout.read(4096)
                if data:
                    return data
            except (OSError, ValueError):
                pass
        return None

    async def stop(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except (TimeoutError, subprocess.TimeoutExpired):
                    self._process.kill()
            self._process = None

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PTY-based CLI agent adapter with Windows subprocess.PIPE fallback.

Platform detection per BUG-001:
- Unix: uses pty.openpty() for pseudo-terminal
- Windows: uses subprocess.PIPE as fallback
"""

import asyncio
import os
import re
import subprocess
import sys
import threading

try:
    import pty
    _HAS_PTY = True
except ImportError:
    _HAS_PTY = False

from agent_os.agent.adapters.base import AdapterConfig, AdapterError, AgentAdapter, OutputChunk
from agent_os.agent.adapters.output_parser import OutputParser


# ANSI escape sequence stripping (module-level function, tests import it directly)
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b[()][AB012]|\x1b\[?[0-9;]*[hl]')


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


class CLIAdapter(AgentAdapter):
    agent_type = "cli"

    def __init__(self, handle: str, display_name: str,
                 platform_provider=None, project_id: str | None = None,
                 mode: str = "interactive", prompt_flag: str = "-p",
                 resume_flag: str = "--resume", session_id_pattern: str = "",
                 transport=None):
        self.handle = handle
        self.display_name = display_name
        self._platform_provider = platform_provider
        self._project_id = project_id
        self._process = None
        self._master_fd = None
        self._idle = False
        self._pending_response = False
        self._parser = None
        self._proc_handle = None  # ProcessHandle from provider
        self._mode = mode
        self._prompt_flag = prompt_flag
        self._resume_flag = resume_flag
        self._session_id_pattern = session_id_pattern
        self._session_id = None
        self._config = None
        self._on_output = None  # callback for pipe mode output
        self._last_response = None  # stores last pipe-mode response
        self._transport = transport  # Optional AgentTransport for delegation
        self._send_lock = asyncio.Lock()  # serialize concurrent send() calls

    @property
    def _using_provider(self) -> bool:
        return self._platform_provider is not None and self._project_id is not None

    async def start(self, config: AdapterConfig) -> None:
        self._config = config  # Always store config (pipe mode needs it for send())
        if self._transport:
            await self._transport.start(
                config.command, config.args or [], config.workspace, config.env,
            )
            return
        if self._using_provider:
            await self._start_via_provider(config)
        else:
            await self._start_fallback(config)
        self._parser = OutputParser(config.approval_patterns)

    async def _start_via_provider(self, config: AdapterConfig) -> None:
        self._proc_handle = await self._platform_provider.run_process(
            project_id=self._project_id,
            command=config.command,
            args=config.args or [],
            working_dir=config.workspace,
            extra_env=config.env,
            use_pty=True,
        )

    async def _start_fallback(self, config: AdapterConfig) -> None:
        merged_env = os.environ.copy()
        merged_env.pop("CLAUDECODE", None)  # Prevent nested session detection
        if config.env:
            merged_env.update(config.env)

        # Build command string with args
        cmd = config.command
        if config.args:
            args_str = " ".join(
                f'"{a}"' if " " in a else a for a in config.args
            )
            cmd = f"{cmd} {args_str}"

        if _HAS_PTY:
            master_fd, slave_fd = pty.openpty()
            self._process = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=config.workspace,
                env=merged_env,
                shell=True,
            )
            os.close(slave_fd)
            self._master_fd = master_fd
            # Set master_fd to non-blocking
            import fcntl
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        else:
            # Windows fallback: subprocess.PIPE
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=config.workspace,
                env=merged_env,
                shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            self._master_fd = None

        # Brief pause to let process start, then check if it exited immediately
        await asyncio.sleep(0.05)
        if self._process.poll() is not None:
            self._process = None
            raise AdapterError("Process exited immediately")

    async def send(self, message: str) -> None:
        async with self._send_lock:
            if self._transport:
                self._idle = False
                self._pending_response = True
                response = await self._transport.send(message)
                self._pending_response = False
                if response is not None:
                    self._last_response = response
                    if self._on_output:
                        self._on_output(response)
                self._idle = True
                return
            data = (message + "\n").encode("utf-8")
            self._idle = False
            if self._using_provider and self._proc_handle:
                self._proc_handle.stdin.write(data)
                self._proc_handle.stdin.flush()
            elif _HAS_PTY and self._master_fd is not None:
                os.write(self._master_fd, data)
            elif self._process and self._process.stdin:
                self._process.stdin.write(data)
                self._process.stdin.flush()

    async def read_stream(self):
        if self._transport:
            from agent_os.agent.transports.base import transport_event_to_chunk
            async for event in self._transport.read_stream():
                yield transport_event_to_chunk(event)
            return
        if self._using_provider and self._proc_handle:
            async for chunk in self._read_provider():
                yield chunk
        else:
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
                    yield chunk
                except OSError:
                    break

        # Final chunk if process exited
        if not self.is_alive() and self._parser:
            yield self._parser.parse("")

    async def _read_provider(self):
        """Read from provider-managed ProcessHandle via a single persistent reader thread.

        Uses an asyncio.Queue to decouple the blocking pipe read from the async
        consumer.  A lone daemon thread owns the read call — no abandoned threads,
        no lost data.
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()
        stop_event = threading.Event()

        def _reader_thread():
            while not stop_event.is_set():
                try:
                    data = self._proc_handle.stdout.read(4096)
                    if not data:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, data)
                except Exception:
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                    break

        thread = threading.Thread(target=_reader_thread, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if not self.is_alive():
                        break
                    continue  # no data yet, process still alive
                if data is None:
                    break  # EOF
                text = data.decode("utf-8", errors="replace")
                text = strip_ansi(text)
                if self._parser:
                    chunk = self._parser.parse(text)
                else:
                    chunk = OutputChunk(text=text, chunk_type="response")
                yield chunk
        finally:
            stop_event.set()

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
        if self._transport:
            await self._transport.stop()
            self._idle = False
            return
        if self._using_provider:
            await self._stop_via_provider()
        else:
            await self._stop_fallback()
        self._idle = False

    async def _stop_via_provider(self) -> None:
        if self._platform_provider and self._project_id:
            try:
                await self._platform_provider.stop_process(self._project_id)
            except Exception:
                pass
        self._proc_handle = None

    async def _stop_fallback(self) -> None:
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

    def is_idle(self) -> bool:
        return self._idle and not self._pending_response

    def is_alive(self) -> bool:
        if self._transport:
            return self._transport.is_alive()
        if self._using_provider and self._proc_handle:
            is_alive_fn = self._proc_handle._native_handles.get("is_alive")
            if is_alive_fn:
                return is_alive_fn()
            return False
        return self._process is not None and self._process.poll() is None

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        """Delegate permission response to transport if available."""
        if self._transport:
            await self._transport.respond_to_permission(permission_id, approved)

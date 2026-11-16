# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""What an adapter gets to work with: its source's settings, its secrets,
a small persisted state dict, and helpers that keep blocking work off the
event loop."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger("agent_os.sources.adapter")


@dataclass
class RunResult:
    code: int
    stdout: str
    stderr: str


class CommandNotFound(Exception):
    """The program to run is not installed (``FileNotFoundError``)."""


class CommandTimeout(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"timed out after {seconds:.0f}s")
        self.seconds = seconds


def _decode(b: bytes | None) -> str:
    return (b or b"").decode("utf-8", errors="replace")


async def _finish(proc, timeout: float, input_bytes: bytes | None):
    try:
        out, err = await asyncio.wait_for(proc.communicate(input_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        _kill(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass
        raise CommandTimeout(timeout) from None
    except asyncio.CancelledError:
        _kill(proc)
        raise
    return RunResult(proc.returncode if proc.returncode is not None else -1, _decode(out), _decode(err))


def _kill(proc) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _spawn_kwargs() -> dict:
    # The console-window flag is passed at each call site (creationflags=
    # win_no_window_flags(), bug #55's sweep reads it there); here only the
    # POSIX process group, so a timeout can kill the whole tree.
    return {} if sys.platform == "win32" else {"start_new_session": True}


async def run_argv(argv: list[str], *, timeout: float = 60, env: dict | None = None,
                   cwd: str | None = None, input_text: str | None = None) -> RunResult:
    """Run a program without a shell, off the loop, killed on timeout."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=full_env, cwd=cwd, creationflags=win_no_window_flags(), **_spawn_kwargs())
    except FileNotFoundError:
        raise CommandNotFound(argv[0]) from None
    return await _finish(proc, timeout, input_text.encode() if input_text is not None else None)


async def run_shell(command: str, *, timeout: float = 60, cwd: str | None = None,
                    env: dict | None = None) -> RunResult:
    """Run a command line through the platform shell (``/bin/sh -c`` /
    ``cmd.exe /c``), off the loop, killed on timeout."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = await asyncio.create_subprocess_shell(
        command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=full_env, cwd=cwd, creationflags=win_no_window_flags(),
        **_spawn_kwargs())
    return await _finish(proc, timeout, None)


class SourceContext:
    """Per call. ``config``/``kinds``/``account`` are copies: changing them
    changes nothing. ``state`` is the adapter's own dict, persisted by the
    core after every poll / emit (keep it small: cursors, last status)."""

    def __init__(self, *, source_id: str | None, type_: str, config: dict, kinds: dict,
                 account: str | None, secrets_fn, state: dict | None = None,
                 command: dict | None = None, emit_fn=None, connected_fn=None,
                 label: str | None = None, lang: str = "en"):
        self.source_id = source_id
        self.type = type_
        self.config = dict(config or {})
        self.kinds = {k: dict(v or {}) for k, v in (kinds or {}).items()}
        self.account = account
        self.label = label
        # "zh" | "en": the language for text Orbital itself composes (a
        # website-down title) — the interface language the app last
        # reported, as the daemon's own cards use.
        self.lang = lang if lang in ("zh", "en") else "en"
        self.command = dict(command) if command else None
        self.state = state if state is not None else {}
        self._secrets_fn = secrets_fn
        self._emit_fn = emit_fn
        self._connected_fn = connected_fn
        self.log = logger

    def secret(self, key: str) -> str | None:
        return self._secrets_fn(key) if self._secrets_fn else None

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    async def run(self, argv: list[str], **kw) -> RunResult:
        return await run_argv(argv, **kw)

    async def run_shell(self, command: str, **kw) -> RunResult:
        return await run_shell(command, **kw)

    async def to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def emit(self, events: list) -> None:
        if self._emit_fn is None:
            raise RuntimeError("emit is only available to a running stream")
        await self._emit_fn(events)

    def connected(self) -> None:
        if self._connected_fn is not None:
            self._connected_fn()

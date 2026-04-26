# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""DaemonProcess — manage a live uvicorn subprocess for integration tests.

Key behaviours:

- Spawns ``python -m uvicorn agent_os.api.app:create_app --factory``
  (matches ``scripts/restart-daemon.sh``). The import path
  ``agent_os.daemon_v2`` referenced in the task spec is NOT an executable
  entrypoint on this codebase — the FastAPI app factory is the real one.
- Picks an ephemeral TCP port from the OS when caller doesn't supply one
  so two harness daemons can run side-by-side on the same machine.
- Isolates the daemon's on-disk state by pointing ``HOME`` / ``USERPROFILE``
  at a throwaway directory. This moves ``~/orbital/daemon.pid``
  (singleton PID file) and ``~/orbital/browser-profile`` out of the real
  user account, which matters on workstations where a developer daemon
  is already running and holding the default pid file.
- Drains both stdout and stderr in a background thread (uvicorn logs to
  stderr). The thread strips CRLF so ``log_contains`` regexes stay
  portable.
- Polls ``GET /api/v2/settings`` for readiness — the daemon exposes no
  ``/health`` endpoint (confirmed via ``grep @router.`` across
  ``agent_os/api/routes/``). ``/api/v2/settings`` is also the endpoint
  ``scripts/restart-daemon.sh`` uses.
- Graceful shutdown tries CTRL_BREAK_EVENT on Windows and SIGTERM on
  POSIX, then falls back to ``kill_process_tree``. There is no
  ``POST /shutdown`` HTTP endpoint.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Iterable

import psutil

from . import process_tools

logger = logging.getLogger(__name__)


def _pick_ephemeral_port() -> int:
    """Ask the OS for an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll_http_ok(url: str, timeout: float = 15.0, interval: float = 0.2) -> bool:
    """Return True if ``GET url`` returns any 2xx response within ``timeout``."""
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as e:  # noqa: PERF203
            # Daemon reachable but returned non-2xx; treat as "up enough".
            if 200 <= e.code < 500:
                return True
            last_exc = e
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_exc = e
        time.sleep(interval)
    logger.debug("HTTP poll gave up on %s: %r", url, last_exc)
    return False


class DaemonProcess:
    """Manage a live uvicorn/FastAPI daemon subprocess.

    Typical usage (see ``conftest.py``)::

        d = DaemonProcess()
        d.start()
        try:
            ...  # exercise d.port
        finally:
            d.shutdown()
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._reader_threads: list[threading.Thread] = []
        self._log_lines: list[str] = []
        self._log_lock = threading.Lock()
        self._home_dir: tempfile.TemporaryDirectory | None = None
        self._data_dir: tempfile.TemporaryDirectory | None = None
        self._stop_readers = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(
        self,
        port: int | None = None,
        env: dict[str, str] | None = None,
        startup_timeout: float = 20.0,
    ) -> None:
        """Spawn the daemon and block until it answers health checks."""
        if self._proc is not None:
            raise RuntimeError("daemon already started")

        self._port = port if port is not None else _pick_ephemeral_port()

        # Isolate on-disk state (pid file, browser profile) so a real
        # developer daemon on the host can coexist with the test daemon.
        self._home_dir = tempfile.TemporaryDirectory(prefix="orbital-home-")
        self._data_dir = tempfile.TemporaryDirectory(prefix="orbital-data-")

        proc_env = os.environ.copy()
        proc_env["HOME"] = self._home_dir.name
        proc_env["USERPROFILE"] = self._home_dir.name  # Windows Path.home()
        # The app factory defaults its on-disk data dir to the *relative*
        # path "orbital-data". If we let the subprocess inherit the repo
        # root as its cwd, that path resolves to
        # ``<repo>/orbital-data/`` — the developer's real daemon data.
        # Make PYTHONPATH explicit so the subprocess can still import
        # ``agent_os`` from the repo root.
        repo_root = self._project_root()
        existing_pp = proc_env.get("PYTHONPATH", "")
        proc_env["PYTHONPATH"] = (
            repo_root + os.pathsep + existing_pp if existing_pp else repo_root
        )
        # Ensure subprocess stdout isn't buffered so we see startup lines
        # promptly. uvicorn writes through the root logger which respects
        # this flag for its streams.
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        # Avoid relay connection during tests — makes startup deterministic.
        proc_env.pop("AGENT_OS_RELAY_URL", None)
        if env:
            proc_env.update(env)

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "agent_os.api.app:create_app",
            "--factory",
            "--port",
            str(self._port),
            "--host",
            "127.0.0.1",
        ]

        popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            # Run from the isolated data dir so the daemon's default
            # relative "orbital-data" path lands inside our tempdir
            # instead of under the developer's real workspace. The
            # repo root is reachable via PYTHONPATH in proc_env.
            cwd=self._data_dir.name,
            # Line-buffering is unsupported in binary mode on Python
            # 3.13+ and emits a RuntimeWarning. Use the default.
            bufsize=-1,
            text=False,  # we decode ourselves to tolerate bad bytes
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]

        logger.info("Spawning daemon: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603

        # Start log capture before we start polling — early failures are
        # the most valuable logs to see.
        self._start_log_readers()

        health_url = f"http://127.0.0.1:{self._port}/api/v2/settings"
        if not _poll_http_ok(health_url, timeout=startup_timeout):
            # Drain what we have so the error has context.
            tail = "\n".join(self.log_tail(50))
            self._force_shutdown()
            raise TimeoutError(
                f"daemon on port {self._port} did not become healthy in "
                f"{startup_timeout}s. Last log lines:\n{tail}"
            )

    def shutdown(self, grace_seconds: float = 5.0) -> None:
        """Stop the daemon, reap children, and drain logs."""
        if self._proc is None:
            return

        proc = self._proc
        pid = proc.pid
        try:
            # No /shutdown HTTP endpoint exists — go straight to signals.
            if proc.poll() is None:
                self._signal_terminate(proc)

                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Daemon PID %s did not exit after %.1fs — killing tree",
                        pid,
                        grace_seconds,
                    )
                    process_tools.kill_process_tree(pid, timeout=grace_seconds)
        finally:
            self._stop_readers.set()
            for t in self._reader_threads:
                t.join(timeout=2.0)
            self._reader_threads.clear()

            # Final sweep — any surviving descendants (reparented after
            # the daemon exited) should be reaped now.
            process_tools.kill_process_tree(pid, timeout=2.0)

            self._proc = None

            if self._home_dir is not None:
                try:
                    self._home_dir.cleanup()
                except OSError:
                    pass
                self._home_dir = None
            if self._data_dir is not None:
                try:
                    self._data_dir.cleanup()
                except OSError:
                    pass
                self._data_dir = None

    def _force_shutdown(self) -> None:
        """Best-effort teardown used when ``start`` bails out."""
        try:
            self.shutdown(grace_seconds=2.0)
        except Exception:  # pragma: no cover - defensive
            logger.exception("force_shutdown raised")

    def _signal_terminate(self, proc: subprocess.Popen) -> None:
        """Send the platform-appropriate 'please exit' signal."""
        try:
            if sys.platform == "win32":
                # CTRL_BREAK_EVENT works for processes spawned with
                # CREATE_NEW_PROCESS_GROUP; plain terminate() on Windows
                # calls TerminateProcess which is equivalent to SIGKILL
                # so we prefer the signal path for a graceful try first.
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Signal the whole process group so uvicorn's reloader
                # helpers (if any) go down with the parent.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:  # pragma: no cover - defensive
                logger.exception("terminate fallback raised")

    # ------------------------------------------------------------------ #
    # Log capture
    # ------------------------------------------------------------------ #

    def _start_log_readers(self) -> None:
        assert self._proc is not None
        self._stop_readers.clear()
        for stream, label in ((self._proc.stdout, "stdout"), (self._proc.stderr, "stderr")):
            if stream is None:
                continue
            t = threading.Thread(
                target=self._drain_stream,
                args=(stream, label),
                name=f"daemon-{label}-reader",
                daemon=True,
            )
            t.start()
            self._reader_threads.append(t)

    def _drain_stream(self, stream: IO[bytes], label: str) -> None:
        try:
            for raw in iter(stream.readline, b""):
                if self._stop_readers.is_set():
                    break
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    line = repr(raw)
                # Normalise CRLF so callers write portable regexes.
                line = line.replace("\r\n", "\n").rstrip("\n")
                with self._log_lock:
                    self._log_lines.append(f"[{label}] {line}")
        except (ValueError, OSError):
            # Happens when the pipe is closed out from under us.
            return
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def log_lines(self) -> list[str]:
        """Snapshot of all captured log lines (prefixed with stream label)."""
        with self._log_lock:
            return list(self._log_lines)

    def log_tail(self, n: int) -> list[str]:
        with self._log_lock:
            return list(self._log_lines[-n:])

    def log_contains(self, pattern: str | re.Pattern) -> bool:
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        with self._log_lock:
            return any(regex.search(line) for line in self._log_lines)

    def log_count(self, pattern: str | re.Pattern) -> int:
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        with self._log_lock:
            return sum(1 for line in self._log_lines if regex.search(line))

    def log_since(self, marker_line: int) -> list[str]:
        """Return log lines captured at or after the given index."""
        with self._log_lock:
            return list(self._log_lines[marker_line:])

    def mark_log_position(self) -> int:
        """Return the current log length for later use with :meth:`log_since`."""
        with self._log_lock:
            return len(self._log_lines)

    def wait_for_log(
        self,
        pattern: str | re.Pattern,
        timeout: float = 10.0,
        interval: float = 0.1,
    ) -> bool:
        """Block until ``pattern`` appears in captured logs or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.log_contains(pattern):
                return True
            time.sleep(interval)
        return False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def pid(self) -> int:
        if self._proc is None:
            raise RuntimeError("daemon not running")
        return self._proc.pid

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("daemon not started")
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def home_dir(self) -> str | None:
        return self._home_dir.name if self._home_dir else None

    @property
    def data_dir(self) -> str | None:
        return self._data_dir.name if self._data_dir else None

    def child_pids(self) -> list[int]:
        """Return PIDs of current descendants."""
        return [p.pid for p in process_tools.get_children(self.pid)]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _project_root() -> str:
        # tests/integration/harness/daemon.py -> repo root is three parents up
        return str(Path(__file__).resolve().parents[3])

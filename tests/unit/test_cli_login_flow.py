# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the in-app CLI login flow (bug #64).

The bug: ``claude login`` is not a subcommand. It parses as the positional
prompt, so the CLI answers conversationally, burns an LLM turn, and **exits
0** without authenticating anything. ``_run_login_job`` keyed success off
``rc == 0`` alone, so Orbital reported a login that had not happened and the
row flipped straight back to signed-out.

Two fixes are covered, and the second is the one that generalises:

1. The manifest command is ``claude auth login``.
2. A zero exit code is necessary but not sufficient — the manifest's
   credential ``check_command`` is re-run before a login is reported as
   *confirmed*. That closes the false-success class for every agent with a
   login flow, not just claude-code.

Also covered: OSC-8/ANSI sanitising of ``login.progress`` (the CLI wraps its
URL in a terminal hyperlink, so the raw line carries the URL twice plus
control bytes) and the idle timeout that stops an abandoned flow leaking a
process forever.

No real login command runs here — the CLI is a stub script. A real
``codex login`` clears the developer's own credentials on start.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import patch

import pytest

from agent_os.agents.setup_engine import SetupEngine
from agent_os.api.routes import settings as settings_routes


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _RecordingWs:
    """Captures broadcast_global payloads instead of sending them."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def broadcast_global(self, payload: dict) -> None:
        self.events.append(dict(payload))

    @property
    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type_]

    def terminal(self) -> dict:
        """The single login.complete / login.failed the job ends on."""
        finals = [e for e in self.events
                  if e["type"] in ("login.complete", "login.failed")]
        assert len(finals) == 1, f"expected one terminal event, got {finals}"
        return finals[0]


class _StubEngine:
    """The SetupEngine seam ``_run_login_job`` reads for its re-check.

    ``configured`` is what the manifest's ``check_command`` would report
    *after* the login process exits.
    """

    def __init__(self, slug: str = "claude-code", configured: bool = False):
        self._manifest = object()  # opaque — the stub never inspects it
        self._registry = {slug: self._manifest}
        self._configured = configured
        self.credential_checks = 0
        self.invalidations = 0

    def invalidate_cache(self) -> None:
        self.invalidations += 1

    def resolve_binary(self, manifest) -> str:
        return "/fake/bin/claude"

    def check_credentials(self, manifest, resolved_binary=None):
        self.credential_checks += 1
        if self._configured:
            return (True, [])
        return (False, ["claude_auth"])


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict | None]] = []
        self.latched: list[str] = []

    def emit(self, event: str, fields: dict | None = None, **_kw) -> None:
        self.events.append((event, fields))

    def latch(self, milestone: str) -> None:
        self.latched.append(milestone)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]


def _stub_cli(tmp_path, body: str, name: str = "fake_cli.py") -> str:
    """A shell command string running ``body`` under this interpreter.

    Double quotes so the same string works under both ``sh`` and ``cmd.exe``
    — ``_run_login_job`` spawns through the shell.
    """
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


def _run_job(command: str, slug: str = "claude-code", job_id: str = "job1") -> None:
    """Seed the job table the way the route does, then run the job."""
    settings_routes._login_jobs[job_id] = {
        "slug": slug, "status": "running", "command": command,
    }
    asyncio.run(settings_routes._run_login_job(slug, job_id, command))


@pytest.fixture
def login_env(monkeypatch):
    """Wire the module globals ``_run_login_job`` reads. Yields (ws, engine)."""
    ws = _RecordingWs()
    engine = _StubEngine()
    monkeypatch.setattr(settings_routes, "_ws_manager", ws)
    monkeypatch.setattr(settings_routes, "_setup_engine", engine)
    settings_routes._login_jobs.clear()
    yield ws, engine
    settings_routes._login_jobs.clear()


# ---------------------------------------------------------------------------
# 1. D1 — the manifest command must exist
# ---------------------------------------------------------------------------

class TestClaudeLoginCommand:
    """``claude login`` is not a subcommand; ``claude auth login`` is."""

    @staticmethod
    def _load_claude_manifest():
        from agent_os.agents.manifest import ManifestLoader

        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "agent_os", "agents", "manifests", "claude_code.yaml",
        )
        return ManifestLoader.load(os.path.abspath(path))

    def test_manifest_declares_auth_login(self):
        manifest = self._load_claude_manifest()
        setup_commands = [c.setup_command for c in manifest.setup.credentials
                          if c.setup_command]
        assert setup_commands, "claude-code manifest lost its login command"
        assert all(cmd.endswith("auth login") for cmd in setup_commands), (
            f"expected 'claude auth login', got {setup_commands} — a bare "
            f"'claude login' parses as a prompt and exits 0 without "
            f"authenticating"
        )

    @patch("agent_os.agents.setup_engine.shutil.which",
           return_value="/opt/fake/claude")
    def test_resolve_setup_command_ends_with_auth_login(self, _which,
                                                        monkeypatch):
        """End-to-end through the resolver the Login button calls."""
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.register(self._load_claude_manifest())
        monkeypatch.setattr(settings_routes, "_setup_engine",
                            SetupEngine(registry))

        cmd = settings_routes._resolve_setup_command("claude-code", "login")

        assert cmd, "_resolve_setup_command('claude-code', 'login') was empty"
        assert cmd.endswith("auth login"), cmd
        # F1's other half: the resolved absolute path still gets substituted
        # in even though the command is now three words.
        assert cmd == "/opt/fake/claude auth login"

    def test_substitute_binary_rewrites_multiword_command(self):
        """_substitute_binary only matches at the start of the string — the
        multi-word form must still pick up the resolved path."""
        assert SetupEngine._substitute_binary(
            "claude auth login", "claude", "/Users/x/.local/bin/claude",
        ) == "/Users/x/.local/bin/claude auth login"


# ---------------------------------------------------------------------------
# 2. The false-success class — rc == 0 is not proof of a login
# ---------------------------------------------------------------------------

class TestLoginSuccessIsVerified:
    """The regression test for the whole bug: a CLI that exits 0 without
    authenticating must not be reported as a *confirmed* success."""

    def test_exit_zero_without_auth_is_not_confirmed(self, login_env, tmp_path):
        ws, engine = login_env  # engine reports still-not-configured
        cmd = _stub_cli(tmp_path, 'print("login is not something I can run")\n')

        _run_job(cmd)

        final = ws.terminal()
        assert final.get("verified") is False, (
            "an exit-0 that left the CLI unauthenticated was reported as a "
            "confirmed login — this is bug #64"
        )
        assert settings_routes._login_jobs["job1"]["status"] != "complete"
        assert engine.credential_checks >= 1, (
            "the manifest check_command was never re-run after the login exit"
        )

    def test_unconfirmed_exit_zero_is_not_a_hard_failure(self, login_env,
                                                         tmp_path):
        """Locked decision 2: a slow keychain write must not turn a real login
        into a reported error. The event stays login.complete."""
        ws, _engine = login_env
        cmd = _stub_cli(tmp_path, 'print("Login successful.")\n')

        _run_job(cmd)

        assert ws.terminal()["type"] == "login.complete"

    def test_verified_exit_zero_is_a_confirmed_success(self, login_env,
                                                       tmp_path, monkeypatch):
        ws, _engine = login_env
        monkeypatch.setattr(settings_routes, "_setup_engine",
                            _StubEngine(configured=True))
        cmd = _stub_cli(tmp_path, 'print("Login successful.")\n')

        _run_job(cmd)

        final = ws.terminal()
        assert final["type"] == "login.complete"
        assert final["verified"] is True
        assert settings_routes._login_jobs["job1"]["status"] == "complete"

    def test_nonzero_exit_still_fails(self, login_env, tmp_path):
        ws, engine = login_env
        cmd = _stub_cli(tmp_path, 'import sys\nsys.exit(3)\n')

        _run_job(cmd)

        final = ws.terminal()
        assert final["type"] == "login.failed"
        assert final["return_code"] == 3
        assert settings_routes._login_jobs["job1"]["status"] == "failed"
        assert engine.credential_checks == 0, (
            "no point re-checking credentials when the CLI itself failed"
        )


# ---------------------------------------------------------------------------
# 3. Progress lines are sanitised before they reach the UI
# ---------------------------------------------------------------------------

# What the claude CLI actually prints: an OSC-8 terminal hyperlink, so the
# URL appears twice (target + label) wrapped in control bytes.
_URL = "https://claude.com/cai/oauth/authorize?state=abc123"
_OSC8_LINE = f"\x1b]8;;{_URL}\x1b\\{_URL}\x1b]8;;\x1b\\"


class TestProgressSanitising:

    def test_strips_osc8_hyperlink_to_a_single_url(self):
        cleaned = settings_routes._strip_terminal_escapes(_OSC8_LINE)
        assert cleaned == _URL
        assert cleaned.count("https://") == 1
        assert "\x1b" not in cleaned

    def test_strips_sgr_colour_codes(self):
        assert settings_routes._strip_terminal_escapes(
            "\x1b[1;32mOpening browser…\x1b[0m") == "Opening browser…"

    def test_keeps_a_plain_line_untouched(self):
        assert settings_routes._strip_terminal_escapes(
            "Opening browser to sign in…") == "Opening browser to sign in…"

    def test_spinner_redraws_collapse_to_the_last_frame(self):
        assert settings_routes._strip_terminal_escapes(
            "- waiting\r\\ waiting\r| done") == "| done"

    # A CRLF line terminator is NOT a spinner redraw. Getting this wrong made
    # every progress line reduce to "" on Windows (`"waiting\r\n"` → split on
    # \r → "\n" → ""), which the broadcaster skips as falsy — so the whole
    # login progress stream, sign-in URL included, vanished on the platform
    # carrying most users. Asserted on the pure function so the guard holds on
    # every platform; the original tests drove a real child process and so only
    # ever saw the host's own line ending.
    def test_crlf_terminator_is_not_treated_as_a_redraw(self):
        assert settings_routes._strip_terminal_escapes(
            "still waiting 0\r\n") == "still waiting 0"

    def test_crlf_terminated_osc8_url_survives(self):
        assert settings_routes._strip_terminal_escapes(_OSC8_LINE + "\r\n") == _URL

    def test_redraws_still_collapse_when_crlf_terminated(self):
        assert settings_routes._strip_terminal_escapes(
            "- waiting\r| done\r\n") == "| done"

    def test_bare_cr_terminator_is_not_treated_as_a_redraw(self):
        assert settings_routes._strip_terminal_escapes("waiting\r") == "waiting"

    def test_broadcast_progress_line_is_clean(self, login_env, tmp_path):
        ws, _engine = login_env
        body = (
            'url = "https://claude.com/cai/oauth/authorize?state=abc123"\n'
            'print("\\x1b]8;;" + url + "\\x1b\\\\" + url + '
            '"\\x1b]8;;\\x1b\\\\", flush=True)\n'
        )
        _run_job(_stub_cli(tmp_path, body))

        progress = ws.of_type("login.progress")
        assert progress, "no login.progress broadcast"
        line = progress[0]["line"]
        assert line == _URL
        assert "\x1b" not in line
        assert line.count("https://") == 1


# ---------------------------------------------------------------------------
# 4. Idle timeout (locked decision 4)
# ---------------------------------------------------------------------------

class TestLoginIdleTimeout:

    def test_idle_process_is_killed_and_reported_failed(self, login_env,
                                                        tmp_path, monkeypatch):
        ws, _engine = login_env
        monkeypatch.setattr(settings_routes, "LOGIN_IDLE_TIMEOUT_SECONDS", 0.5)
        body = (
            'import time\n'
            'print("Paste code here if prompted >", flush=True)\n'
            'time.sleep(60)\n'
        )

        started = time.monotonic()
        _run_job(_stub_cli(tmp_path, body))
        elapsed = time.monotonic() - started

        assert elapsed < 20, (
            f"the job waited {elapsed:.1f}s — the idle timeout did not fire"
        )
        final = ws.terminal()
        assert final["type"] == "login.failed"
        assert final.get("timed_out") is True
        assert settings_routes._login_jobs["job1"]["status"] == "failed"

    def test_output_resets_the_idle_clock(self, login_env, tmp_path,
                                          monkeypatch):
        """The timeout is *since last activity*, not a wall-clock cap: a CLI
        that keeps printing must be allowed to run past it."""
        ws, _engine = login_env
        monkeypatch.setattr(settings_routes, "LOGIN_IDLE_TIMEOUT_SECONDS", 0.6)
        body = (
            'import time\n'
            'for i in range(6):\n'
            '    print("still waiting %d" % i, flush=True)\n'
            '    time.sleep(0.2)\n'
        )

        _run_job(_stub_cli(tmp_path, body))

        final = ws.terminal()
        assert final["type"] == "login.complete", (
            "a CLI that kept printing was killed by the idle timeout"
        )
        assert len(ws.of_type("login.progress")) == 6


# ---------------------------------------------------------------------------
# 5. Telemetry emits on the login path
# ---------------------------------------------------------------------------

class TestLoginTelemetry:

    @pytest.fixture
    def recorder(self, monkeypatch):
        rec = _RecordingTelemetry()
        monkeypatch.setattr(settings_routes, "telemetry", rec)
        return rec

    def test_attempt_is_emitted_when_the_job_starts(self, login_env, tmp_path,
                                                    recorder):
        _run_job(_stub_cli(tmp_path, 'print("hi")\n'))
        assert ("login_attempted", {"agent": "claude-code"}) in recorder.events

    def test_unverified_completion_counts_as_a_failure(self, login_env,
                                                       tmp_path, recorder):
        _run_job(_stub_cli(tmp_path, 'print("Login successful.")\n'))
        assert "login_failed" in recorder.names

    def test_verified_completion_does_not_count_as_a_failure(
            self, login_env, tmp_path, recorder, monkeypatch):
        monkeypatch.setattr(settings_routes, "_setup_engine",
                            _StubEngine(configured=True))
        _run_job(_stub_cli(tmp_path, 'print("Login successful.")\n'))
        assert "login_failed" not in recorder.names

    def test_nonzero_exit_counts_as_a_failure(self, login_env, tmp_path,
                                              recorder):
        _run_job(_stub_cli(tmp_path, 'import sys\nsys.exit(1)\n'))
        assert "login_failed" in recorder.names

    def test_timeout_counts_as_a_failure(self, login_env, tmp_path, recorder,
                                         monkeypatch):
        monkeypatch.setattr(settings_routes, "LOGIN_IDLE_TIMEOUT_SECONDS", 0.4)
        body = 'import time\ntime.sleep(60)\n'
        _run_job(_stub_cli(tmp_path, body))
        assert "login_failed" in recorder.names

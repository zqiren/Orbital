# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 193: Google's Antigravity CLI (``agy``) as a pinnable ACP worker.

Covered here:
1. The manifest and its param schema (and the per-OS install command).
2. The retired ``gemini-cli`` slug: gone from the registry and the schema,
   and an existing project that still names it loads and resolves quietly.
3. Paste-the-code login: a manifest-declared prompt keeps the login's stdin
   open, asks the card for the value, writes it to the CLI, and the value is
   redacted everywhere; a login without the prompt keeps its old stdin.
4. Region refusal: a turn whose whole output is a manifest-declared refusal
   (and ran no tool) becomes an error carrying the reason; anything else is
   unchanged. The reason reaches the durable row's meta.

No real ``agy`` runs: the login CLI is a stub script and the ACP agent is
the shared dummy fixture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.api.routes import settings as settings_routes

MANIFESTS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "agent_os", "agents", "manifests"))
ACP_FIXTURE = Path(__file__).parents[1] / "fixtures" / "dummy_acp_sdk_agent.py"

REFUSAL = ("⚠ You do not have a valid license of this product. Please contact "
           "your administrator to request a license. If you are not an "
           "enterprise user and believe you are receiving this message as an "
           "error, please try using the latest version and logging in again. "
           "(#3501)")
ELIGIBILITY = ("Eligibility check failed: Your current account is not eligible "
               "for Antigravity, because it is not currently available in your "
               "location.")


def _load():
    return ManifestLoader.load(os.path.join(MANIFESTS, "antigravity.yaml"))


def _bundled() -> AgentRegistry:
    registry = AgentRegistry()
    registry.load_directory(MANIFESTS)
    return registry


# ---------------------------------------------------------------------------
# 1. Manifest + schema
# ---------------------------------------------------------------------------

class TestManifest:

    def test_runtime_is_agy_acp_over_stdio(self):
        m = _load()
        assert m.slug == "antigravity"
        assert m.name == "Antigravity"
        assert m.runtime.adapter == "cli"
        assert m.runtime.command == "agy"
        assert m.runtime.transport == "acp-sdk"
        assert m.runtime.mode == "pipe"
        assert m.runtime.interactive is False
        assert m.runtime.args == ["acp"]
        assert m.listed is True

    def test_login_is_the_paste_the_code_flow_with_an_exit_code_probe(self):
        (cred,) = _load().setup.credentials
        assert cred.type == "oauth_cli"
        assert cred.required is True
        # `agy models`: exit 0 signed in, exit 1 "Please sign in" signed out.
        assert cred.check_command == "agy models"
        assert cred.check_field == ""
        assert cred.setup_command == "agy acp --login"
        # Matches agy 1.5.13's prompt line (case-insensitive).
        assert cred.setup_input_prompt.lower() in (
            "Then paste the authorization code, or the full URL of the page "
            "you land on, and press Enter:").lower()

    def test_detection_and_permissions(self):
        m = _load()
        assert m.setup.check_command == "agy --version"
        assert m.setup.dependencies == []
        assert "$HOME/.local/bin/agy" in m.setup.auto_detect["macos"]
        assert "$HOME/.local/bin/agy" in m.setup.auto_detect["linux"]
        # install.ps1: $TARGET_DIR = Join-Path $env:LOCALAPPDATA "agy\bin".
        assert m.setup.auto_detect["windows"] == ["%LOCALAPPDATA%\\agy\\bin\\agy.exe"]
        assert m.permissions.network_domains == []
        assert m.permissions.autonomy_recommended == "hands_off"

    @pytest.mark.parametrize("os_name,expected", [
        ("macos", "curl -fsSL https://antigravity.google/cli/install.sh | bash"),
        ("linux", "curl -fsSL https://antigravity.google/cli/install.sh | bash"),
        ("windows", "irm https://antigravity.google/cli/install.ps1 | iex"),
    ])
    def test_install_command_is_this_oses(self, monkeypatch, os_name, expected):
        import agent_os.daemon_v2.models as models
        monkeypatch.setattr(models, "detect_os", lambda: os_name)
        assert _load().setup.install_command == expected

    def test_a_plain_install_command_is_unchanged(self):
        kimi = ManifestLoader.load(os.path.join(MANIFESTS, "kimi-code.yaml"))
        assert kimi.setup.install_command == "npm install -g @moonshot-ai/kimi-code"

    def test_refusal_patterns_are_parsed(self):
        patterns = _load().runtime.refusal_patterns
        assert patterns and all(p["reason"] == "region_unavailable" for p in patterns)
        # Every other manifest declares none.
        for m in _bundled().list_all():
            if m.slug != "antigravity":
                assert m.runtime.refusal_patterns == []

    def test_param_schema(self):
        from agent_os.daemon_v2.sub_agent_config_store import SCHEMA, resolve_params
        assert SCHEMA["antigravity"]["model"].allowed is None
        assert SCHEMA["antigravity"]["model"].flag_template == "--model {value}"
        mode = SCHEMA["antigravity"]["permission-mode"]
        assert mode.allowed == ("auto", "ask")
        assert mode.default == "auto"
        assert mode.flag_template == "--orbital-permission-mode {value}"
        assert resolve_params("antigravity") == {"permission-mode": "auto"}

    def test_model_flag_lands_before_the_acp_subcommand(self):
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        args, mode, model = ACPSDKTransport._extract_internal_args(
            ["acp", "--model", "gemini-3.8-flash", "--orbital-permission-mode", "ask"])
        assert args == ["--model", "gemini-3.8-flash", "acp"]
        assert (mode, model) == ("ask", "gemini-3.8-flash")


# ---------------------------------------------------------------------------
# 2. Retired gemini-cli (spec 193 §5.3 / unknown §7.4)
# ---------------------------------------------------------------------------

class TestGeminiCliRetired:

    def test_gone_from_registry_and_schema_but_kept_in_telemetry(self):
        from agent_os.daemon_v2.sub_agent_config_store import SCHEMA
        from agent_os.telemetry.rollup import SUBAGENT_SLUGS
        assert _bundled().get("gemini-cli") is None
        assert "gemini-cli" not in SCHEMA
        assert "gemini-cli" in SUBAGENT_SLUGS
        assert "antigravity" in SUBAGENT_SLUGS

    def test_a_saved_gemini_cli_config_loads_and_does_nothing(self, tmp_path):
        import json
        from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore
        path = tmp_path / "sub_agent_config.json"
        path.write_text(json.dumps({
            "gemini-cli": {"model": "gemini-2.5-pro", "approval-mode": "yolo"},
            "codex": {"model": "gpt-5"},
        }), encoding="utf-8")
        store = SubAgentConfigStore(str(path))
        assert store.schema_for("gemini-cli") == {}
        assert store.build_extra_args("gemini-cli") == []
        assert store.build_extra_args("codex") == ["-m", "gpt-5"]

    def test_peer_list_drops_the_unknown_slug(self):
        from agent_os.daemon_v2.sub_agent_visibility import (
            resolve_visible_sub_agent_slugs,
        )
        statuses = [SimpleNamespace(slug="claude-code", installed=True,
                                    credentials_configured=True)]
        engine = MagicMock()
        engine.check_all.return_value = statuses
        slugs = resolve_visible_sub_agent_slugs(
            enabled_sub_agents=["gemini-cli", "claude-code"],
            disabled_sub_agents=["gemini-cli"], setup_engine=engine,
            ready_only=True)
        assert slugs == ["claude-code"]
        slugs = resolve_visible_sub_agent_slugs(
            enabled_sub_agents=["gemini-cli", "claude-code"],
            disabled_sub_agents=[], setup_engine=engine, ready_only=True)
        assert slugs == ["claude-code"]

    def test_a_project_still_naming_gemini_cli_loads_quietly(self, tmp_path,
                                                             monkeypatch):
        """An upgraded project whose row still says gemini-cli (sticky pin,
        explicit allowlist, denylist) loads, lists, and starts new sessions
        exactly as if the slug were not there."""
        monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
        from tests.unit.keyless_harness import build_client, create_project
        for keyed, expected in ((True, None), (False, "claude-code")):
            tc, deps, row = (lambda c: (c[0], c[1], None))(
                build_client(tmp_path / str(keyed), keyed=keyed))
            ws = tmp_path / f"ws-{keyed}"
            ws.mkdir()
            row = create_project(tc, ws, name=f"old-{keyed}")
            pid = row["project_id"]
            deps["project_store"].update_project(pid, {
                "default_pinned_target": "gemini-cli",
                "enabled_sub_agents": ["gemini-cli", "claude-code"],
                "disabled_sub_agents": ["gemini-cli"],
            })
            got = tc.get(f"/api/v2/projects/{pid}")
            assert got.status_code == 200, got.text
            body = got.json()
            assert body["effective_pinned_target"] == expected
            listed = tc.get("/api/v2/projects")
            assert listed.status_code == 200
            assert pid in {p["project_id"] for p in listed.json()}
            new = tc.post(f"/api/v2/agents/{pid}/new-session")
            assert new.status_code == 200, new.text

    def test_a_send_to_the_retired_slug_is_an_error_not_a_crash(self, tmp_path):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        project_store = MagicMock()
        project_store.get_project.return_value = {"workspace": str(tmp_path)}
        mgr = SubAgentManager(
            process_manager=MagicMock(), registry=_bundled(),
            project_store=project_store)
        out = asyncio.run(mgr.start("p1", "gemini-cli", session_id="s1"))
        # A pinned session or automation that still names it gets an error
        # row ("no adapter config for handle 'gemini-cli'"), never a crash.
        assert out.startswith("Error: ") and "'gemini-cli'" in out


# ---------------------------------------------------------------------------
# 3. Paste-the-code login
# ---------------------------------------------------------------------------

class _Ws:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast_global(self, payload):
        self.events.append(dict(payload))

    def of_type(self, t):
        return [e for e in self.events if e["type"] == t]


class _Engine:
    def __init__(self, registry=None, configured=True):
        self._registry = registry or {"antigravity": object()}
        self._configured = configured

    def invalidate_cache(self):
        pass

    def resolve_binary(self, manifest):
        return "/fake/bin/agy"

    def check_credentials(self, manifest, resolved_binary=None):
        return (self._configured, [] if self._configured else ["antigravity_auth"])


CODE = "4/0AbCdEfGh-SECRET-code"

# The real prompt (agy 1.5.13) ends with no newline and waits on stdin.
_FAKE_AGY_LOGIN = r'''
import sys
sys.stdout.write("Open this link in a browser to log in with Google:\n"
                 "https://accounts.google.com/o/oauth2/v2/auth?code_challenge=abc&state=xyz\n"
                 "Then paste the authorization code, or the full URL of the page you land on, and press Enter: ")
sys.stdout.flush()
code = sys.stdin.readline().strip()
if not code:
    print("login cancelled: no authorization code was entered")
    sys.exit(1)
print("exchanging " + code)
print("Logged in as someone@example.com.")
'''


def _stub(tmp_path, body, name="fake_agy.py"):
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


@pytest.fixture
def login_env(monkeypatch):
    ws = _Ws()
    monkeypatch.setattr(settings_routes, "_ws_manager", ws)
    monkeypatch.setattr(settings_routes, "_setup_engine", _Engine())

    async def _no_refresh(slug):
        return None
    monkeypatch.setattr(settings_routes, "_refresh_agent", _no_refresh)
    settings_routes._login_jobs.clear()
    yield ws
    settings_routes._login_jobs.clear()


async def _wait_for(predicate, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


class TestPasteCodeLogin:

    def _seed(self, command, job_id="job1"):
        settings_routes._login_jobs[job_id] = {
            "slug": "antigravity", "status": "running", "command": command,
            "kind": "login"}

    def test_prompt_asks_for_input_and_the_pasted_code_reaches_the_cli(
            self, login_env, tmp_path, caplog):
        ws = login_env
        command = _stub(tmp_path, _FAKE_AGY_LOGIN)
        self._seed(command)

        async def scenario():
            job = asyncio.create_task(settings_routes._run_login_job(
                "antigravity", "job1", command,
                input_prompt="paste the authorization code"))
            await _wait_for(lambda: ws.of_type("login.input_needed"))
            (needed,) = ws.of_type("login.input_needed")
            assert needed == {"type": "login.input_needed", "job_id": "job1",
                              "slug": "antigravity", "kind": "login"}
            # The URL line came first, so the card can offer its fallback link.
            assert any("accounts.google.com" in e["line"]
                       for e in ws.of_type("login.progress"))
            out = await settings_routes.submit_login_input(
                "antigravity", "job1",
                settings_routes.LoginInputRequest(value=f"  {CODE}  "))
            assert out["status"] == "sent"
            await asyncio.wait_for(job, timeout=15)

        with caplog.at_level(logging.DEBUG):
            asyncio.run(scenario())

        (final,) = ws.of_type("login.complete")
        assert final["verified"] is True and final["return_code"] == 0
        lines = [e["line"] for e in ws.of_type("login.progress")]
        assert "exchanging [redacted]" in lines
        assert "Logged in as someone@example.com." in lines
        # The code never appears: not in progress, the log, or the job view.
        assert all(CODE not in line for line in lines)
        assert CODE not in caplog.text
        view = settings_routes._login_job_view("job1", settings_routes._login_jobs["job1"])
        assert CODE not in repr(view)
        assert CODE not in repr(settings_routes._login_jobs["job1"])
        # The pipe does not outlive the process.
        assert "stdin" not in settings_routes._login_jobs["job1"]

    def test_input_route_refuses_a_job_that_is_not_waiting(self, login_env):
        from fastapi import HTTPException
        settings_routes._login_jobs["j"] = {"slug": "antigravity", "status": "running"}
        with pytest.raises(HTTPException) as exc:
            asyncio.run(settings_routes.submit_login_input(
                "antigravity", "j", settings_routes.LoginInputRequest(value="x")))
        assert exc.value.status_code == 409
        with pytest.raises(HTTPException) as exc:
            asyncio.run(settings_routes.submit_login_input(
                "codex", "j", settings_routes.LoginInputRequest(value="x")))
        assert exc.value.status_code == 404

    def test_input_route_rejects_multiline_values(self, login_env):
        from fastapi import HTTPException
        settings_routes._login_jobs["j"] = {
            "slug": "antigravity", "status": "running", "stdin": MagicMock()}
        with pytest.raises(HTTPException) as exc:
            asyncio.run(settings_routes.submit_login_input(
                "antigravity", "j",
                settings_routes.LoginInputRequest(value="a\nrm -rf /")))
        assert exc.value.status_code == 400

    def test_a_login_without_the_prompt_keeps_its_stdin(self, login_env, tmp_path,
                                                        monkeypatch):
        seen = {}
        real = asyncio.create_subprocess_shell

        async def spy(cmd, **kwargs):
            seen.update(kwargs)
            return await real(cmd, **kwargs)
        monkeypatch.setattr(settings_routes.asyncio, "create_subprocess_shell", spy)
        command = _stub(tmp_path, 'print("Opening browser…")\n')
        self._seed(command)
        asyncio.run(settings_routes._run_login_job("antigravity", "job1", command))
        assert "stdin" not in seen
        assert not login_env.of_type("login.input_needed")

    def test_the_prompt_line_is_matched_without_a_trailing_newline(self, login_env,
                                                                   tmp_path,
                                                                   monkeypatch):
        """agy's prompt has no newline; a readline-only runner would sit on
        it until the idle timeout. Here the CLI is killed by a short idle
        timeout after the ask, which proves the ask came first."""
        monkeypatch.setattr(settings_routes, "LOGIN_IDLE_TIMEOUT_SECONDS", 1.0)
        command = _stub(tmp_path, _FAKE_AGY_LOGIN)
        self._seed(command)
        asyncio.run(settings_routes._run_login_job(
            "antigravity", "job1", command,
            input_prompt="paste the authorization code"))
        assert login_env.of_type("login.input_needed")
        (failed,) = login_env.of_type("login.failed")
        assert failed.get("timed_out") is True

    def test_login_route_hands_the_manifest_prompt_to_the_runner(self, login_env,
                                                                 monkeypatch):
        registry = _bundled()
        monkeypatch.setattr(settings_routes, "_setup_engine", _Engine(registry=registry))
        monkeypatch.setattr(settings_routes, "_resolve_setup_command",
                            lambda slug, action: "agy acp --login")
        calls = []

        async def fake(slug, job_id, command, kind="login", **kwargs):
            calls.append((slug, command, kwargs))
        monkeypatch.setattr(settings_routes, "_run_login_job", fake)

        async def go(slug):
            await settings_routes.trigger_sub_agent_login(slug)
            await asyncio.sleep(0)
        asyncio.run(go("antigravity"))
        asyncio.run(go("claude-code"))
        assert calls[0] == ("antigravity", "agy acp --login",
                            {"input_prompt": "paste the authorization code"})
        assert calls[1] == ("claude-code", "agy acp --login", {})


# ---------------------------------------------------------------------------
# 4. Region refusal through the real ACP transport
# ---------------------------------------------------------------------------

async def _start_acp(tmp_path, **kwargs):
    from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
    t = ACPSDKTransport(**kwargs)
    await t.start(sys.executable, [str(ACP_FIXTURE)], str(tmp_path))
    while not t._event_queue.empty():
        t._event_queue.get_nowait()
    return t


async def _turn_events(t, message):
    await t.dispatch(message)
    events = []
    while True:
        ev = await asyncio.wait_for(t._event_queue.get(), timeout=10)
        events.append(ev)
        if ev.event_type == "turn_complete":
            return events


def _agy_kwargs():
    m = _load()
    return {"refusal_patterns": m.runtime.refusal_patterns, "agent_name": m.name}


class TestRegionRefusal:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [REFUSAL, ELIGIBILITY])
    async def test_a_refusal_turn_is_an_error_not_a_reply(self, tmp_path, text):
        t = await _start_acp(tmp_path, **_agy_kwargs())
        try:
            events = await _turn_events(t, f"say:{text}")
            kinds = [e.event_type for e in events]
            assert "message" not in kinds
            (err,) = [e for e in events if e.event_type == "error"]
            assert err.data["reason"] == "region_unavailable"
            assert err.data["name"] == "Antigravity"
            assert err.data["error"] == text
            done = events[-1]
            assert done.data["cause"] == "error"
            assert done.data["reason"] == "region_unavailable"
            # The session stays usable: the next turn is an ordinary reply.
            events = await _turn_events(t, "say:PONG")
            assert [e.raw_text for e in events if e.event_type == "message"] == ["PONG"]
            assert events[-1].data["cause"] == "success"
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_a_reply_that_quotes_the_code_is_a_reply(self, tmp_path):
        t = await _start_acp(tmp_path, **_agy_kwargs())
        try:
            quoted = f"Here is what the CLI printed earlier:\n\n{REFUSAL}\n\nIt means the region is blocked."
            events = await _turn_events(t, f"say:{quoted}")
            assert [e.event_type for e in events if e.event_type in ("message", "error")] == ["message"]
            assert events[-1].data["cause"] == "success"
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_a_turn_that_ran_a_tool_is_never_a_refusal(self, tmp_path):
        t = await _start_acp(tmp_path, **_agy_kwargs())
        try:
            events = await _turn_events(t, f"tool-then-say:{REFUSAL}")
            assert "error" not in [e.event_type for e in events]
            assert events[-1].data["cause"] == "success"
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_without_patterns_the_refusal_text_stays_a_reply(self, tmp_path):
        t = await _start_acp(tmp_path)
        try:
            events = await _turn_events(t, f"say:{REFUSAL}")
            assert [e.raw_text for e in events if e.event_type == "message"] == [REFUSAL]
            assert events[-1].data["cause"] == "success"
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_an_agent_error_with_the_refusal_text_carries_the_reason(self, tmp_path):
        t = await _start_acp(tmp_path, **_agy_kwargs())
        try:
            events = await _turn_events(t, f"raise:{ELIGIBILITY}")
            (err,) = [e for e in events if e.event_type == "error"]
            assert err.data["reason"] == "region_unavailable"
            assert events[-1].data["cause"] == "error"
            events = await _turn_events(t, "raise:something else broke")
            (err,) = [e for e in events if e.event_type == "error"]
            assert "reason" not in err.data
        finally:
            await t.stop()

    def test_a_broken_pattern_is_dropped_not_fatal(self):
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        t = ACPSDKTransport(refusal_patterns=[
            {"regex": "(unclosed", "reason": "x"},
            {"regex": "ok", "reason": "y"}])
        assert [r for _p, r in t._refusal_patterns] == ["y"]

    def test_the_manager_hands_the_manifest_patterns_to_the_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        mgr = SubAgentManager(process_manager=MagicMock(), registry=_bundled(),
                              project_store=MagicMock())
        t = mgr._resolve_transport(_load(), {})
        assert t._agent_name == "Antigravity"
        assert [r for _p, r in t._refusal_patterns] == ["region_unavailable"] * 2
        kimi = ManifestLoader.load(os.path.join(MANIFESTS, "kimi-code.yaml"))
        assert mgr._resolve_transport(kimi, {})._refusal_patterns == []


class TestReasonReachesTheRow:

    def test_process_manager_hands_reason_and_name_to_on_error(self):
        from agent_os.daemon_v2.process_manager import ProcessManager

        class Chunk:
            def __init__(self, text, chunk_type, metadata=None):
                self.text, self.chunk_type = text, chunk_type
                self.timestamp = "2026-11-22T00:00:00+00:00"
                self.metadata = metadata or {}

        class Adapter:
            _transport = None

            async def read_stream(self):
                yield Chunk(REFUSAL, "error", {"error": REFUSAL,
                                              "reason": "region_unavailable",
                                              "name": "Antigravity"})
                yield Chunk("", "turn_complete", {"cause": "error", "session_id": "s"})

        lifecycle = MagicMock()
        lifecycle.on_error = AsyncMock()
        lifecycle.on_thread_update = AsyncMock()
        ws = MagicMock()
        pm = ProcessManager(ws, MagicMock(), lifecycle)

        class Tx:
            filepath = "/tmp/_agy_tx.jsonl"

            def append(self, entry):
                pass

        async def go():
            await pm.start("p", "antigravity", Adapter(), transcript=Tx(), session_id="sess")
            await pm._tasks[pm._key("p", "sess", "antigravity")]
        asyncio.run(go())
        lifecycle.on_error.assert_awaited_once()
        args, kwargs = lifecycle.on_error.call_args
        assert args[2] == REFUSAL
        assert kwargs["detail"] == {"reason": "region_unavailable", "name": "Antigravity"}

    def test_on_error_puts_the_reason_on_the_row_and_the_event(self):
        from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        am.has_running_children = MagicMock(return_value=False)
        ws = MagicMock()
        obs = LifecycleObserver(am, ws)
        asyncio.run(obs.on_error(
            "p", "antigravity", REFUSAL, "/tmp/t.jsonl", session_id="s",
            detail={"reason": "region_unavailable", "name": "Antigravity",
                    "junk": "dropped"}))
        meta = am.inject_system_message.call_args.kwargs["meta"]
        assert meta["kind"] == "error"
        assert meta["reason"] == "region_unavailable"
        assert meta["name"] == "Antigravity"
        assert "junk" not in meta
        event = ws.broadcast.call_args.args[1]
        assert event["type"] == "sub_agent.error"
        assert event["reason"] == "region_unavailable"
        # The content keeps the vendor's own words for the log and the manager.
        content = am.inject_system_message.call_args.args[1]
        assert "(#3501)" in content

    def test_on_error_without_detail_is_unchanged(self):
        from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
        am = MagicMock()
        am.inject_system_message = AsyncMock(return_value="delivered")
        am.has_running_children = MagicMock(return_value=False)
        obs = LifecycleObserver(am, MagicMock())
        asyncio.run(obs.on_error("p", "codex", "boom", "/tmp/t.jsonl", session_id="s"))
        meta = am.inject_system_message.call_args.kwargs["meta"]
        assert "reason" not in meta and "name" not in meta

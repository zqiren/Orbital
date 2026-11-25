# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Honest agent status (2026-11-25): an agent reads ready ONLY when a message
to it would work as far as Orbital can cheaply tell; anything else names the
issue (``not_ready: {reason, detail}``).

The case that started it: Antigravity signed in from a refused region. Its
probe (``agy models``) exits 0 and lists models there, so the agent read
ready, the Welcome page's Start opened on it, a keyless project picked it,
and the first message failed. The refusal is only visible in a real turn
(the manifest's ``refusal_patterns`` turn it into ``reason:
region_unavailable``), so the first refused turn is remembered in memory, the
status turns not ready, and a successful turn, a sign-in/out or the user's
"Check again" (POST …/refresh) forgets it.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCredential,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.agents.setup_types import AgentSetupStatus, not_ready, status_ready
from agent_os.daemon_v2.sub_agent_visibility import worker_ready

REFUSAL_TEXT = ("⚠ You do not have a valid license of this product. Please contact your "
                "administrator to request a license. (#3501)")


def _status(**kw) -> AgentSetupStatus:
    base = dict(slug="antigravity", name="Antigravity", installed=True,
                binary_path="/x/agy", version="1.5.16", dependencies_met=True,
                missing_dependencies=[], credentials_configured=True,
                missing_credentials=[])
    base.update(kw)
    return AgentSetupStatus(**base)


# ---------------------------------------------------------------------------
# The one readiness definition and its reason
# ---------------------------------------------------------------------------

class TestNotReadyReason:

    def test_a_ready_agent_has_no_reason(self):
        s = _status()
        assert status_ready(s) is True
        assert not_ready(s) is None
        assert worker_ready(s) is True

    def test_a_refused_turn_makes_it_not_ready_with_the_vendor_words(self):
        s = _status(refusal={"reason": "region_unavailable", "detail": REFUSAL_TEXT})
        assert status_ready(s) is False
        assert worker_ready(s) is False
        assert not_ready(s) == {"reason": "region_unavailable", "detail": REFUSAL_TEXT}

    def test_an_expired_sign_in_is_not_ready(self):
        s = _status(auth_expired=True)
        assert status_ready(s) is False
        assert worker_ready(s) is False
        assert not_ready(s) == {"reason": "auth_expired", "detail": None}

    def test_signed_out_and_key_missing_are_told_apart_by_the_action(self):
        out = _status(credentials_configured=False, missing_credentials=["a"],
                      setup_actions=[SimpleNamespace(action="run_cli_auth")])
        key = _status(credentials_configured=False, missing_credentials=["k"],
                      setup_actions=[SimpleNamespace(action="configure_credential")])
        assert not_ready(out) == {"reason": "signed_out", "detail": None}
        assert not_ready(key) == {"reason": "key_missing", "detail": None}

    def test_update_handshake_and_dependency(self):
        assert not_ready(_status(update_needed=True, min_version="2.0")) == {
            "reason": "update_needed", "detail": None}
        assert not_ready(_status(handshake={"state": "failed", "detail": "EOF"})) == {
            "reason": "handshake_failed", "detail": "EOF"}
        assert not_ready(_status(dependencies_met=False, missing_dependencies=["node", "npm"])) == {
            "reason": "missing_dependency", "detail": "node, npm"}

    def test_not_installed(self):
        assert not_ready(_status(installed=False)) == {"reason": "not_installed", "detail": None}

    def test_the_first_blocking_issue_wins(self):
        """An update is what unblocks a too-old CLI, before its sign-in."""
        s = _status(update_needed=True, credentials_configured=False,
                    refusal={"reason": "region_unavailable", "detail": "x"})
        assert not_ready(s)["reason"] == "update_needed"

    def test_a_pending_handshake_is_still_ready(self):
        """The routine sweep reports an unfinished handshake as unknown: as
        far as Orbital can cheaply tell, it works."""
        s = _status(handshake={"state": "unknown", "detail": None})
        assert status_ready(s) is True and not_ready(s) is None

    def test_a_mocked_status_never_trips_the_new_checks(self):
        s = SimpleNamespace(slug="x", installed=True, dependencies_met=True,
                            credentials_configured=True, update_needed=False,
                            handshake=None, auth_expired=MagicMock(), refusal=MagicMock())
        assert worker_ready(s) is True


# ---------------------------------------------------------------------------
# The engine remembers a refused turn
# ---------------------------------------------------------------------------

def _engine(refusal_reasons=("region_unavailable",)):
    manifest = AgentManifest(
        manifest_version="1", name="Antigravity", slug="antigravity", description="",
        author="t", version="1",
        runtime=ManifestRuntime(adapter="cli", command="agy", transport="acp-sdk",
                                args=["acp"],
                                refusal_patterns=[{"regex": "x", "reason": r}
                                                  for r in refusal_reasons]),
        setup=ManifestSetup(check_command="agy --version", credentials=[ManifestCredential(
            key="antigravity_auth", label="Google", type="oauth_cli", required=True,
            check_command="agy models", setup_command="agy acp --login")]),
    )
    registry = AgentRegistry()
    registry.register(manifest)
    engine = SetupEngine(registry=registry)
    engine._resolved_paths["antigravity"] = "/x/agy"
    # No handshake spawn in a unit test.
    engine._handshake_for = lambda *a, **k: None
    return engine


def _run(cmd, *a, **kw):
    if "models" in str(cmd):
        return subprocess.CompletedProcess(args="", returncode=0, stdout="gemini\n", stderr="")
    return subprocess.CompletedProcess(args="", returncode=0, stdout="1.5.16\n", stderr="")


class TestRefusalRecord:

    def test_a_declared_refusal_turns_the_status_not_ready(self):
        engine = _engine()
        assert engine.note_refusal("antigravity", "region_unavailable", f"  {REFUSAL_TEXT}\n") is True
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            status = engine.check_agent("antigravity")
        assert status.credentials_configured is True  # still signed in
        assert status.refusal == {"reason": "region_unavailable", "detail": REFUSAL_TEXT}
        assert status_ready(status) is False

    def test_a_reason_the_manifest_does_not_declare_is_not_recorded(self):
        """usage_limit rides the same error field (spec 197) and has its own
        store; only the manifest's refusal reasons are "not available"."""
        engine = _engine()
        assert engine.note_refusal("antigravity", "usage_limit", "limit") is False
        assert engine.note_refusal("antigravity", "", "x") is False
        assert engine.note_refusal("no-such-agent", "region_unavailable", "x") is False
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert engine.check_agent("antigravity").refusal is None

    def test_the_vendor_text_is_capped(self):
        engine = _engine()
        engine.note_refusal("antigravity", "region_unavailable", "y" * 5000)
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert len(engine.check_agent("antigravity").refusal["detail"]) <= 500

    def test_a_successful_turn_clears_it(self):
        engine = _engine()
        engine.note_refusal("antigravity", "region_unavailable", REFUSAL_TEXT)
        engine.clear_refusal("antigravity")
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert engine.check_agent("antigravity").refusal is None

    def test_check_again_forgets_it(self):
        engine = _engine()
        engine.note_refusal("antigravity", "region_unavailable", REFUSAL_TEXT)
        engine.forget_refusals()
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert engine.check_agent("antigravity").refusal is None

    def test_a_routine_resweep_keeps_it(self):
        """The probe cannot see the region: a sweep must not wipe the record."""
        engine = _engine()
        engine.note_refusal("antigravity", "region_unavailable", REFUSAL_TEXT)
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            engine.check_all()
            engine.invalidate_cache()
            assert engine.check_all()[0].refusal is not None

    def test_recording_patches_the_cached_sweep_and_announces_it(self):
        engine = _engine()
        published = []
        engine.set_status_listener(published.append)
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert engine.check_all()[0].refusal is None
            published.clear()
            engine.note_refusal("antigravity", "region_unavailable", REFUSAL_TEXT)
            # Served from the patched sweep: no re-probe needed.
            with patch("agent_os.agents.setup_engine.subprocess.run",
                       side_effect=AssertionError("no re-probe")):
                assert engine.check_all()[0].refusal["reason"] == "region_unavailable"
        assert published, "the change must reach the WS (sub_agents.updated)"

    def test_a_signed_out_probe_drops_it(self):
        """Signed out speaks for itself; after the next sign-in the agent must
        not still read refused from the old account."""
        engine = _engine()
        engine.note_refusal("antigravity", "region_unavailable", REFUSAL_TEXT)

        def signed_out(cmd, *a, **kw):
            if "models" in str(cmd):
                return subprocess.CompletedProcess(args="", returncode=1, stdout="",
                                                   stderr="Please sign in")
            return _run(cmd)
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=signed_out):
            assert engine.check_agent("antigravity").refusal is None
        with patch("agent_os.agents.setup_engine.subprocess.run", side_effect=_run):
            assert engine.check_agent("antigravity").refusal is None


# ---------------------------------------------------------------------------
# The wire: settings list, the pin menu's list, the single-agent status
# ---------------------------------------------------------------------------

class TestWire:

    def test_settings_entry_carries_not_ready(self, monkeypatch):
        from agent_os.api.routes import settings as settings_routes
        monkeypatch.setattr(settings_routes, "_sub_agent_config_store", None)
        entry = settings_routes._build_sub_agent_status_entry(
            _status(refusal={"reason": "region_unavailable", "detail": REFUSAL_TEXT}))
        assert entry["ready"] is False
        assert entry["credentials_configured"] is True
        assert entry["not_ready"] == {"reason": "region_unavailable", "detail": REFUSAL_TEXT}
        ready = settings_routes._build_sub_agent_status_entry(_status())
        assert ready["ready"] is True and ready["not_ready"] is None

    @pytest.mark.asyncio
    async def test_available_agents_carries_not_ready(self, monkeypatch):
        from agent_os.api.routes import agents_v2
        engine = MagicMock()
        engine.check_all.return_value = [
            _status(refusal={"reason": "region_unavailable", "detail": REFUSAL_TEXT}),
            _status(slug="codex", name="Codex")]
        monkeypatch.setattr(agents_v2, "_setup_engine", engine)
        out = {e["slug"]: e for e in await agents_v2.available_agents()}
        assert out["antigravity"]["ready"] is False
        assert out["antigravity"]["not_ready"]["reason"] == "region_unavailable"
        assert out["codex"]["ready"] is True and out["codex"]["not_ready"] is None

    @pytest.mark.asyncio
    async def test_refresh_forgets_refusals(self, monkeypatch):
        """"Check again" is the user saying something changed (a VPN)."""
        from agent_os.api.routes import settings as settings_routes
        engine = MagicMock()
        engine.check_all.return_value = []
        engine._registry.list_all.return_value = []
        monkeypatch.setattr(settings_routes, "_setup_engine", engine)
        monkeypatch.setattr("agent_os.platform.shell_path.merge_shell_path_into_environ",
                            lambda: None)
        await settings_routes.refresh_sub_agent_status()
        engine.forget_refusals.assert_called_once_with()


# ---------------------------------------------------------------------------
# A refused turn reaches the engine; a good one clears it
# ---------------------------------------------------------------------------

class _ProcessManager:
    def set_turn_closed_callback(self, callback):
        pass

    def set_permission_request_callback(self, callback):
        pass


class _Engine:
    def __init__(self):
        self.refused: list[tuple] = []
        self.cleared: list[str] = []

    def note_auth_failure(self, slug, text):
        return False

    def clear_auth_failure(self, slug):
        pass

    def note_refusal(self, slug, reason, text):
        self.refused.append((slug, reason, text))
        return True

    def clear_refusal(self, slug):
        self.cleared.append(slug)


class TestTurnOutcome:

    @pytest.mark.asyncio
    async def test_a_refused_turn_is_recorded(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        engine = _Engine()
        manager = SubAgentManager(_ProcessManager(), setup_engine=engine)
        await manager._on_prompt_turn_closed(
            "p1", "antigravity", session_id="s1", cause="error",
            refusal={"reason": "region_unavailable", "text": REFUSAL_TEXT})
        assert engine.refused == [("antigravity", "region_unavailable", REFUSAL_TEXT)]

    @pytest.mark.asyncio
    async def test_a_successful_turn_clears_the_refusal(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        engine = _Engine()
        manager = SubAgentManager(_ProcessManager(), setup_engine=engine)
        await manager._on_prompt_turn_closed("p1", "antigravity", session_id="s1",
                                             cause="success")
        assert engine.cleared == ["antigravity"]
        assert engine.refused == []

    @pytest.mark.asyncio
    async def test_an_engine_without_the_hooks_is_harmless(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        manager = SubAgentManager(_ProcessManager(), setup_engine=SimpleNamespace())
        await manager._on_prompt_turn_closed(
            "p1", "antigravity", session_id="s1", cause="error",
            refusal={"reason": "region_unavailable", "text": "x"})

    def test_process_manager_hands_the_refusal_to_turn_closed(self):
        """The error chunk's reason and the vendor's words reach
        ``on_turn_closed`` (captured before the turn's state is reset)."""
        import asyncio
        from unittest.mock import AsyncMock
        from agent_os.daemon_v2.process_manager import ProcessManager

        class Chunk:
            def __init__(self, text, chunk_type, metadata=None):
                self.text, self.chunk_type = text, chunk_type
                self.timestamp = "2026-11-25T00:00:00+00:00"
                self.metadata = metadata or {}

        class Adapter:
            _transport = None

            async def read_stream(self):
                yield Chunk(REFUSAL_TEXT, "error", {"error": REFUSAL_TEXT,
                                                   "reason": "region_unavailable",
                                                   "name": "Antigravity"})
                yield Chunk("", "turn_complete", {"cause": "error", "session_id": "s"})
                yield Chunk("PONG", "response")
                yield Chunk("", "turn_complete", {"cause": "success", "session_id": "s"})

        lifecycle = MagicMock()
        lifecycle.on_error = AsyncMock()
        lifecycle.on_completed = AsyncMock()
        lifecycle.on_thread_update = AsyncMock()
        pm = ProcessManager(MagicMock(), MagicMock(), lifecycle)
        closed = []

        async def on_turn_closed(project_id, handle, **kw):
            closed.append(kw)
        pm.on_turn_closed = on_turn_closed

        class Tx:
            filepath = "/tmp/_agy_tx_1125.jsonl"

            def append(self, entry):
                pass

        async def go():
            await pm.start("p", "antigravity", Adapter(), transcript=Tx(), session_id="sess")
            await pm._tasks[pm._key("p", "sess", "antigravity")]
        asyncio.run(go())
        assert [c.get("cause") for c in closed] == ["error", "success"]
        assert closed[0]["refusal"] == {"reason": "region_unavailable", "text": REFUSAL_TEXT}
        # The next turn carries no stale refusal.
        assert "refusal" not in closed[1]


async def _async_noop(*a, **k):
    return None


# ---------------------------------------------------------------------------
# A keyless send whose only agent is not ready names it and the reason
# ---------------------------------------------------------------------------

class TestKeylessPreflight:

    @pytest.mark.asyncio
    async def test_a_refused_agent_is_named_with_its_reason(self, tmp_path):
        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.models import AgentConfig
        from agent_os.daemon_v2.provider_errors import ProviderConfigError
        refused = _status(refusal={"reason": "region_unavailable", "detail": REFUSAL_TEXT})
        engine = MagicMock()
        engine.check_all.return_value = [refused]
        engine.check_all_cached.return_value = [refused]
        registry = MagicMock()
        registry.get.return_value = SimpleNamespace(
            slug="antigravity", setup=SimpleNamespace(credentials=[]))
        mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                           sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
                           process_manager=MagicMock(), registry=registry, setup_engine=engine)
        mgr._ws.broadcast = MagicMock()
        mgr._project_store.get_project.return_value = {
            "name": "proj", "workspace": str(tmp_path), "disabled_sub_agents": []}
        mgr._platform_provider = None
        mgr._build_agent_config_from_project = lambda pid: AgentConfig(
            workspace=str(tmp_path), model="m", api_key="", provider="stub-provider",
            sdk="openai", project_name="proj")
        mgr.start_agent = MagicMock(side_effect=AssertionError("must not start"))
        with pytest.raises(ProviderConfigError) as info:
            await mgr.inject_message("proj", "hello", nonce="n-1", session_id="proj_x")
        err = info.value
        assert err.code == "agent_not_ready"
        assert err.fields == {"agent": "antigravity", "agent_name": "Antigravity",
                              "reason": "region_unavailable", "detail": REFUSAL_TEXT}
        assert REFUSAL_TEXT in str(err)

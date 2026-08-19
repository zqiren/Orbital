# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Telemetry data integrity (spec 063): three instruments that lied.

- §2 the sender transmitted from every test process (~98% of the stored
  install fleet): guarded by an env var plus a structural temp-dir backstop,
  on ``start()`` ALONE so ``run_cycle()`` stays drivable by the existing
  sender tests and local spooling is untouched.
- §3 ``session_created`` sat in one route, so 3 of 4 callers were uncounted.
- §4 ``llm_error`` carried a code and no provider, so no failure could be
  attributed to a vendor.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.telemetry.identity import InstallIdentity
from agent_os.telemetry.sender import TelemetrySender, env_disabled, is_temp_path
from agent_os.telemetry.spool import Spool

# A path that is emphatically NOT under any temp root. Never written to: the
# guard is a string/realpath check and `start()` is stopped before its first
# cycle in these tests.
REAL_DATA_DIR = Path.home() / "Library" / "Application Support" / "Orbital-test-fake"


class FakePost:
    def __init__(self):
        self.calls = []

    async def __call__(self, url, payload, headers):
        self.calls.append((url, payload, headers))
        return 200


def make_sender(data_dir, spool_dir) -> TelemetrySender:
    return TelemetrySender(
        data_dir,
        InstallIdentity(spool_dir),
        Spool(spool_dir),
        is_enabled=lambda: True,
        endpoint="https://example.invalid/ingest",
        post=FakePost(),
    )


class TestEnvGuard:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "anything"])
    def test_truthy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("AGENT_OS_TELEMETRY_DISABLED", value)
        assert env_disabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsey_values_do_not_disable(self, monkeypatch, value):
        monkeypatch.setenv("AGENT_OS_TELEMETRY_DISABLED", value)
        assert env_disabled() is False

    def test_unset_does_not_disable(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_TELEMETRY_DISABLED", raising=False)
        assert env_disabled() is False

    async def test_start_is_inert_when_set(self, monkeypatch, tmp_path):
        # Env var set (as the suite-wide conftest fixture does) → no task, so
        # no cycle, so no POST — even from a real-looking data dir.
        monkeypatch.setenv("AGENT_OS_TELEMETRY_DISABLED", "1")
        sender = make_sender(REAL_DATA_DIR, tmp_path)
        sender.start()
        assert sender._task is None

    async def test_run_cycle_is_not_guarded(self, monkeypatch, tmp_path):
        """The guard is on start() ALONE — run_cycle() is the injected-post
        seam the rollup/sender tests drive directly, and must keep working."""
        monkeypatch.setenv("AGENT_OS_TELEMETRY_DISABLED", "1")
        post = FakePost()
        sender = TelemetrySender(
            tmp_path, InstallIdentity(tmp_path), Spool(tmp_path),
            is_enabled=lambda: True,
            endpoint="https://example.invalid/ingest", post=post,
        )
        await sender.run_cycle()
        assert len(post.calls) == 1

    async def test_start_runs_when_unset_and_dir_is_real(self, monkeypatch, tmp_path):
        """Positive control: neither guard fires → the loop task is created."""
        monkeypatch.delenv("AGENT_OS_TELEMETRY_DISABLED", raising=False)
        sender = make_sender(REAL_DATA_DIR, tmp_path)
        sender.start()
        try:
            assert sender._task is not None
        finally:
            sender.stop()


class TestTempDirGuard:
    def test_pytest_tmp_path_is_temp(self, tmp_path):
        assert is_temp_path(tmp_path) is True

    def test_slash_tmp_is_temp(self):
        # Symlinked to /private/tmp on macOS — realpath'd on both sides.
        assert is_temp_path("/tmp/orbital-data") is True

    def test_home_data_dir_is_not_temp(self):
        assert is_temp_path(REAL_DATA_DIR) is False

    def test_repo_data_dir_is_not_temp(self):
        assert is_temp_path(Path(__file__).resolve().parents[2] / "orbital-data") is False

    async def test_start_is_inert_for_temp_data_dir(self, monkeypatch, tmp_path):
        """The backstop the env var cannot forget: a CI job or a
        daemon-spawning test with no env var set still cannot transmit."""
        monkeypatch.delenv("AGENT_OS_TELEMETRY_DISABLED", raising=False)
        sender = make_sender(tmp_path, tmp_path)
        sender.start()
        assert sender._task is None

    async def test_suppression_logs_once(self, monkeypatch, tmp_path, caplog):
        monkeypatch.delenv("AGENT_OS_TELEMETRY_DISABLED", raising=False)
        sender = make_sender(tmp_path, tmp_path)
        with caplog.at_level("INFO", logger="agent_os.telemetry.sender"):
            sender.start()
            sender.start()
        lines = [r for r in caplog.records if "telemetry sender not started" in r.message]
        assert len(lines) == 1

    def test_spooling_is_unaffected_by_either_guard(self, monkeypatch, tmp_path):
        """Guards suppress transmission only — the local spool and the
        settings viewer's next-payload preview keep working."""
        monkeypatch.setenv("AGENT_OS_TELEMETRY_DISABLED", "1")
        from agent_os import telemetry

        telemetry.reset_for_tests()
        try:
            sender = telemetry.configure(tmp_path, is_enabled=lambda: True)
            telemetry.emit("session_created")
            sender.start()
            assert sender._task is None
            assert sender.next_pending_payload()["counters"]["sessions"] == 1
        finally:
            telemetry.reset_for_tests()


def _manager():
    from agent_os.daemon_v2.agent_manager import AgentManager

    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        settings_store=MagicMock(),
        credential_store=MagicMock(),
        browser_manager=MagicMock(),
        provider_registry=MagicMock(),
    )


class TestSessionsCountedAtChokepoint:
    """§3: the emit sat in the route, so 3 of 4 callers were uncounted."""

    async def test_new_session_emits_for_non_route_caller(self, tmp_path):
        """Called directly — the way the queue dispatcher, the cold-start scan
        and the workbench spawn call it, none of which touch the route."""
        import json

        from agent_os import telemetry

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: True)
            manager = _manager()
            # The uuid mint reads the project's display name.
            manager._project_store.get_project.return_value = {"name": "demo"}
            result = await manager.new_session("proj_x")
            assert result["status"] == "ok"

            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry" / "events.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            assert [r["event"] for r in rows] == ["session_created"]
            # No ids leak into the spool row.
            assert "project_id" not in rows[0] and "session_id" not in rows[0]

            identity = json.loads(
                (tmp_path / "telemetry" / "install.json").read_text(encoding="utf-8")
            )
            assert identity["milestones"] == {"first_session": True}
        finally:
            telemetry.reset_for_tests()

    async def test_route_no_longer_emits_directly(self, tmp_path, monkeypatch):
        """The route must delegate: an emit left there would double-count the
        button relative to the other three callers. Driven with a stub manager
        so the only emit that could appear is the route's own."""
        from agent_os import telemetry
        from agent_os.api.routes import agents_v2

        stub = MagicMock()

        async def _new_session(project_id, session_id=None):
            return {"status": "ok", "session_id": "s1", "session_uuid": "s1"}

        stub.new_session = _new_session
        monkeypatch.setattr(agents_v2, "_agent_manager", stub)

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: True)
            await agents_v2.new_session("proj_x")
            assert not (tmp_path / "telemetry" / "events.jsonl").exists()
        finally:
            telemetry.reset_for_tests()


class TestErrorAttribution:
    """§4: `llm_error` shipped a code and nothing else — 17 rows on the dev
    machine, none attributable to a vendor. Answering "why so many
    invalid_api_key?" required leaving telemetry and reading four days of
    daemon.log."""

    def _spool_rows(self, tmp_path):
        import json

        path = tmp_path / "telemetry" / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_llm_error_carries_running_handles_provider(self, tmp_path):
        from agent_os import telemetry
        from agent_os.daemon_v2.agent_manager import ProjectHandle
        from agent_os.daemon_v2.models import make_session_key
        from agent_os.daemon_v2.provider_errors import ProviderConfigError

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: True)
            manager = _manager()
            manager._handles[make_session_key("proj_x", "sess_1")] = ProjectHandle(
                session=MagicMock(), loop=MagicMock(), provider=MagicMock(),
                registry=MagicMock(), context_manager=MagicMock(),
                interceptor=MagicMock(), task=None,
                config_snapshot={"provider": "deepseek"},
            )
            manager._record_loop_error(
                "proj_x", "sess_1",
                ProviderConfigError("invalid_api_key", "401 Unauthorized"),
            )
            rows = self._spool_rows(tmp_path)
            assert len(rows) == 1
            assert rows[0]["event"] == "llm_error"
            assert rows[0]["error_code"] == "invalid_api_key"
            assert rows[0]["provider"] == "deepseek"
        finally:
            telemetry.reset_for_tests()

    def test_llm_error_falls_back_to_unknown(self, tmp_path):
        """A failure with no handle and no caller-supplied provider — the
        construction-failure case §8 calls out — must still emit a value."""
        from agent_os import telemetry
        from agent_os.daemon_v2.provider_errors import ProviderConfigError

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: True)
            _manager()._record_loop_error(
                "proj_x", "sess_1",
                ProviderConfigError("missing_api_key", "No LLM API key configured"),
            )
            assert self._spool_rows(tmp_path)[0]["provider"] == "unknown"
        finally:
            telemetry.reset_for_tests()

    def test_start_failure_uses_the_callers_config_provider(self, tmp_path):
        """Start failures happen before the handle exists, so the config the
        caller already holds is the only provider source."""
        from agent_os import telemetry

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: True)
            manager = _manager()
            manager._record_start_failure(
                "proj_x", "sess_1", MagicMock(),
                RuntimeError("boom"), provider="minimax",
            )
            assert self._spool_rows(tmp_path)[0]["provider"] == "minimax"
        finally:
            telemetry.reset_for_tests()

    def test_llm_error_still_spools_with_the_toggle_off(self, tmp_path):
        """Q2 (spec 046) is unchanged by the new field: the provider enum has
        the same local-only debugging value as the code."""
        from agent_os import telemetry
        from agent_os.daemon_v2.provider_errors import ProviderConfigError

        telemetry.reset_for_tests()
        try:
            telemetry.configure(tmp_path, is_enabled=lambda: False)
            _manager()._record_loop_error(
                "proj_x", "sess_1",
                ProviderConfigError("invalid_api_key", "401"),
            )
            rows = self._spool_rows(tmp_path)
            assert [r["event"] for r in rows] == ["llm_error"]
        finally:
            telemetry.reset_for_tests()


class TestBroadcastStampsProjectId:
    """Every WS payload needs project_id: the backend routes by subscription
    and never stamped it, while the frontend filters events on it."""

    def test_project_id_is_stamped(self):
        manager = _manager()
        manager._broadcast("proj_x", {"type": "agent.status"}, session_id="sess_1")
        payload = manager._ws.broadcast.call_args[0][1]
        assert payload["project_id"] == "proj_x"
        assert payload["session_id"] == "sess_1"

    def test_caller_supplied_project_id_wins(self):
        manager = _manager()
        manager._broadcast(
            "proj_x", {"type": "agent.status", "project_id": "proj_other"},
        )
        assert manager._ws.broadcast.call_args[0][1]["project_id"] == "proj_other"

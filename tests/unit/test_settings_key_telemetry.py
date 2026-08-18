# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""``key_set`` telemetry fires on every route that actually sets a key.

Three routes write an API key; only ``PUT /settings/api-key`` emitted. The
other two were silent, so the counter under-reported:

- ``PUT /settings`` — the write the provider dropdown uses, i.e. where most
  keys really arrive.
- ``POST /settings/sub-agents/{slug}/api-key`` — hands the key to the CLI's
  own ingestion command.

No real credential is stored: the keyring backend is in-memory for the suite
and the sub-agent CLI is a stub script.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest
from fastapi.testclient import TestClient

from agent_os.api.routes import settings as settings_routes


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict | None]] = []
        self.latched: list[str] = []

    def emit(self, event: str, fields: dict | None = None, **_kw) -> None:
        self.events.append((event, fields))

    def latch(self, milestone: str) -> None:
        self.latched.append(milestone)

    def fields_for(self, event: str) -> list[dict | None]:
        return [f for name, f in self.events if name == event]


@pytest.fixture
def recorder(monkeypatch):
    rec = _RecordingTelemetry()
    monkeypatch.setattr(settings_routes, "telemetry", rec)
    return rec


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


class TestGlobalSettingsKeyTelemetry:

    def test_put_settings_with_a_key_emits_key_set(self, client, recorder):
        resp = client.put("/api/v2/settings",
                          json={"llm_api_key": "sk-test-1234567890"})
        assert resp.status_code == 200
        assert recorder.fields_for("key_set"), (
            "PUT /settings wrote the key but emitted no key_set"
        )
        assert "key_set" in recorder.latched

    def test_provider_reported_is_the_one_set_in_the_same_request(
            self, client, recorder):
        """Key and provider usually change together in one PUT; the emit has
        to report the provider the key belongs to, not the previous one."""
        client.put("/api/v2/settings", json={"llm_provider": "deepseek"})
        recorder.events.clear()

        client.put("/api/v2/settings", json={
            "llm_api_key": "sk-test-1234567890",
            "llm_provider": "moonshot",
        })

        assert recorder.fields_for("key_set") == [{"provider": "moonshot"}]

    def test_put_settings_without_a_key_emits_nothing(self, client, recorder):
        resp = client.put("/api/v2/settings", json={"llm_model": "kimi-k2"})
        assert resp.status_code == 200
        assert recorder.fields_for("key_set") == []


@pytest.mark.skipif(os.name == "nt",
                    reason="stub CLI relies on a shebang; not a thing on Windows")
class TestSubAgentApiKeyTelemetry:
    """``POST /settings/sub-agents/{slug}/api-key`` emitted nothing at all."""

    class _StubEngine:
        """Real registry (so the codex manifest resolves), stub binary."""

        def __init__(self, registry, binary: str):
            self._registry = registry
            self._binary = binary
            self.invalidations = 0

        def resolve_binary(self, manifest) -> str:
            return self._binary

        def invalidate_cache(self) -> None:
            self.invalidations += 1

    @staticmethod
    def _stub_cli(tmp_path, exit_code: int) -> str:
        script = tmp_path / "stub-codex"
        script.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.stdin.read()\n"          # drain the piped key, never store it
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
        return str(script)

    def _wire(self, client, monkeypatch, tmp_path, exit_code: int):
        registry = getattr(settings_routes._setup_engine, "_registry", None)
        assert registry is not None, "app did not wire a setup engine"
        monkeypatch.setattr(
            settings_routes, "_setup_engine",
            self._StubEngine(registry, self._stub_cli(tmp_path, exit_code)))

    def test_accepted_key_emits_key_set_with_the_slug(
            self, client, recorder, monkeypatch, tmp_path):
        self._wire(client, monkeypatch, tmp_path, exit_code=0)

        resp = client.post("/api/v2/settings/sub-agents/codex/api-key",
                           json={"api_key": "sk-test-1234567890"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["return_code"] == 0
        assert recorder.fields_for("key_set") == [{"provider": "codex"}]
        assert "key_set" in recorder.latched

    def test_rejected_key_does_not_emit(self, client, recorder, monkeypatch,
                                        tmp_path):
        """The CLI refused the key, so nothing was set — counting it would
        inflate key_set with failures."""
        self._wire(client, monkeypatch, tmp_path, exit_code=1)

        resp = client.post("/api/v2/settings/sub-agents/codex/api-key",
                           json={"api_key": "not-a-real-key"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["return_code"] == 1
        assert recorder.fields_for("key_set") == []

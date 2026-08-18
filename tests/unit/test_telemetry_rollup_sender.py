"""Daily rollup + sender (spec 046 §5, §11)."""

import json
from pathlib import Path

from agent_os.telemetry.identity import InstallIdentity
from agent_os.telemetry.rollup import build_ping
from agent_os.telemetry.sender import TelemetrySender
from agent_os.telemetry.spool import Spool

DAY = "2026-08-08"


def seed_spool(tmp_path: Path, rows: list[dict]) -> Spool:
    path = tmp_path / "telemetry" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Spool(tmp_path)


def ev(event: str, **fields) -> dict:
    return {"event": event, "ts": f"{DAY}T10:00:00+00:00", **fields}


class TestRollup:
    def test_aggregates_counters_and_disjoint_tokens(self, tmp_path):
        spool = seed_spool(tmp_path, [
            ev("app_start"),
            ev("app_start"),
            ev("project_created"),
            ev("session_created"),
            ev("turn_completed", provider="deepseek",
               uncached_input=100, cache_read=900, cache_write=50, output=80),
            ev("turn_completed", provider="deepseek",
               uncached_input=10, cache_read=0, cache_write=0, output=20),
            ev("turn_completed", provider="minimax",
               uncached_input=5, cache_read=5, cache_write=0, output=5),
            ev("llm_error", error_code="provider_unreachable"),
            ev("llm_error", error_code="provider_unreachable"),
            ev("llm_error", error_code="invalid_api_key"),
            # Different day must be excluded:
            {"event": "app_start", "ts": "2026-08-07T10:00:00+00:00"},
        ])
        identity = InstallIdentity(tmp_path)
        identity.latch_milestone("key_set")
        identity.latch_milestone("first_turn")

        ping = build_ping(identity, spool, DAY, version="0.8.4", os_name="darwin")

        assert ping["schema"] == 1
        assert ping["install_id"] == identity.install_id
        assert ping["account_id"] is None
        assert ping["version"] == "0.8.4"
        assert ping["os"] == "darwin"
        assert ping["date"] == DAY
        assert ping["milestones"] == {
            "key_set": True, "first_project": False,
            "first_session": False, "first_turn": True,
        }
        c = ping["counters"]
        assert c["app_starts"] == 2
        assert c["projects_created"] == 1
        assert c["sessions"] == 1
        assert c["turns"] == 3
        # in = uncached_input + cache_read + cache_write (disjoint ledger fields)
        assert c["tokens_by_provider"]["deepseek"] == {"in": 1060, "out": 100}
        assert c["tokens_by_provider"]["minimax"] == {"in": 10, "out": 5}
        assert c["errors"] == {"provider_unreachable": 2, "invalid_api_key": 1}

    def test_empty_day_yields_zero_ping(self, tmp_path):
        ping = build_ping(
            InstallIdentity(tmp_path), Spool(tmp_path), DAY,
            version="0.8.4", os_name="darwin",
        )
        assert ping["counters"]["turns"] == 0
        assert ping["counters"]["tokens_by_provider"] == {}
        assert ping["counters"]["errors_by_provider"] == {}
        assert ping["counters"]["login_attempted"] == 0
        assert ping["counters"]["login_failed"] == 0

    def test_errors_split_by_provider_alongside_by_code(self, tmp_path):
        """Spec 063 §4 + decision 2: the new map is ADDITIVE — `errors` is the
        only error series with history and keeps its exact shape."""
        spool = seed_spool(tmp_path, [
            ev("llm_error", error_code="invalid_api_key", provider="deepseek"),
            ev("llm_error", error_code="provider_unreachable", provider="deepseek"),
            ev("llm_error", error_code="invalid_api_key", provider="minimax"),
        ])
        c = build_ping(
            InstallIdentity(tmp_path), spool, DAY, version="0.9.1", os_name="windows",
        )["counters"]
        assert c["errors"] == {"invalid_api_key": 2, "provider_unreachable": 1}
        assert c["errors_by_provider"] == {"deepseek": 2, "minimax": 1}

    def test_unresolved_provider_falls_back_to_unknown(self, tmp_path):
        """Construction failures fail before the provider is resolved; the
        same `unknown` bucket tokens_by_provider already uses (§8)."""
        spool = seed_spool(tmp_path, [
            ev("llm_error", error_code="missing_api_key"),          # field absent
            ev("llm_error", error_code="missing_api_key", provider=None),
            ev("llm_error", error_code="invalid_api_key", provider="deepseek"),
        ])
        c = build_ping(
            InstallIdentity(tmp_path), spool, DAY, version="0.9.1", os_name="darwin",
        )["counters"]
        assert c["errors_by_provider"] == {"unknown": 2, "deepseek": 1}
        assert sum(c["errors_by_provider"].values()) == sum(c["errors"].values())

    def test_login_funnel_counters_aggregate(self, tmp_path):
        """Spec 063 §12 decision 5: sub-agent setup — the product's
        differentiator — was invisible between key_set and first_turn. The
        agent slug stays in the local spool; only totals are transmitted."""
        spool = seed_spool(tmp_path, [
            ev("login_attempted", agent="claude-code"),
            ev("login_attempted", agent="codex"),
            ev("login_attempted", agent="codex"),
            ev("login_failed", agent="codex"),
            # Different day must be excluded:
            {"event": "login_attempted", "ts": "2026-08-07T10:00:00+00:00"},
        ])
        c = build_ping(
            InstallIdentity(tmp_path), spool, DAY, version="0.9.1", os_name="windows",
        )["counters"]
        assert c["login_attempted"] == 3
        assert c["login_failed"] == 1
        assert isinstance(c["login_attempted"], int)


class FakePost:
    def __init__(self, status=200, exc: Exception | None = None):
        self.status = status
        self.exc = exc
        self.calls: list[tuple[str, dict, dict]] = []

    async def __call__(self, url, payload, headers):
        self.calls.append((url, payload, headers))
        if self.exc:
            raise self.exc
        return self.status


def make_sender(tmp_path, post, enabled=True) -> TelemetrySender:
    return TelemetrySender(
        tmp_path,
        InstallIdentity(tmp_path),
        Spool(tmp_path),
        is_enabled=lambda: enabled,
        endpoint="https://example.invalid/ingest",
        post=post,
    )


class TestSender:
    async def test_cycle_snapshots_today_and_sends(self, tmp_path):
        post = FakePost()
        sender = make_sender(tmp_path, post)
        await sender.run_cycle()

        assert len(post.calls) == 1
        url, payload, headers = post.calls[0]
        assert payload["schema"] == 1
        assert headers["X-Orbital-Telemetry-Token"]
        # Today's snapshot survives for the next cycle (upsert semantics).
        pending = list((tmp_path / "telemetry" / "pending").glob("*.json"))
        assert len(pending) == 1
        # Viewer surfaces: last-sent recorded verbatim.
        assert sender.last_sent_payload() == payload

    async def test_past_day_pending_deleted_on_success_today_kept(self, tmp_path):
        post = FakePost()
        sender = make_sender(tmp_path, post)
        pending = tmp_path / "telemetry" / "pending"
        pending.mkdir(parents=True)
        stale = {"schema": 1, "date": "2020-01-01", "install_id": "inst_x"}
        (pending / "2020-01-01.json").write_text(json.dumps(stale))

        await sender.run_cycle()

        names = sorted(p.name for p in pending.glob("*.json"))
        assert "2020-01-01.json" not in names  # flushed + deleted
        assert len(names) == 1  # today's snapshot remains
        assert len(post.calls) == 2

    async def test_offline_keeps_all_pending(self, tmp_path):
        post = FakePost(exc=OSError("no network"))
        sender = make_sender(tmp_path, post)
        await sender.run_cycle()
        pending = list((tmp_path / "telemetry" / "pending").glob("*.json"))
        assert len(pending) == 1  # today's snapshot queued for next cycle
        assert sender.last_sent_payload() is None

    async def test_server_error_keeps_pending(self, tmp_path):
        post = FakePost(status=500)
        sender = make_sender(tmp_path, post)
        await sender.run_cycle()
        assert len(list((tmp_path / "telemetry" / "pending").glob("*.json"))) == 1
        assert sender.last_sent_payload() is None

    async def test_toggle_off_no_send_no_snapshot(self, tmp_path):
        post = FakePost()
        sender = make_sender(tmp_path, post, enabled=False)
        await sender.run_cycle()
        assert post.calls == []
        assert not (tmp_path / "telemetry" / "pending").exists()

    async def test_next_pending_payload_matches_schema(self, tmp_path):
        sender = make_sender(tmp_path, FakePost())
        payload = sender.next_pending_payload()
        assert set(payload) == {
            "schema", "install_id", "account_id", "version", "os",
            "date", "first_seen", "milestones", "counters",
        }

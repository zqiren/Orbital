"""Telemetry install identity + event spool (spec 046 §3, §4)."""

import json
import os
import stat
from pathlib import Path

from agent_os.telemetry.identity import InstallIdentity
from agent_os.telemetry.spool import Spool


class TestInstallIdentity:
    def test_creates_once_and_is_stable(self, tmp_path):
        ident = InstallIdentity(tmp_path)
        first = ident.install_id
        assert first.startswith("inst_") and len(first) == 5 + 12

        again = InstallIdentity(tmp_path)
        assert again.install_id == first

    def test_first_seen_is_iso_date(self, tmp_path):
        ident = InstallIdentity(tmp_path)
        assert len(ident.first_seen) == 10  # YYYY-MM-DD
        json.loads((tmp_path / "telemetry" / "install.json").read_text())

    def test_reset_mints_new_identity(self, tmp_path):
        ident = InstallIdentity(tmp_path)
        old = ident.install_id
        new = ident.reset()
        assert new != old
        assert InstallIdentity(tmp_path).install_id == new

    def test_milestones_latch_and_persist(self, tmp_path):
        ident = InstallIdentity(tmp_path)
        assert ident.milestones == {}
        ident.latch_milestone("key_set")
        ident.latch_milestone("key_set")  # idempotent
        assert InstallIdentity(tmp_path).milestones == {"key_set": True}

    def test_reset_clears_milestones(self, tmp_path):
        ident = InstallIdentity(tmp_path)
        ident.latch_milestone("first_turn")
        ident.reset()
        assert InstallIdentity(tmp_path).milestones == {}

    def test_never_raises_on_unwritable_dir(self, tmp_path):
        target = tmp_path / "ro"
        target.mkdir()
        os.chmod(target, stat.S_IRUSR | stat.S_IXUSR)
        try:
            ident = InstallIdentity(target)
            # Still yields a usable in-memory identity.
            assert ident.install_id.startswith("inst_")
            ident.latch_milestone("key_set")  # must not raise
        finally:
            os.chmod(target, stat.S_IRWXU)


class TestSpool:
    def test_append_writes_jsonl_with_ts(self, tmp_path):
        spool = Spool(tmp_path)
        spool.append("key_set", {"provider": "deepseek"})
        rows = [
            json.loads(line)
            for line in (tmp_path / "telemetry" / "events.jsonl").read_text().splitlines()
        ]
        assert rows[0]["event"] == "key_set"
        assert rows[0]["provider"] == "deepseek"
        assert rows[0]["ts"].endswith("+00:00") or rows[0]["ts"].endswith("Z")

    def test_read_day_filters_by_utc_date(self, tmp_path):
        spool = Spool(tmp_path)
        path = tmp_path / "telemetry" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"event": "app_start", "ts": "2026-08-07T23:59:00+00:00"}) + "\n"
            + json.dumps({"event": "app_start", "ts": "2026-08-08T00:01:00+00:00"}) + "\n"
            + "corrupt line\n"
            + json.dumps({"event": "turn_completed", "ts": "2026-08-08T10:00:00+00:00"}) + "\n"
        )
        events = list(spool.read_day("2026-08-08"))
        assert [e["event"] for e in events] == ["app_start", "turn_completed"]

    def test_rotation_caps_size(self, tmp_path):
        spool = Spool(tmp_path, max_bytes=500)
        for _ in range(50):
            spool.append("turn_completed", {"provider": "deepseek", "in": 1, "out": 2})
        live = tmp_path / "telemetry" / "events.jsonl"
        rotated = tmp_path / "telemetry" / "events.jsonl.1"
        assert live.stat().st_size <= 500 + 200  # one row of slack
        assert rotated.exists()

    def test_never_raises_on_unwritable_dir(self, tmp_path):
        target = tmp_path / "ro"
        target.mkdir()
        os.chmod(target, stat.S_IRUSR | stat.S_IXUSR)
        try:
            spool = Spool(target)
            spool.append("app_start", {})  # must not raise
        finally:
            os.chmod(target, stat.S_IRWXU)

    def test_read_day_tolerates_missing_file(self, tmp_path):
        assert list(Spool(tmp_path).read_day("2026-08-08")) == []

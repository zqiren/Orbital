# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 154 §5.1 — the daemon-side checked state (``checked.json``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_os.daemon_v2.checked_store import CheckedStore, session_key

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
RECENT = "2026-11-03T10:00:00+00:00"
LATER = "2026-11-04T09:00:00+00:00"
OLD = "2026-10-01T10:00:00+00:00"  # outside the 14-day window


def _store(tmp_path):
    return CheckedStore.in_dir(str(tmp_path), now_fn=lambda: NOW)


def test_session_key_prefers_uuid():
    assert session_key({"session_id": "sess_1", "session_uuid": "u1"}) == "u1"
    assert session_key({"session_id": "u2"}) == "u2"
    assert session_key({}) is None


def test_seed_once_marks_existing_replies_checked(tmp_path):
    st = _store(tmp_path)
    rows = [
        {"session_id": "a", "last_reply_at": RECENT},
        {"session_id": "b", "last_reply_at": None},
        {"session_id": "c", "last_reply_at": OLD},  # outside the window: not kept
    ]
    assert st.seed_if_absent("p", rows) == {"a": RECENT}
    # Second seed with a newer reply never overwrites: seeding runs once.
    assert st.seed_if_absent("p", [{"session_id": "a", "last_reply_at": LATER}]) == {"a": RECENT}
    on_disk = json.loads((tmp_path / "checked.json").read_text())
    assert on_disk == {"p": {"a": RECENT}}


def test_empty_project_seeds_to_present_empty_map(tmp_path):
    st = _store(tmp_path)
    assert st.project("p") is None
    st.seed_if_absent("p", [])
    assert st.project("p") == {}
    # A reply landing later is unchecked (no re-seed).
    rows = st.annotate("p", [{"session_id": "a", "last_reply_at": LATER}])
    assert rows[0]["checked_reply_at"] is None


def test_mark_keeps_the_later_time(tmp_path):
    st = _store(tmp_path)
    st.seed_if_absent("p", [])
    assert st.mark("p", "a", LATER) is True
    assert st.mark("p", "a", RECENT) is False  # an older check never moves it back
    assert st.project("p") == {"a": LATER}
    assert st.mark("p", "a", LATER) is False


def test_mark_drops_entries_older_than_the_window(tmp_path):
    (tmp_path / "checked.json").write_text(json.dumps({"p": {"old": OLD, "a": RECENT}}))
    st = _store(tmp_path)
    st.mark("p", "b", LATER)
    assert st.project("p") == {"a": RECENT, "b": LATER}


def test_mark_rejects_a_non_time(tmp_path):
    st = _store(tmp_path)
    with pytest.raises(ValueError):
        st.mark("p", "a", "yesterday")


def test_annotate_adds_checked_reply_at(tmp_path):
    st = _store(tmp_path)
    rows = [{"session_id": "a", "session_uuid": "ua", "last_reply_at": RECENT}]
    st.annotate("p", rows)
    assert rows[0]["checked_reply_at"] == RECENT
    rows2 = [{"session_id": "a", "session_uuid": "ua", "last_reply_at": LATER},
             {"session_id": "n", "last_reply_at": LATER}]
    st.annotate("p", rows2)
    assert rows2[0]["checked_reply_at"] == RECENT  # the newer reply is unchecked
    assert rows2[1]["checked_reply_at"] is None


def test_corrupt_file_reads_as_absent_and_is_replaced(tmp_path):
    (tmp_path / "checked.json").write_text("{not json")
    st = _store(tmp_path)
    assert st.project("p") is None
    st.seed_if_absent("p", [{"session_id": "a", "last_reply_at": RECENT}])
    assert json.loads((tmp_path / "checked.json").read_text()) == {"p": {"a": RECENT}}


def test_reload_from_disk(tmp_path):
    _store(tmp_path).seed_if_absent("p", [{"session_id": "a", "last_reply_at": RECENT}])
    assert _store(tmp_path).project("p") == {"a": RECENT}


@pytest.mark.parametrize("live", [
    {"status": "running"}, {"status": "waiting"}, {"status": "queued"},
    {"status": "idle", "worker_running": True},
])
def test_seed_skips_a_session_that_is_live_at_seed_time(tmp_path, live):
    """Smoke 2026-11-04: the first read of a project landed while a turn was
    still in its end-of-run record step — the reply already written, the
    session still running — and the seed swallowed that fresh reply, so the
    finished task never reached the Workbench. A live session's reply is the
    one being produced now, not history: it stays unchecked."""
    st = _store(tmp_path)
    rows = [{"session_id": "hist", "status": "idle", "last_reply_at": RECENT},
            {"session_id": "live", "last_reply_at": LATER, **live}]
    assert st.seed_if_absent("p", rows) == {"hist": RECENT}
    annotated = st.annotate("p", [dict(r, status="idle", worker_running=False) for r in rows])
    assert [r["checked_reply_at"] for r in annotated] == [RECENT, None]

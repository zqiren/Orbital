# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/workbench routes + computed cards (Task 5).

Mounts only the workbench router over an in-process ASGI transport with a
fake project_store / agent_manager and a real CalendarHub (no sources needed —
Task 5 only calls ``hub.refresh()``). Workspaces are tmp dirs seeded with a
PROJECT_STATE.md written in the ``[user]`` grammar.
"""

import json
import os
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agent import user_flags, workbench_cards
from agent_os.api.routes import workbench as workbench_routes
from agent_os.calendar_hub.hub import CalendarHub
from agent_os.calendar_hub.linkage import Linkage

# Frozen "now" for deterministic age/overdue math (UTC noon on 2026-07-24).
NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
TODAY = "2026-07-24"

FORMAT_LINE = (
    "<!--format PROJECT_STATE is what is true NOW. legacy header placeholder-->"
)

# A seeded state file exercising the three shapes: a flagged+dated entry, a
# flagged (no due) entry, an unflagged dated fact, and plain prose.
SEEDED_STATE = "\n".join([
    FORMAT_LINE,
    "## Focus",
    "- [user due:2026-07-28] Send 宝玉 + Simon DM drafts — only you can send from your accounts.",
    '  <!--mem id:x7f3a2 from:orbital-marketing_7c045c40 evidence:"EN 这边宝玉和 Simon 的 draft 写好就准备发" confidence:unconfirmed created:2026-07-19 touched:2026-07-23-->',
    "- [user] Approve the Q3 budget before Friday.",
    '  <!--mem id:a1b2c3 from:sess_x evidence:"我们需要你批准预算" created:2026-07-20-->',
    "- [due:2026-07-20] Ship the marketing site.",
    "  <!--mem id:d4e5f6 created:2026-07-15-->",
    "- Just a plain architectural fact, agent reference only.",
    "",
])


class FakeProjectStore:
    def __init__(self, projects):
        self._projects = projects  # {pid: dict}

    def list_projects(self):
        return list(self._projects.values())

    def get_project(self, pid):
        return self._projects.get(pid)

    def update_project(self, pid, updates):
        self._projects[pid].update(updates)


class FakeAgentManager:
    def __init__(self):
        self.injected = []      # (project_id, content, session_id)
        self._counter = 0
        self.sessions = {}      # pid -> list[dict]

    async def new_session(self, project_id, session_id=None):
        self._counter += 1
        sid = f"minted_{self._counter}"
        return {"status": "ok", "session_id": sid, "session_uuid": sid}

    async def inject_message(self, project_id, content, *, session_id=None, nonce=None):
        self.injected.append((project_id, content, session_id))
        return "started"

    def list_sessions(self, project_id):
        return self.sessions.get(project_id, [])


def _seed_project(tmp_path, pid="proj_a", *, state=SEEDED_STATE, extra=None):
    ws = tmp_path / pid
    (ws / "orbital").mkdir(parents=True, exist_ok=True)
    if state is not None:
        (ws / "orbital" / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
    # Pin tz to UTC so age/overdue assertions don't depend on the test host's
    # local zone (project_timezone precedence is tested separately).
    project = {"project_id": pid, "name": pid, "workspace": str(ws),
               "timezone": "UTC"}
    if extra:
        project.update(extra)
    return project


def _make_client(tmp_path, projects, *, agent_manager=None, hub=None, now=None):
    store = FakeProjectStore({p["project_id"]: p for p in projects})
    am = agent_manager or FakeAgentManager()
    h = hub or CalendarHub(sources=[], linkage=Linkage(str(tmp_path / "_linkage")))
    frozen = now or NOW
    app = FastAPI()
    workbench_routes.configure(store, am, h, now_fn=lambda: frozen)
    app.include_router(workbench_routes.router)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return client, store, am, h


def _state_path(project):
    return os.path.join(project["workspace"], "orbital", "PROJECT_STATE.md")


# --------------------------------------------------------------------------
# GET /api/v2/workbench
# --------------------------------------------------------------------------

async def test_get_parses_seeded_file(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200, r.text
        body = r.json()

    ids = {e["id"] for e in body["entries"]}
    assert ids == {"x7f3a2", "a1b2c3"}  # flagged only; plain prose excluded

    x = next(e for e in body["entries"] if e["id"] == "x7f3a2")
    assert x["text"].startswith("Send 宝玉 + Simon DM drafts")
    assert x["due"] == "2026-07-28"
    assert x["confidence"] == "unconfirmed"
    assert x["from_session"] == "orbital-marketing_7c045c40"
    assert x["evidence"].startswith("EN 这边宝玉")
    assert x["age_days"] == 5           # 07-19 -> 07-24
    assert x["overdue"] is False
    assert x["project_id"] == "proj_a"

    # The unflagged dated fact, being past-due, surfaces as an overdue computed
    # card (never as a flagged entry).
    overdue = [c for c in body["computed"] if c["type"] == "overdue"]
    assert [c["key"] for c in overdue] == ["d4e5f6"]
    assert overdue[0]["since"] == "2026-07-20"
    assert overdue[0]["project_id"] == "proj_a"


async def test_get_sort_overdue_first_then_oldest(tmp_path):
    state = "\n".join([
        FORMAT_LINE,
        "- [user] Newer unflagged-due question.",
        "  <!--mem id:newr created:2026-07-22-->",
        "- [user due:2026-07-01] Overdue flagged obligation.",
        "  <!--mem id:ovrd created:2026-07-20-->",
        "- [user] Older waiting question.",
        "  <!--mem id:oldr created:2026-07-10-->",
        "",
    ])
    project = _seed_project(tmp_path, state=state)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    order = [e["id"] for e in body["entries"]]
    # overdue first; then remaining by oldest created.
    assert order == ["ovrd", "oldr", "newr"]


async def test_global_get_survives_invalid_utf8_project(tmp_path):
    """A single project with invalid UTF-8 in PROJECT_STATE.md must not 500 the
    whole global GET — the healthy project's entries still come back."""
    healthy = _seed_project(tmp_path, pid="proj_ok")
    bad = _seed_project(tmp_path, pid="proj_bad", state="placeholder")
    with open(_state_path(bad), "wb") as f:
        f.write(b"\xff\xfe not valid utf-8 \x80\x81\n- [user] garbled\n")
    client, *_ = _make_client(tmp_path, [healthy, bad])
    async with client:
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200, r.text
        body = r.json()
    assert {"x7f3a2", "a1b2c3"} <= {e["id"] for e in body["entries"]}


async def test_privacy_toggle_skips_project_in_global_view(tmp_path):
    a = _seed_project(tmp_path, pid="proj_a")
    b = _seed_project(tmp_path, pid="proj_b",
                      extra={"workbench_exclude_global": True})
    client, *_ = _make_client(tmp_path, [a, b])
    async with client:
        glob = (await client.get("/api/v2/workbench")).json()
        assert {e["project_id"] for e in glob["entries"]} == {"proj_a"}
        # ...but the per-project lens still shows the excluded project.
        lensed = (await client.get("/api/v2/workbench",
                                   params={"project_id": "proj_b"})).json()
        assert {e["project_id"] for e in lensed["entries"]} == {"proj_b"}


# --------------------------------------------------------------------------
# Exits
# --------------------------------------------------------------------------

async def test_fulfilled_exit_rewrites_file_exactly(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ , hub = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/x7f3a2/exit",
            json={"kind": "fulfilled", "reason": "sent them this morning"},
        )
        assert r.status_code == 200, r.text
        # The fulfilled item leaves the Workbench (a done item is no longer
        # surfaced). Asserted at the GET level so it holds regardless of the
        # parser's internal treatment of the tag-less resolved fact.
        body = (await client.get("/api/v2/workbench")).json()
        surfaced = {e["id"] for e in body["entries"]}
        assert "x7f3a2" not in surfaced
        assert "a1b2c3" in surfaced          # the other flagged entry stays

    content = open(_state_path(project), encoding="utf-8").read()
    # The resolved stamp is written into the (now plain-fact) bullet's comment.
    assert "resolved:2026-07-24" in content
    # The sentence survives as a plain fact.
    assert "Send 宝玉 + Simon DM drafts" in content
    # The other flagged entry is byte-for-byte unchanged; the dated fact intact.
    assert "- [user] Approve the Q3 budget before Friday." in content
    assert "- [due:2026-07-20] Ship the marketing site." in content


async def test_fulfilled_exit_keeps_comment_adjacent_and_preserves_id(tmp_path):
    """Design guard (Task 3 review): the fulfilled bullet drops its ``[user]``
    tag but keeps its mem-comment PHYSICALLY ADJACENT — the bullet line is
    immediately followed by its comment line — so the id + resolved stamp stay
    re-associable and the comment never orphans into plain prose."""
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/x7f3a2/exit",
            json={"kind": "fulfilled", "reason": "done"},
        )
        assert r.status_code == 200, r.text

    lines = open(_state_path(project), encoding="utf-8").read().split("\n")
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("- Send 宝玉"))
    bullet, comment = lines[idx], lines[idx + 1]
    assert "[user" not in bullet                     # bracket tag dropped
    assert comment.lstrip().startswith("<!--mem")    # comment immediately follows
    assert "id:x7f3a2" in comment                    # id preserved
    assert "resolved:2026-07-24" in comment          # resolved stamp present


async def test_fulfilled_exit_leaves_parseable_resolved_trace(tmp_path):
    """Post-fulfilled, the retired entry re-parses via the (amended)
    parse_entries as a present, UNFLAGGED entry with its resolved stamp and id
    intact — the anti-resurrection trace the chokepoint re-associates on the
    next agent rewrite (spec §5.3)."""
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/x7f3a2/exit",
            json={"kind": "fulfilled", "reason": "sent this morning"},
        )
        assert r.status_code == 200, r.text

    content = open(_state_path(project), encoding="utf-8").read()
    entry = next((e for e in user_flags.parse_entries(content) if e.id == "x7f3a2"), None)
    assert entry is not None            # present
    assert entry.flagged is False       # unflagged (tag dropped)
    assert entry.resolved == "2026-07-24"  # resolved stamp set
    assert entry.id == "x7f3a2"         # id preserved


async def test_resolved_dated_fact_never_produces_overdue_card(tmp_path):
    """(c) guard: a dated fact carrying a resolved stamp must not resurrect as
    an overdue computed card off its now-stale due — while a normal (unresolved)
    past-due dated fact still does."""
    state = "\n".join([
        FORMAT_LINE,
        # resolved dated fact with a stale past due -> must be silent
        "- [due:2026-07-01] Already-finished shipment.",
        "  <!--mem id:done1 created:2026-06-01 resolved:2026-07-10-->",
        # unresolved past-due dated fact -> positive control (still overdue)
        "- [due:2026-07-02] Live overdue item.",
        "  <!--mem id:live1 created:2026-06-01-->",
        "",
    ])
    project = _seed_project(tmp_path, state=state)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    overdue_keys = {c["key"] for c in body["computed"] if c["type"] == "overdue"}
    assert "done1" not in overdue_keys   # resolved -> skipped
    assert "live1" in overdue_keys       # unresolved -> still fires
    # And a retired/dated fact is never a flagged entry row either.
    assert {e["id"] for e in body["entries"]} == set()


async def test_fulfilled_exit_survives_concurrent_write(tmp_path, monkeypatch):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])

    orig = workbench_routes._load_state
    calls = {"n": 0}

    def racing_load(path):
        mtime, content = orig(path)
        if calls["n"] == 0:
            calls["n"] += 1
            # A concurrent writer bumps the file's mtime AFTER we captured the
            # baseline but before the guarded write -> forces the retry path.
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n<!-- concurrent touch -->\n")
        return mtime, content

    monkeypatch.setattr(workbench_routes, "_load_state", racing_load)

    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/x7f3a2/exit",
            json={"kind": "fulfilled", "reason": "done"},
        )
        assert r.status_code == 200, r.text
        # GET-level exclusion: the retired item is no longer surfaced.
        body = (await client.get("/api/v2/workbench")).json()
        assert "x7f3a2" not in {e["id"] for e in body["entries"]}
    assert calls["n"] == 1  # exactly one simulated conflict, one retry
    content = open(_state_path(project), encoding="utf-8").read()
    assert "resolved:2026-07-24" in content
    # The retired entry stays parse-visible as the anti-resurrection trace:
    # present, unflagged, resolved stamped (per the parse_entries amendment).
    retired = next(e for e in user_flags.parse_entries(content) if e.id == "x7f3a2")
    assert retired.flagged is False
    assert retired.resolved == "2026-07-24"


async def test_irrelevant_exit_removes_entry_and_writes_retraction(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/a1b2c3/exit",
            json={"kind": "irrelevant", "reason": "changed my mind"},
        )
        assert r.status_code == 200, r.text

    content = open(_state_path(project), encoding="utf-8").read()
    assert "a1b2c3" not in {e.id for e in user_flags.parse_entries(content)}
    assert "Approve the Q3 budget" not in content

    retractions = open(
        os.path.join(project["workspace"], "orbital", "retractions.md"),
        encoding="utf-8",
    ).read()
    assert "[a1b2c3]" in retractions
    assert "Approve the Q3 budget before Friday." in retractions
    assert "changed my mind" in retractions


async def test_exit_unknown_id_is_404(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/nope99/exit",
            json={"kind": "fulfilled", "reason": ""},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------
# Dismiss
# --------------------------------------------------------------------------

async def test_dismiss_persists_and_filters(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        # d4e5f6 is an overdue computed card in the seeded file.
        before = (await client.get("/api/v2/workbench")).json()
        assert any(c["key"] == "d4e5f6" for c in before["computed"])

        r = await client.post(
            "/api/v2/workbench/proj_a/computed/overdue/d4e5f6/dismiss")
        assert r.status_code == 200, r.text

        after = (await client.get("/api/v2/workbench")).json()
        assert not any(c["key"] == "d4e5f6" for c in after["computed"])

    dismissals = json.load(open(
        os.path.join(project["workspace"], "orbital", "workbench_dismissals.json"),
        encoding="utf-8",
    ))
    assert any(d["type"] == "overdue" and d["key"] == "d4e5f6" for d in dismissals)


# --------------------------------------------------------------------------
# Open / Migrate (spawn seam)
# --------------------------------------------------------------------------

async def test_open_spawns_seeded_session(tmp_path):
    project = _seed_project(tmp_path)
    am = FakeAgentManager()
    client, *_ = _make_client(tmp_path, [project], agent_manager=am)
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/entries/x7f3a2/open")
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
    assert sid == "minted_1"
    assert len(am.injected) == 1
    pid, content, session_id = am.injected[0]
    assert pid == "proj_a"
    assert session_id == "minted_1"
    assert 'Send 宝玉 + Simon DM drafts' in content
    assert "Let's handle this." in content


async def test_migrate_refreshes_header_and_spawns(tmp_path):
    from agent_os.agent.memory_entries import FORMAT_HEADERS
    project = _seed_project(tmp_path)
    am = FakeAgentManager()
    client, *_ = _make_client(tmp_path, [project], agent_manager=am)
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/migrate")
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == "minted_1"

    content = open(_state_path(project), encoding="utf-8").read()
    # Legacy placeholder header replaced with the canonical current template.
    assert FORMAT_HEADERS["state"] in content
    assert "legacy header placeholder" not in content
    # Body content preserved.
    assert "Approve the Q3 budget before Friday." in content
    # A migration session was spawned with the [user] grammar instruction.
    assert len(am.injected) == 1
    assert "flag entries that need the user" in am.injected[0][1]


# --------------------------------------------------------------------------
# Computed detectors (unit-level)
# --------------------------------------------------------------------------

def test_overdue_boundary_uses_project_tz_not_utc():
    # UTC 2026-07-23T20:00 is 2026-07-24T04:00 in Shanghai (UTC+8).
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    # A due date of 2026-07-23: still "today" in UTC (not overdue), but already
    # yesterday in Shanghai (overdue).
    assert workbench_cards.is_overdue("2026-07-23", "UTC", now=now) is False
    assert workbench_cards.is_overdue("2026-07-23", "Asia/Shanghai", now=now) is True


def test_days_late_uses_project_tz_not_utc():
    # UTC 2026-07-23T20:00 == 2026-07-24T04:00 in Shanghai (UTC+8).
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    d = "2026-07-23"
    assert workbench_cards.days_late(d, "UTC", now=now) is None          # today in UTC
    assert workbench_cards.days_late(d, "Asia/Shanghai", now=now) == 1   # yesterday there
    # Non-overdue / unknown due -> None.
    assert workbench_cards.days_late("2026-08-01", "Asia/Shanghai", now=now) is None
    assert workbench_cards.days_late(None, "UTC", now=now) is None


async def test_entry_row_days_late_computed_in_project_tz(tmp_path):
    """days_late on the entry row is project-tz, not browser/UTC: an item due
    2026-07-23 with 'now' at UTC 20:00 (already past midnight in Shanghai) is
    1 day late in an Asia/Shanghai project."""
    state = "\n".join([
        FORMAT_LINE,
        "- [user due:2026-07-23] Confirm the venue booking.",
        "  <!--mem id:late1 created:2026-07-20-->",
        "",
    ])
    project = _seed_project(tmp_path, state=state,
                            extra={"timezone": "Asia/Shanghai"})
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    client, *_ = _make_client(tmp_path, [project], now=now)
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    row = next(e for e in body["entries"] if e["id"] == "late1")
    assert row["overdue"] is True
    assert row["days_late"] == 1     # project tz (Shanghai), not UTC (which is 0/None)


async def test_days_late_null_when_not_overdue(tmp_path):
    """A flagged entry that is not past due carries days_late = null."""
    project = _seed_project(tmp_path)  # x7f3a2 due 2026-07-28, now 2026-07-24
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    row = next(e for e in body["entries"] if e["id"] == "x7f3a2")
    assert row["overdue"] is False
    assert row["days_late"] is None


def test_project_timezone_precedence():
    triggers = [{"type": "schedule", "schedule": {"cron": "0 9 * * *",
                                                  "timezone": "Europe/Paris"}}]
    # explicit project setting wins
    assert workbench_cards.project_timezone(
        {"timezone": "Asia/Tokyo"}, triggers) == "Asia/Tokyo"
    # else first schedule trigger's tz
    assert workbench_cards.project_timezone({}, triggers) == "Europe/Paris"
    # else a resolvable daemon-local zone name
    assert workbench_cards.project_timezone({}, []) not in (None, "")


def test_broken_automation_detector_on_synthetic_trigger():
    now = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    healthy = {
        "id": "t1", "type": "schedule", "enabled": True,
        "schedule": {"cron": "0 9 * * *", "timezone": "UTC"},
        "last_triggered": "2026-07-24T09:00:00+00:00",
    }
    broken = {
        "id": "t2", "type": "schedule", "enabled": True,
        "schedule": {"cron": "0 9 * * *", "timezone": "UTC"},
        "last_triggered": "2026-07-20T09:00:00+00:00",
    }
    disabled = {**broken, "id": "t3", "enabled": False}
    assert workbench_cards.is_broken_automation(healthy, now) is False
    assert workbench_cards.is_broken_automation(broken, now) is True
    assert workbench_cards.is_broken_automation(disabled, now) is False


async def test_broken_automation_suppresses_matching_flagged_entry(tmp_path):
    state = "\n".join([
        FORMAT_LINE,
        "- [user] The nightly-digest automation looks broken, please check.",
        "  <!--mem id:supp created:2026-07-20-->",
        "- [user] Unrelated question about pricing.",
        "  <!--mem id:keep created:2026-07-20-->",
        "",
    ])
    trigger = {
        "id": "trg1", "name": "nightly-digest", "type": "schedule",
        "enabled": True,
        "schedule": {"cron": "0 9 * * *", "timezone": "UTC"},
        "last_triggered": "2026-07-19T09:00:00+00:00",
    }
    project = _seed_project(tmp_path, state=state, extra={"triggers": [trigger]})
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    ids = {e["id"] for e in body["entries"]}
    # The entry naming the broken trigger is suppressed; the computed card wins.
    assert "supp" not in ids
    assert "keep" in ids
    assert any(c["type"] == "broken_automation" and c["key"] == "trg1"
               for c in body["computed"])


def test_paused_thread_detector_on_unanswered_question():
    # cheapest reliable heuristic: a non-running session whose last persisted
    # message is an assistant turn ending in '?'.
    sessions = [
        {"session_id": "s1", "session_uuid": "u1", "status": "waiting",
         "last_activity_at": "2026-07-24T10:00:00+00:00"},
        {"session_id": "s2", "session_uuid": "u2", "status": "running",
         "last_activity_at": "2026-07-24T11:00:00+00:00"},
    ]
    tails = {
        "u1": {"role": "assistant", "content": "Which vendor should I use?",
               "timestamp": "2026-07-24T10:00:00+00:00"},
        "u2": {"role": "assistant", "content": "Working on it now.",
               "timestamp": "2026-07-24T11:00:00+00:00"},
    }
    cards = workbench_cards.paused_thread_cards(
        "proj_a", sessions, lambda uuid: tails.get(uuid),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc))
    assert len(cards) == 1
    assert cards[0]["type"] == "paused_thread"
    assert cards[0]["key"] == "u1"          # resume key = session uuid
    assert cards[0]["text"].endswith("?")


def test_paused_thread_noise_guards():
    """Real-data regression (2026-07-24 screenshot): months-old sessions must
    not surface, only ONE card per project, and the card text is the final
    question sentence — not the whole markdown-laden assistant turn."""
    long_turn = (
        "[STATUS: drafting]\n\nDone. Created `content-bank/drafts/005.md` "
        "with both versions. **How the rigor landed:** 1. **Two-dimensional "
        "compounding** — separated context capture drift from execution "
        "accuracy. Voice check passed. Good to add to your weekly rotation?"
    )
    sessions = [
        {"session_uuid": "old", "status": "idle",
         "last_activity_at": "2026-04-26T10:00:00+00:00"},   # 89 days old
        {"session_uuid": "new1", "status": "waiting",
         "last_activity_at": "2026-07-22T10:00:00+00:00"},
        {"session_uuid": "new2", "status": "idle",
         "last_activity_at": "2026-07-23T10:00:00+00:00"},
    ]
    tails = {
        "old": {"role": "assistant", "content": "Still want this?",
                "timestamp": "2026-04-26T10:00:00+00:00"},
        "new1": {"role": "assistant", "content": "Ship v1 or wait?",
                 "timestamp": "2026-07-22T10:00:00+00:00"},
        "new2": {"role": "assistant", "content": long_turn,
                 "timestamp": "2026-07-23T10:00:00+00:00"},
    }
    cards = workbench_cards.paused_thread_cards(
        "proj_a", sessions, lambda uuid: tails.get(uuid),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc))
    # 89-day-old thread skipped; one card max; newest wins.
    assert len(cards) == 1
    assert cards[0]["key"] == "new2"
    # Text is the final question only, markdown/status noise stripped.
    assert cards[0]["text"] == "Good to add to your weekly rotation?"


async def test_workbench_exclude_global_persists_via_project_update(tmp_path):
    """The privacy toggle PATCHes ``workbench_exclude_global`` through the
    project-update route; the field must be declared on ProjectUpdate so it
    persists (extra fields are silently dropped otherwise). End-to-end: PUT →
    re-GET shows it → global Workbench excludes the project."""
    from agent_os.daemon_v2.project_store import ProjectStore
    from agent_os.api.routes import agents_v2

    store = ProjectStore(data_dir=str(tmp_path / "store"))

    def mk(name):
        ws = tmp_path / name
        (ws / "orbital").mkdir(parents=True)
        (ws / "orbital" / "PROJECT_STATE.md").write_text(SEEDED_STATE, encoding="utf-8")
        return store.create_project({"name": name, "workspace": str(ws),
                                     "timezone": "UTC"})

    pid_excl = mk("projexcl")
    pid_keep = mk("projkeep")

    app = FastAPI()
    agents_v2.configure(store, None, None)   # minimal PUT touches only the store
    app.include_router(agents_v2.router)
    hub = CalendarHub(sources=[], linkage=Linkage(str(tmp_path / "_lk")))
    workbench_routes.configure(store, FakeAgentManager(), hub, now_fn=lambda: NOW)
    app.include_router(workbench_routes.router)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test")

    async with client:
        # Before: both projects surface in the global Workbench.
        before = (await client.get("/api/v2/workbench")).json()
        assert {pid_excl, pid_keep} <= {e["project_id"] for e in before["entries"]}

        # PATCH the toggle through the existing project-update route.
        r = await client.put(f"/api/v2/projects/{pid_excl}",
                             json={"workbench_exclude_global": True})
        assert r.status_code == 200, r.text
        assert r.json().get("workbench_exclude_global") is True

        # Re-GET the project: the field is persisted.
        got = (await client.get(f"/api/v2/projects/{pid_excl}")).json()
        assert got.get("workbench_exclude_global") is True

        # Global Workbench now excludes it; the other project remains.
        after = (await client.get("/api/v2/workbench")).json()
        pids = {e["project_id"] for e in after["entries"]}
        assert pid_excl not in pids
        assert pid_keep in pids

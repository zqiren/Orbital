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


def _make_client(tmp_path, projects, *, agent_manager=None, hub=None):
    store = FakeProjectStore({p["project_id"]: p for p in projects})
    am = agent_manager or FakeAgentManager()
    h = hub or CalendarHub(sources=[], linkage=Linkage(str(tmp_path / "_linkage")))
    app = FastAPI()
    workbench_routes.configure(store, am, h, now_fn=lambda: NOW)
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

    content = open(_state_path(project), encoding="utf-8").read()
    entries = user_flags.parse_entries(content)
    ids = {e.id for e in entries}
    # x7f3a2 unflagged (tag dropped) -> no longer a parsed entry; others intact.
    assert "x7f3a2" not in ids
    assert {"a1b2c3", "d4e5f6"} <= ids
    # The resolved stamp is written into the (now plain-fact) bullet's comment.
    assert "resolved:2026-07-24" in content
    # The sentence survives as a plain fact.
    assert "Send 宝玉 + Simon DM drafts" in content
    # The other flagged entry is byte-for-byte unchanged.
    assert "- [user] Approve the Q3 budget before Friday." in content


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
    assert calls["n"] == 1  # exactly one simulated conflict, one retry
    content = open(_state_path(project), encoding="utf-8").read()
    assert "resolved:2026-07-24" in content
    assert "x7f3a2" not in {e.id for e in user_flags.parse_entries(content)}


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
        "proj_a", sessions, lambda uuid: tails.get(uuid))
    assert len(cards) == 1
    assert cards[0]["type"] == "paused_thread"
    assert cards[0]["key"] == "u1"          # resume key = session uuid
    assert cards[0]["text"].endswith("?")

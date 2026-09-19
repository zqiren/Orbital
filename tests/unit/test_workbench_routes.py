# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/workbench routes on the asks fold (spec 089 §3.6).

Mounts only the workbench router over an in-process ASGI transport with a
fake project_store / agent_manager and a real CalendarHub. Workspaces are tmp
dirs seeded with an ``orbital/ASKS.md`` event log. The cards response shape is
the v0.13.0 one byte-for-byte (phones run an older UI); "Recently closed" is
opt-in via ``?recently_closed=1``.
"""

import os
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agent import asks, workbench_cards
from agent_os.api.routes import workbench as workbench_routes
from agent_os.calendar_hub.hub import CalendarHub
from agent_os.calendar_hub.linkage import Linkage

# Frozen "now" for deterministic age/overdue math (UTC noon on 2026-07-24).
NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
TODAY = "2026-07-24"

# The v0.13.0 card row, exactly. New fields must never appear here.
CARD_KEYS = {
    "project_id", "id", "text", "section", "due", "created", "touched",
    "age_days", "overdue", "days_late",
}

SEEDED_ASKS = "\n".join([
    asks.FORMAT_HEADER,
    "- open 0f7fa2 2026-07-19 due:2026-07-28 Send 宝玉 + Simon DM drafts — only you can send from your accounts.",
    "- open a1b2c3 2026-07-20 Approve the Q3 budget before Friday.",
    "- open d4e5f6 2026-07-15 Ship the marketing site.",
    '- done d4e5f6 2026-07-22 by:agent "shipped, thanks"',
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

    async def new_session(self, project_id, session_id=None):
        self._counter += 1
        sid = f"minted_{self._counter}"
        return {"status": "ok", "session_id": sid, "session_uuid": sid}

    async def inject_message(self, project_id, content, *, session_id=None, nonce=None):
        self.injected.append((project_id, content, session_id))
        return "started"


class CountingHub(CalendarHub):
    def __init__(self, tmp_path):
        super().__init__(sources=[], linkage=Linkage(str(tmp_path / "_linkage")))
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1
        return super().refresh()


def _seed_project(tmp_path, pid="proj_a", *, asks_text=SEEDED_ASKS, extra=None):
    ws = tmp_path / pid
    (ws / "orbital").mkdir(parents=True, exist_ok=True)
    if asks_text is not None:
        (ws / "orbital" / "ASKS.md").write_text(asks_text, encoding="utf-8")
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
    h = hub or CountingHub(tmp_path)
    frozen = now or NOW
    app = FastAPI()
    workbench_routes.configure(store, am, h, now_fn=lambda: frozen)
    app.include_router(workbench_routes.router)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return client, store, am, h


def _orbital(project):
    return os.path.join(project["workspace"], "orbital")


def _asks_file(project):
    with open(os.path.join(_orbital(project), "ASKS.md"), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------
# GET /api/v2/workbench
# --------------------------------------------------------------------------

async def test_get_lists_open_asks_in_the_v013_card_shape(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200, r.text
        body = r.json()
    assert set(body) == {"entries"}                  # nothing new unless asked for
    ids = {e["id"] for e in body["entries"]}
    assert ids == {"0f7fa2", "a1b2c3"}               # the done ask is not a card
    for row in body["entries"]:
        assert set(row) == CARD_KEYS
    x = next(e for e in body["entries"] if e["id"] == "0f7fa2")
    assert x["text"].startswith("Send 宝玉 + Simon DM drafts")
    assert x["due"] == "2026-07-28"
    assert x["created"] == "2026-07-19"
    assert x["age_days"] == 5            # 07-19 -> 07-24
    assert x["overdue"] is False
    assert x["days_late"] is None
    assert x["section"] is None
    assert x["project_id"] == "proj_a"


async def test_get_sort_overdue_first_then_oldest(tmp_path):
    text = "\n".join([
        asks.FORMAT_HEADER,
        "- open 0000a1 2026-07-22 Newer waiting question.",
        "- open 0000a2 2026-07-20 due:2026-07-01 Overdue obligation.",
        "- open 0000a3 2026-07-10 Older waiting question.",
        "",
    ])
    project = _seed_project(tmp_path, asks_text=text)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    assert [e["id"] for e in body["entries"]] == ["0000a2", "0000a3", "0000a1"]


async def test_get_never_rewrites_the_file(tmp_path):
    """Reads never write: an id-less line from an external agent is shown
    under its derived id and left exactly as it was on disk."""
    text = "- open Written by an external agent\n"
    project = _seed_project(tmp_path, asks_text=text)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    assert len(body["entries"]) == 1 and body["entries"][0]["id"]
    assert _asks_file(project) == text


async def test_global_get_survives_a_broken_project(tmp_path):
    healthy = _seed_project(tmp_path, pid="proj_ok")
    bad = _seed_project(tmp_path, pid="proj_bad", asks_text=None)
    with open(os.path.join(_orbital(bad), "ASKS.md"), "wb") as f:
        f.write(b"\xff\xfe not valid utf-8 \x80\x81\n- open garbled\n")
    client, *_ = _make_client(tmp_path, [healthy, bad])
    async with client:
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200, r.text
    assert {"0f7fa2", "a1b2c3"} <= {e["id"] for e in r.json()["entries"]}


async def test_project_without_asks_file_is_empty(tmp_path):
    project = _seed_project(tmp_path, asks_text=None)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    assert body == {"entries": []}


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
# Exits: Done / Delete append events
# --------------------------------------------------------------------------

async def test_done_appends_a_user_close(tmp_path):
    project = _seed_project(tmp_path)
    client, _store, _am, hub = _make_client(tmp_path, [project])
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/entries/a1b2c3/exit",
                              json={"kind": "fulfilled"})
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok"}
        body = (await client.get("/api/v2/workbench")).json()
    text = _asks_file(project)
    assert text.startswith(SEEDED_ASKS)               # nothing rewritten
    assert text.endswith(f"- done a1b2c3 {TODAY} by:user\n")
    assert "a1b2c3" not in {e["id"] for e in body["entries"]}
    assert hub.refreshes == 1


async def test_delete_appends_a_user_drop_with_reason(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/entries/0f7fa2/exit",
                              json={"kind": "irrelevant", "reason": "changed my mind"})
        assert r.status_code == 200, r.text
    text = _asks_file(project)
    assert text.endswith(f"- dropped 0f7fa2 {TODAY} by:user changed my mind\n")
    got = {a.id: a for a in asks.read_asks(_orbital(project))}["0f7fa2"]
    assert got.state == "dropped" and got.closed_by == "user"


async def test_exit_uses_the_project_timezone_date(tmp_path):
    project = _seed_project(tmp_path, extra={"timezone": "Asia/Shanghai"})
    late_utc = datetime(2026, 7, 24, 20, 0, tzinfo=timezone.utc)  # 07-25 in Shanghai
    client, *_ = _make_client(tmp_path, [project], now=late_utc)
    async with client:
        await client.post("/api/v2/workbench/proj_a/entries/a1b2c3/exit",
                          json={"kind": "fulfilled"})
    assert _asks_file(project).endswith("- done a1b2c3 2026-07-25 by:user\n")


async def test_exit_unknown_id_is_404(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/entries/zzzzzz/exit",
                              json={"kind": "fulfilled"})
        assert r.status_code == 404
    assert _asks_file(project) == SEEDED_ASKS


async def test_exit_of_an_already_closed_ask_is_a_no_op(tmp_path):
    project = _seed_project(tmp_path)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/entries/d4e5f6/exit",
                              json={"kind": "irrelevant"})
        assert r.status_code == 200
    assert _asks_file(project) == SEEDED_ASKS


async def test_exit_of_an_idless_line_lines_up_with_its_stamp(tmp_path):
    project = _seed_project(tmp_path, asks_text="- open Written by an external agent\n")
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        ask_id = (await client.get("/api/v2/workbench")).json()["entries"][0]["id"]
        r = await client.post(f"/api/v2/workbench/proj_a/entries/{ask_id}/exit",
                              json={"kind": "fulfilled"})
        assert r.status_code == 200
        assert (await client.get("/api/v2/workbench")).json()["entries"] == []
    assert f"- open {ask_id} {TODAY} Written by an external agent" in _asks_file(project)


# --------------------------------------------------------------------------
# Recently closed (opt-in) + reopen
# --------------------------------------------------------------------------

CLOSED_ASKS = "\n".join([
    asks.FORMAT_HEADER,
    "- open 00000a 2026-07-01 Closed by the agent this week",
    '- done 00000a 2026-07-22 by:agent "yes, done"',
    "- open 00000b 2026-07-01 Closed by the editor, dropped",
    '- dropped 00000b 2026-07-23 by:editor "forget it"',
    "- open 00000c 2026-07-01 Closed by the user",
    "- done 00000c 2026-07-23 by:user",
    "- open 00000d 2026-07-01 Closed by the agent too long ago",
    '- done 00000d 2026-07-10 by:agent "old"',
    "- open 00000e 2026-07-01 Still open",
    "",
])


async def test_recently_closed_is_opt_in(tmp_path):
    project = _seed_project(tmp_path, asks_text=CLOSED_ASKS)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        plain = (await client.get("/api/v2/workbench")).json()
        opted = (await client.get("/api/v2/workbench",
                                  params={"recently_closed": "1"})).json()
    assert "recently_closed" not in plain
    assert [e["id"] for e in opted["entries"]] == ["00000e"]
    rows = opted["recently_closed"]
    # agent/editor closes within 7 days, newest first; the user's own close
    # and the stale one are not offered for undo.
    assert [r["id"] for r in rows] == ["00000b", "00000a"]
    b = rows[0]
    assert b["kind"] == "dropped" and b["closed_by"] == "editor"
    assert b["closed"] == "2026-07-23" and b["note"] == '"forget it"'
    assert b["project_id"] == "proj_a" and b["text"] == "Closed by the editor, dropped"


async def test_recently_closed_respects_the_privacy_toggle(tmp_path):
    a = _seed_project(tmp_path, pid="proj_a", asks_text=CLOSED_ASKS)
    b = _seed_project(tmp_path, pid="proj_b", asks_text=CLOSED_ASKS,
                      extra={"workbench_exclude_global": True})
    client, *_ = _make_client(tmp_path, [a, b])
    async with client:
        body = (await client.get("/api/v2/workbench",
                                 params={"recently_closed": "true"})).json()
    assert {r["project_id"] for r in body["recently_closed"]} == {"proj_a"}


async def test_reopen_appends_and_the_card_comes_back(tmp_path):
    project = _seed_project(tmp_path, asks_text=CLOSED_ASKS)
    client, _s, _a, hub = _make_client(tmp_path, [project])
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/asks/00000a/reopen")
        assert r.status_code == 200, r.text
        body = (await client.get("/api/v2/workbench",
                                 params={"recently_closed": "1"})).json()
    assert _asks_file(project).endswith(f"- reopen 00000a {TODAY} by:user\n")
    assert "00000a" in {e["id"] for e in body["entries"]}
    assert "00000a" not in {r["id"] for r in body["recently_closed"]}
    assert hub.refreshes == 1


async def test_reopen_unknown_is_404_and_open_is_a_no_op(tmp_path):
    project = _seed_project(tmp_path, asks_text=CLOSED_ASKS)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        assert (await client.post(
            "/api/v2/workbench/proj_a/asks/ffffff/reopen")).status_code == 404
        assert (await client.post(
            "/api/v2/workbench/proj_a/asks/00000e/reopen")).status_code == 200
        assert (await client.post(
            "/api/v2/workbench/nope/asks/00000a/reopen")).status_code == 404
    assert _asks_file(project) == CLOSED_ASKS


# --------------------------------------------------------------------------
# Migrate (spawn seam) — the empty-state CTA now reviews PROJECT_STATE into
# asks instead of flagging lines in place.
# --------------------------------------------------------------------------

async def test_migrate_refreshes_header_and_spawns_an_asks_review(tmp_path):
    from agent_os.agent.memory_entries import FORMAT_HEADERS
    project = _seed_project(tmp_path)
    state_path = os.path.join(_orbital(project), "PROJECT_STATE.md")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("<!--format legacy header placeholder-->\n## Focus\n- A fact.\n")
    am = FakeAgentManager()
    client, *_ = _make_client(tmp_path, [project], agent_manager=am)
    async with client:
        r = await client.post("/api/v2/workbench/proj_a/migrate")
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == "minted_1"

    content = open(state_path, encoding="utf-8").read()
    assert content.startswith(FORMAT_HEADERS["state"])
    assert "- A fact." in content
    assert len(am.injected) == 1
    msg = am.injected[0][1]
    assert "orbital/ASKS.md" in msg
    assert "- open <text>" in msg
    assert "this message IS the confirmation" in msg
    assert "do not ask" in msg.lower()
    assert "[user]" not in msg


# --------------------------------------------------------------------------
# Timezone / overdue / age math (unit-level)
# --------------------------------------------------------------------------

def test_overdue_boundary_uses_project_tz_not_utc():
    # UTC 2026-07-23T20:00 is 2026-07-24T04:00 in Shanghai (UTC+8).
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    # A due date of 2026-07-23: still "today" in UTC (not overdue), but already
    # yesterday in Shanghai (overdue).
    assert workbench_cards.is_overdue("2026-07-23", "UTC", now=now) is False
    assert workbench_cards.is_overdue("2026-07-23", "Asia/Shanghai", now=now) is True


def test_days_late_uses_project_tz_not_utc():
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    d = "2026-07-23"
    assert workbench_cards.days_late(d, "UTC", now=now) is None          # today in UTC
    assert workbench_cards.days_late(d, "Asia/Shanghai", now=now) == 1   # yesterday there
    assert workbench_cards.days_late("2026-08-01", "Asia/Shanghai", now=now) is None
    assert workbench_cards.days_late(None, "UTC", now=now) is None


async def test_entry_row_days_late_computed_in_project_tz(tmp_path):
    text = "\n".join([
        asks.FORMAT_HEADER,
        "- open 1a7e01 2026-07-20 due:2026-07-23 Confirm the venue booking.",
        "",
    ])
    project = _seed_project(tmp_path, asks_text=text,
                            extra={"timezone": "Asia/Shanghai"})
    now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
    client, *_ = _make_client(tmp_path, [project], now=now)
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
    row = body["entries"][0]
    assert row["overdue"] is True
    assert row["days_late"] == 1     # project tz (Shanghai), not UTC


def test_project_timezone_precedence():
    triggers = [{"type": "schedule", "schedule": {"cron": "0 9 * * *",
                                                  "timezone": "Europe/Paris"}}]
    assert workbench_cards.project_timezone(
        {"timezone": "Asia/Tokyo"}, triggers) == "Asia/Tokyo"
    assert workbench_cards.project_timezone({}, triggers) == "Europe/Paris"
    assert workbench_cards.project_timezone({}, []) not in (None, "")


async def test_workbench_exclude_global_persists_via_project_update(tmp_path):
    """The privacy toggle PATCHes ``workbench_exclude_global`` through the
    project-update route; the field must persist and the global Workbench
    must then exclude the project."""
    from agent_os.daemon_v2.project_store import ProjectStore
    from agent_os.api.routes import agents_v2

    store = ProjectStore(data_dir=str(tmp_path / "store"))

    def mk(name):
        ws = tmp_path / name
        (ws / "orbital").mkdir(parents=True)
        (ws / "orbital" / "ASKS.md").write_text(SEEDED_ASKS, encoding="utf-8")
        return store.create_project({"name": name, "workspace": str(ws),
                                     "timezone": "UTC"})

    pid_excl = mk("projexcl")
    pid_keep = mk("projkeep")

    app = FastAPI()
    agents_v2.configure(store, None, None)
    app.include_router(agents_v2.router)
    hub = CalendarHub(sources=[], linkage=Linkage(str(tmp_path / "_lk")))
    workbench_routes.configure(store, FakeAgentManager(), hub, now_fn=lambda: NOW)
    app.include_router(workbench_routes.router)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test")

    async with client:
        before = (await client.get("/api/v2/workbench")).json()
        assert {pid_excl, pid_keep} <= {e["project_id"] for e in before["entries"]}
        r = await client.put(f"/api/v2/projects/{pid_excl}",
                             json={"workbench_exclude_global": True})
        assert r.status_code == 200, r.text
        got = (await client.get(f"/api/v2/projects/{pid_excl}")).json()
        assert got.get("workbench_exclude_global") is True
        after = (await client.get("/api/v2/workbench")).json()
        pids = {e["project_id"] for e in after["entries"]}
        assert pid_excl not in pids
        assert pid_keep in pids

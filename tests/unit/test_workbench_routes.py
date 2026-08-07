# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/workbench routes (Task 5).

Mounts only the workbench router over an in-process ASGI transport with a
fake project_store / agent_manager and a real CalendarHub (no sources needed —
Task 5 only calls ``hub.refresh()``). Workspaces are tmp dirs seeded with a
PROJECT_STATE.md written in the ``[user]`` grammar.
"""

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

    async def new_session(self, project_id, session_id=None):
        self._counter += 1
        sid = f"minted_{self._counter}"
        return {"status": "ok", "session_id": sid, "session_uuid": sid}

    async def inject_message(self, project_id, content, *, session_id=None, nonce=None):
        self.injected.append((project_id, content, session_id))
        return "started"


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
    # Receipt attributes are legacy-parseable but never cross the wire (the
    # receipt cut, rev 6): verbatim quotes must not leave the daemon when no
    # UI renders them.
    assert "confidence" not in x
    assert "from_session" not in x
    assert "evidence" not in x
    assert x["age_days"] == 5           # 07-19 -> 07-24
    assert x["overdue"] is False
    assert x["project_id"] == "proj_a"
    assert x["section"] == "Focus"      # nearest preceding '## ' heading

    # The unflagged dated fact never surfaces — not as an entry, and the
    # computed-card system that used to promote it as "overdue" is gone.
    assert "d4e5f6" not in ids


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
    # No '## ' heading anywhere above these entries -> section is null.
    assert all(e["section"] is None for e in body["entries"])


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


async def test_fulfilled_exit_numbered_item_preserves_prefix(tmp_path):
    """A numbered flagged item retiring to a fact must stay a numbered item —
    `f"{entry.prefix}{entry.text}"`, not a hardcoded `- ` bullet — so the
    surrounding list's numbering and any cross-references into it survive."""
    state = "\n".join([
        FORMAT_LINE,
        "## Blockers",
        "3. [user] Approve the numbered blocker.",
        '  <!--mem id:num001 from:s1 evidence:"approve it" created:2026-07-19-->',
        "",
    ])
    project = _seed_project(tmp_path, state=state)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.post(
            "/api/v2/workbench/proj_a/entries/num001/exit",
            json={"kind": "fulfilled", "reason": "done"},
        )
        assert r.status_code == 200, r.text

    lines = open(_state_path(project), encoding="utf-8").read().split("\n")
    idx = next(i for i, ln in enumerate(lines) if "Approve the numbered blocker." in ln)
    assert lines[idx] == "3. Approve the numbered blocker."
    assert lines[idx + 1].lstrip().startswith("<!--mem")
    assert "resolved:2026-07-24" in lines[idx + 1]


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
# Migrate (spawn seam)
# --------------------------------------------------------------------------

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
    # A migration session was spawned with the one-voice migration instruction
    # (day-0 flagging AND normalization of already-flagged files, in one edit).
    assert len(am.injected) == 1
    # The instruction must be imperative — edit NOW, no permission-seeking
    # (2026-07-24 live regression: the agent presented an options menu and
    # stalled instead of applying tags).
    msg = am.injected[0][1]
    assert "bring it fully to the format header's rails" in msg
    assert "do not ask" in msg.lower()
    assert "this message IS the confirmation" in msg
    assert "FLAG IN PLACE" in msg
    assert "ONE VOICE" in msg
    assert "COMMENTS" in msg
    assert "SETTLED LINES" in msg
    # Never move a line or convert a numbered item to a bullet.
    assert "never convert a numbered item to a bullet" in msg


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


# --------------------------------------------------------------------------
# Lazy id heal (bug #45): a flagged bullet with no id (no mem-comment, or a
# comment carrying no id:) parses to id=None. That id-less row collapses onto a
# single React key on the client (duplicate-key phantom card on delete) AND
# cannot be exited (POST /entries/null/exit 404s). The read path mints and
# persists a stable id for each such surfaced entry.
# --------------------------------------------------------------------------

IDLESS_STATE = "\n".join([
    FORMAT_LINE,
    "## Focus",
    "- [user] Approve the Q3 budget before Friday.",
    "- [user] Send Simon the invoice draft.",
    "",
])


async def test_get_mints_ids_for_idless_flagged_entries(tmp_path):
    """A flagged bullet with no mem-comment must NOT surface with id=None — the
    read path mints a stable id, persists it to PROJECT_STATE.md, and is
    idempotent across reads."""
    project = _seed_project(tmp_path, state=IDLESS_STATE)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        body = (await client.get("/api/v2/workbench")).json()
        ids = [e["id"] for e in body["entries"]]
        assert len(ids) == 2
        # Non-null, non-empty, and distinct — no two entries share a key.
        assert all(i for i in ids), f"expected minted ids, got {ids}"
        assert len(set(ids)) == 2

        # Persisted to disk — the exit route resolves by the on-file id.
        disk = open(_state_path(project), encoding="utf-8").read()
        for i in ids:
            assert f"id:{i}" in disk
        # The sentences and flags survive the heal untouched.
        assert "Approve the Q3 budget before Friday." in disk
        assert "Send Simon the invoice draft." in disk

        # Idempotent: a second GET returns the SAME ids (no re-mint / churn).
        body2 = (await client.get("/api/v2/workbench")).json()
        assert [e["id"] for e in body2["entries"]] == ids


async def test_exit_of_idless_entry_heals_then_succeeds(tmp_path):
    """An exit against a (formerly id-less) flagged entry returns 200 and
    shrinks the list — because the read path already minted+persisted the id
    the client posts back."""
    project = _seed_project(tmp_path, state=IDLESS_STATE)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        before = (await client.get("/api/v2/workbench")).json()["entries"]
        assert len(before) == 2
        target = before[0]["id"]
        assert target, "read path must have minted a real id"

        r = await client.post(
            f"/api/v2/workbench/proj_a/entries/{target}/exit",
            json={"kind": "irrelevant", "reason": "not needed"},
        )
        assert r.status_code == 200, r.text

        after = (await client.get("/api/v2/workbench")).json()["entries"]
        assert len(after) == 1
        assert target not in {e["id"] for e in after}


async def test_get_excludes_flagged_but_resolved_entries(tmp_path):
    """A resolved entry never renders as a card, even if it is still flagged.

    The write chokepoint unflags these, but a file already on disk (written
    before that guard, or by any writer that bypasses it) must not resurrect a
    card the user already closed with "Done". Read-side guarantee.
    """
    state = "\n".join([
        FORMAT_LINE,
        "## Focus",
        "- [user] Approve the Q3 budget before Friday.",
        "  <!--mem id:open01 created:2026-07-20-->",
        "- [user] Pick option A, B or C (default A)?",
        "  <!--mem id:done01 created:2026-07-20 resolved:2026-07-22-->",
        "",
    ])
    project = _seed_project(tmp_path, state=state)
    client, *_ = _make_client(tmp_path, [project])
    async with client:
        r = await client.get("/api/v2/workbench")
        assert r.status_code == 200, r.text
        ids = {e["id"] for e in r.json()["entries"]}

    assert "open01" in ids
    assert "done01" not in ids, "a resolved entry must not come back as a card"

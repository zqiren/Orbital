# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 160: imported projects, Orbital guesses first.

* §5.1 the import writes ``orbital/`` at once (keyless too); the root
  ``AGENTS.md`` still waits for first use, and ``deferred_seed`` keeps
  meaning "root file not yet written".
* §5.2 a daemon-side queue reads each imported project once with the
  manager's key: 2 at a time, newest activity first, persisted on the project
  (``import_scan``), resumed after a restart, skipped when goals exist,
  retried once on the next trigger after a failure. Only projects stamped
  ``pending`` by the import are ever scanned (open decision 1: only new
  imports), and only on decidable events.
* §5.3 ``unscanned_imports`` on ``GET /settings``.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.onboarding import import_scan
from agent_os.onboarding.deferred_seed import (
    ORBITAL_SEEDED_FIELD, ensure_project_seeded, seed_orbital_folder,
)
from agent_os.onboarding.import_scan import (
    DONE, FAILED, PENDING, RUNNING, SKIPPED, ImportScanQueue, unscanned_count,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(str(tmp_path / "data"))


def _goals(ws) -> str:
    return os.path.join(str(ws), "orbital", "instructions", "project_goals.md")


def _write_goals(ws, text="# Mission\nx\n"):
    path = _goals(ws)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _imported(store, tmp_path, name, *, activity=None, state=PENDING, attempts=None):
    ws = tmp_path / f"ws-{name}"
    ws.mkdir()
    fields = {"name": name, "workspace": str(ws), "deferred_seed": True, "imported": True}
    if state is not None:
        fields.update(import_scan.pending_fields(activity))
        fields[import_scan.STATE_FIELD] = state
    if attempts is not None:
        fields[import_scan.ATTEMPTS_FIELD] = attempts
    return store.create_project(fields), ws


# ---------------------------------------------------------------------------
# §5.1 the seed split
# ---------------------------------------------------------------------------


def test_orbital_part_writes_skills_but_not_agents_md(store, tmp_path):
    pid, ws = _imported(store, tmp_path, "a")
    assert seed_orbital_folder(store, pid) is True
    assert (ws / "orbital" / "skills").is_dir()
    assert not (ws / "AGENTS.md").exists()
    p = store.get_project(pid)
    assert p["deferred_seed"] is True  # the root file is still to come
    assert p[ORBITAL_SEEDED_FIELD] is True
    assert seed_orbital_folder(store, pid) is False  # once


def test_first_use_after_the_orbital_part_writes_agents_md_and_skips_the_refold(
        store, tmp_path, monkeypatch):
    from agent_os.onboarding import deferred_seed

    pid, ws = _imported(store, tmp_path, "a")
    seed_orbital_folder(store, pid)
    calls = []
    monkeypatch.setattr(deferred_seed, "fold_private_notes",
                        lambda *a, **k: calls.append(a) or 0)
    assert ensure_project_seeded(store, pid) is True
    assert (ws / "AGENTS.md").is_file()
    assert store.get_project(pid)["deferred_seed"] is False
    # The fold already ran at import: first use must not copy the notes again
    # over anything the user edited since.
    assert calls == []


def test_failed_fold_at_import_is_retried_on_first_use(store, tmp_path, monkeypatch):
    from agent_os.onboarding import deferred_seed

    pid, ws = _imported(store, tmp_path, "a")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(deferred_seed, "fold_private_notes", boom)
    assert seed_orbital_folder(store, pid) is True
    assert not store.get_project(pid).get(ORBITAL_SEEDED_FIELD)
    calls = []
    monkeypatch.setattr(deferred_seed, "fold_private_notes",
                        lambda *a, **k: calls.append(a) or 0)
    assert ensure_project_seeded(store, pid) is True
    assert len(calls) == 1
    assert store.get_project(pid)["deferred_seed"] is False


# ---------------------------------------------------------------------------
# §5.2 the queue (fake runner: the model never runs here)
# ---------------------------------------------------------------------------


class FakeRunner:
    """Stands in for the manager scan. ``outcomes[pid]`` decides a run:
    "goals" writes the goals file, "fail" writes nothing, "raise" fails the
    start. Every run blocks until ``release`` is set, so concurrency can be
    observed."""

    def __init__(self, store, *, ready=True, outcomes=None, hold=False):
        self.store = store
        self.ready_value = ready
        self.outcomes = outcomes or {}
        self.started: list[str] = []
        self.runners: list[str] = []
        self.discarded: list[str] = []
        self.confirmed: list[str] = []
        self.running = 0
        self.max_running = 0
        self.release = asyncio.Event()
        if not hold:
            self.release.set()

    def pick(self, pid):
        ready = self.ready_value(pid) if callable(self.ready_value) else self.ready_value
        if ready is True:
            return "orbital"
        return ready or None  # a worker name, or nobody

    async def start(self, pid, who):
        outcome = self.outcomes.get(pid, "goals")
        if outcome == "raise":
            raise RuntimeError("provider said no")
        self.started.append(pid)
        self.runners.append(who)
        return f"sess-{pid}-{len(self.started)}"

    async def wait(self, pid, sid, who):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await self.release.wait()
            if self.outcomes.get(pid, "goals") == "goals":
                _write_goals(self.store.get_project(pid)["workspace"])
        finally:
            self.running -= 1

    async def discard(self, pid, sid):
        self.discarded.append(sid)

    async def confirm(self, pid, sid):
        self.confirmed.append(sid)


def _run(coro):
    return asyncio.run(coro)


def test_two_at_a_time_newest_activity_first(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a", activity="2026-10-01T00:00:00Z")
    b, _ = _imported(store, tmp_path, "b", activity="2026-11-01T00:00:00Z")
    c, _ = _imported(store, tmp_path, "c", activity="2026-10-15T00:00:00Z")
    d, _ = _imported(store, tmp_path, "d", activity=None)

    async def go():
        runner = FakeRunner(store, hold=True)
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        for _ in range(50):
            await asyncio.sleep(0)
        assert set(runner.started) == {b, c}  # two slots, the two newest
        assert store.get_project(b)[import_scan.STATE_FIELD] == RUNNING
        runner.release.set()
        await q.idle()
        return runner

    runner = _run(go())
    assert set(runner.started[:2]) == {b, c} and set(runner.started[2:]) == {a, d}
    assert runner.max_running == 2
    for pid in (a, b, c, d):
        p = store.get_project(pid)
        assert p[import_scan.STATE_FIELD] == DONE
        assert p[import_scan.AT_FIELD]


def test_projects_without_the_field_are_never_scanned(store, tmp_path):
    """Open decision 1: a project imported before spec 160 has no
    ``import_scan`` field and stays out, even on a key event."""
    old, _ = _imported(store, tmp_path, "old", state=None)
    plain_ws = tmp_path / "plain"
    plain_ws.mkdir()
    plain = store.create_project({"name": "plain", "workspace": str(plain_ws)})

    async def go():
        runner = FakeRunner(store)
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        await q.idle()
        return runner

    runner = _run(go())
    assert runner.started == []
    assert import_scan.STATE_FIELD not in store.get_project(old)
    assert import_scan.STATE_FIELD not in store.get_project(plain)


def test_keyless_leaves_projects_pending_until_a_key_event(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        runner = FakeRunner(store, ready=False)
        q = ImportScanQueue(store, runner)
        q.kick("import")
        await q.idle()
        assert runner.started == []
        assert store.get_project(a)[import_scan.STATE_FIELD] == PENDING
        runner.ready_value = True
        q.kick("card_created")
        await q.idle()
        return runner

    runner = _run(go())
    assert runner.started == [a]
    assert store.get_project(a)[import_scan.STATE_FIELD] == DONE


def test_goals_already_written_skips_the_scan(store, tmp_path):
    a, ws = _imported(store, tmp_path, "a")
    _write_goals(ws)

    async def go():
        runner = FakeRunner(store)
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        await q.idle()
        return runner

    runner = _run(go())
    assert runner.started == []
    assert store.get_project(a)[import_scan.STATE_FIELD] == SKIPPED


def test_failure_retries_once_on_the_next_trigger_then_stops(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        runner = FakeRunner(store, outcomes={a: "fail"})
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        await q.idle()
        p = store.get_project(a)
        assert p[import_scan.STATE_FIELD] == FAILED
        assert p[import_scan.ATTEMPTS_FIELD] == 1
        assert len(runner.started) == 1  # not retried inside the same trigger
        assert runner.discarded == ["sess-%s-1" % a]
        q.kick("default_card_changed")
        await q.idle()
        assert len(runner.started) == 2
        assert store.get_project(a)[import_scan.ATTEMPTS_FIELD] == 2
        q.kick("card_created")
        await q.idle()
        return runner

    runner = _run(go())
    assert len(runner.started) == 2  # given up: today's first-use behaviour
    assert store.get_project(a)[import_scan.STATE_FIELD] == FAILED


def test_a_start_error_counts_as_a_failure(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        runner = FakeRunner(store, outcomes={a: "raise"})
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        await q.idle()

    _run(go())
    p = store.get_project(a)
    assert p[import_scan.STATE_FIELD] == FAILED and p[import_scan.ATTEMPTS_FIELD] == 1


def test_resume_after_restart_picks_pending_and_crashed_running_but_not_failed(store, tmp_path):
    pend, _ = _imported(store, tmp_path, "pend")
    crashed, _ = _imported(store, tmp_path, "crashed", state=RUNNING)
    failed, _ = _imported(store, tmp_path, "failed", state=FAILED, attempts=1)
    done, _ = _imported(store, tmp_path, "done", state=DONE)
    # A restart: a fresh store reads the persisted states.
    store2 = ProjectStore(str(tmp_path / "data"))

    async def go():
        runner = FakeRunner(store2)
        q = ImportScanQueue(store2, runner)
        q.resume()
        await q.idle()
        return runner

    runner = _run(go())
    assert sorted(runner.started) == sorted([pend, crashed])
    assert store2.get_project(failed)[import_scan.STATE_FIELD] == FAILED
    assert store2.get_project(done)[import_scan.STATE_FIELD] == DONE


def test_a_project_in_flight_is_not_started_twice(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        runner = FakeRunner(store, hold=True)
        q = ImportScanQueue(store, runner)
        q.kick("import")
        for _ in range(20):
            await asyncio.sleep(0)
        q.kick("card_created")
        q.kick("default_card_changed")
        for _ in range(20):
            await asyncio.sleep(0)
        runner.release.set()
        await q.idle()
        return runner

    assert _run(go()).started == [a]


def test_module_kick_is_a_noop_until_configured():
    import_scan.configure(None)
    import_scan.kick("card_created")  # no loop, no queue: must not raise


# ---------------------------------------------------------------------------
# §5.3 the count
# ---------------------------------------------------------------------------


def test_unscanned_count(store, tmp_path):
    _imported(store, tmp_path, "pend")
    _imported(store, tmp_path, "retry", state=FAILED, attempts=1)
    _imported(store, tmp_path, "gave_up", state=FAILED, attempts=2)
    _imported(store, tmp_path, "done", state=DONE)
    _imported(store, tmp_path, "running", state=RUNNING)
    _imported(store, tmp_path, "legacy", state=None)
    _, ws = _imported(store, tmp_path, "has_goals")
    _write_goals(ws)
    # pending + the failed one with its retry left; running is being read
    # already and spends nothing more on save.
    assert unscanned_count(store) == 2
    assert unscanned_count(None) == 0


def test_settings_get_carries_unscanned_imports(store, tmp_path):
    from agent_os.api.routes import settings as settings_routes

    _imported(store, tmp_path, "a")
    _imported(store, tmp_path, "b")
    fake = MagicMock()
    fake.get_masked = MagicMock(return_value={"llm": {"api_key_set": False}})
    settings_routes.configure(fake, project_store=store)
    app = FastAPI()
    app.include_router(settings_routes.router)
    body = TestClient(app).get("/api/v2/settings").json()
    assert body["unscanned_imports"] == 2
    assert body["llm"] == {"api_key_set": False}


# ---------------------------------------------------------------------------
# §5.2 triggers: the import and every key event kick the queue
# ---------------------------------------------------------------------------


@pytest.fixture
def kicks(monkeypatch):
    seen: list[tuple[str, bool]] = []

    class Q:
        def kick(self, reason, *, include_failed=True):
            seen.append((reason, include_failed))

    import_scan.configure(Q())
    yield seen
    import_scan.configure(None)


def test_import_stamps_pending_writes_orbital_and_kicks(tmp_path, monkeypatch, kicks):
    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import onboarding as onboarding_routes

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))

    store = ProjectStore(str(tmp_path / "data"))
    manager = MagicMock()
    manager._ensure_dispatcher = MagicMock()
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    scanner = MagicMock()
    scanner.analysis = MagicMock(return_value=None)
    scanner.wait = MagicMock()
    scanner.is_ready = MagicMock(return_value=False)
    onboarding_routes.configure(project_store=store, scanner=scanner)
    ws = tmp_path / "folder"
    ws.mkdir()
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    r = TestClient(app).post("/api/v2/onboarding/import",
                             json={"folders": [str(ws)], "lang": "zh"})
    assert r.status_code == 200, r.text
    (created,) = r.json()["created"]
    p = store.get_project(created["project_id"])
    assert p[import_scan.STATE_FIELD] == PENDING
    assert p[import_scan.AT_FIELD]
    assert p[import_scan.LANG_FIELD] == "zh"
    assert p["deferred_seed"] is True
    assert (ws / "orbital" / "skills").is_dir()
    assert not (ws / "AGENTS.md").exists()
    assert kicks == [("import", True)]
    onboarding_routes.configure()


def test_an_import_that_creates_nothing_does_not_kick(tmp_path, kicks):
    from agent_os.api.routes import onboarding as onboarding_routes

    store = ProjectStore(str(tmp_path / "data"))
    scanner = MagicMock()
    scanner.analysis = MagicMock(return_value=None)
    onboarding_routes.configure(project_store=store, scanner=scanner)
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    TestClient(app).post("/api/v2/onboarding/import",
                         json={"folders": [str(tmp_path / "missing")]})
    assert kicks == []
    onboarding_routes.configure()


def _settings_app(tmp_path, monkeypatch):
    from agent_os.api.routes import settings as settings_routes

    card = SimpleNamespace(id="c1", provider="opencode-go", model="m", name="n", base_url=None)
    fake = MagicMock()
    fake.create_card = MagicMock(return_value=card)
    fake.resolve_card = MagicMock(return_value=card)
    fake.get_card = MagicMock(return_value=card)
    fake.default_card = MagicMock(return_value=card)
    fake.masked_card = MagicMock(return_value={"id": "c1"})
    fake.effective_default_card_id = MagicMock(return_value="c1")
    fake.get_masked = MagicMock(return_value={"llm": {}})
    fake.get = MagicMock(return_value=SimpleNamespace(
        llm=SimpleNamespace(fallback_models=[]), scratch_workspace=None,
        user_preferences_path=None, user_memory_path=None))
    creds = MagicMock()
    creds.set_api_key = MagicMock(return_value={"source": "keychain"})
    settings_routes.configure(fake, credential_store=creds,
                              project_store=ProjectStore(str(tmp_path / "d")))

    async def tested(card_id, *, reuse_recent=False):
        return {"ok": True}

    monkeypatch.setattr(settings_routes, "_test_and_record", tested)
    monkeypatch.setattr(settings_routes, "_refresh_agents_in_background", lambda *a: None)
    monkeypatch.setattr(settings_routes, "_model_provider_slugs", lambda: [])
    monkeypatch.setattr(settings_routes, "_validate_card_shape", lambda *a: None)
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app)


def test_card_created_kicks(tmp_path, monkeypatch, kicks):
    client = _settings_app(tmp_path, monkeypatch)
    r = client.post("/api/v2/settings/cards",
                    json={"provider": "opencode-go", "model": "m", "api_key": "sk-x"})
    assert r.status_code == 201, r.text
    assert kicks == [("card_created", True)]


def test_card_rekeyed_kicks_but_a_rename_does_not(tmp_path, monkeypatch, kicks):
    client = _settings_app(tmp_path, monkeypatch)
    client.put("/api/v2/settings/cards/c1", json={"name": "renamed"})
    assert kicks == []
    client.put("/api/v2/settings/cards/c1", json={"api_key": "sk-y"})
    assert kicks == [("card_updated", True)]


def test_default_card_changed_kicks(tmp_path, monkeypatch, kicks):
    client = _settings_app(tmp_path, monkeypatch)
    r = client.put("/api/v2/settings/cards/c1/default")
    assert r.status_code == 200, r.text
    assert kicks == [("default_card_changed", True)]


def test_legacy_key_saved_kicks(tmp_path, monkeypatch, kicks):
    client = _settings_app(tmp_path, monkeypatch)
    r = client.put("/api/v2/settings/api-key", json={"api_key": "sk-z"})
    assert r.status_code == 200, r.text
    assert kicks == [("legacy_key_saved", True)]


# ---------------------------------------------------------------------------
# §5.2 / §5.4 the prompts
# ---------------------------------------------------------------------------


def _prompt_ctx(workspace, **over):
    from agent_os.agent.prompt_builder import PromptContext
    from agent_os.daemon_v2.models import Autonomy

    base = dict(
        workspace=str(workspace), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[{"handle": "claude-code", "display_name": "Claude Code",
                         "type": "cli"}],
        tool_names=["read", "write", "edit", "shell"], os_type="macos",
        datetime_now="2026-11-06T10:00:00+00:00", project_name="lib",
        project_instructions="", agent_name="lib", project_id="proj_x",
    )
    base.update(over)
    return PromptContext(**base)


def _section(tmp_path, **over):
    from agent_os.agent.prompt_builder import PromptBuilder
    return PromptBuilder(str(tmp_path))._onboarding_or_directive(
        _prompt_ctx(tmp_path, **over))


def _flat(text):
    return " ".join(text.split())


@pytest.fixture
def en_locale(monkeypatch):
    from agent_os.onboarding import deferred_seed
    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")


def test_guess_scan_writes_first_then_asks(tmp_path, en_locale):
    from agent_os.agent import onboarding_prompts as op

    text = _section(tmp_path, cold_start="guess")
    flat = _flat(text)
    assert text.startswith("## COLD-START WORKSPACE SCAN")
    assert "Write NOTHING until the user confirms" not in text
    assert "Do not wait for a confirmation before writing" in flat
    # State + index first, goals LAST (the prompt turns into the directive
    # the moment the goals file exists).
    i_state = flat.index("PROJECT_STATE.md")
    i_index = flat.index("orbital/INDEX.md")
    i_goals = flat.index("project_goals.md")
    assert i_state < i_index < i_goals
    assert f"{tmp_path}/orbital/instructions/project_goals.md" in text
    # Round 4: no closing question — the daemon posts the confirm card.
    # Round 5: the summary ends with 1-3 numbered next steps, the same ones
    # recorded under PROJECT_STATE's "## Next steps" (the card reads them
    # there); still never started.
    assert "is this right?" not in flat and "这样理解对吗" not in flat
    assert "do not suggest a next step" not in flat
    assert "End the reply with the same steps, numbered" in flat
    assert "`## Next steps` heading (exactly that heading" in flat
    assert "Do not ask a question and do not start any step" in flat
    # Live round 4: one scan ended with mark_task_complete and no summary.
    assert "That summary is your final message, as plain text" in flat
    assert "Do not start any work" in flat
    assert op.SHARED_MEMORY_LINE_EN in flat
    assert op.SHARED_MEMORY_LINE_ZH not in flat
    # The card's propose-first flavour is untouched.
    assert _section(tmp_path, cold_start=True) == op.cold_start_instruction(str(tmp_path))


def test_guess_reply_language_follows_the_system(tmp_path, monkeypatch):
    from agent_os.agent import onboarding_prompts as op
    from agent_os.onboarding import deferred_seed

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "zh")
    flat = _flat(_section(tmp_path, cold_start="guess"))
    assert "Reply to the user in Chinese" in flat
    assert op.SHARED_MEMORY_LINE_ZH in flat
    assert "这样理解对吗" not in flat


def test_shared_memory_line_only_with_more_than_one_agent(tmp_path, en_locale):
    from agent_os.agent import onboarding_prompts as op

    one = _section(tmp_path, cold_start="guess", enabled_agents=[])
    assert op.SHARED_MEMORY_LINE_EN not in _flat(one)
    greet_many = _flat(_section(tmp_path))
    greet_one = _flat(_section(tmp_path, enabled_agents=[]))
    assert "## ONBOARDING MODE" in greet_many
    assert op.SHARED_MEMORY_LINE_ZH in greet_many and op.SHARED_MEMORY_LINE_EN in greet_many
    assert op.SHARED_MEMORY_LINE_EN not in greet_one
    scratch = _flat(_section(tmp_path, is_scratch=True))
    assert op.SHARED_MEMORY_LINE_EN not in scratch


def test_guess_run_keeps_its_section_after_the_goals_write(tmp_path, en_locale):
    """Round 4: the scan run's prompt does not switch to the directive when
    its goals file appears (the guess-aware directive tail is retired); a
    later turn — cold_start reset — gets the plain directive, byte-identical."""
    from agent_os.agent import onboarding_prompts as op

    _write_goals(tmp_path, "# Mission\nA small parser library.\n")
    flat = _flat(_section(tmp_path, cold_start="guess"))
    assert flat.startswith("## COLD-START WORKSPACE SCAN")
    assert not hasattr(op, "guess_first_directive_tail")
    plain = _flat(_section(tmp_path))
    assert plain.startswith("## PROJECT DIRECTIVE")
    assert "what you would do first and which agent would do it" in plain
    assert _flat(_section(tmp_path, cold_start=False)) == plain


def test_gate_arms_on_the_guess_runs_goals_write(tmp_path):
    """The existing onboarding gate (onboarding_gate.creates_goals) fires on
    the scan's goals creation; state/index writes stay allowed after it."""
    from agent_os.agent import onboarding_gate

    goals = _goals(tmp_path)
    tc = {"name": "write", "arguments": {"path": goals, "content": "# Mission\n"}}
    assert onboarding_gate.creates_goals(str(tmp_path), tc) is True
    for rel in ("orbital/PROJECT_STATE.md", "orbital/INDEX.md"):
        assert onboarding_gate.allowed(str(tmp_path), {
            "name": "write", "arguments": {"path": os.path.join(str(tmp_path), rel)}})
    assert not onboarding_gate.allowed(str(tmp_path), {"name": "shell", "arguments": {}})


def test_the_import_language_reaches_the_guess_prompt(tmp_path, monkeypatch):
    from agent_os.agent import onboarding_prompts as op
    from agent_os.onboarding import deferred_seed

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")
    assert import_scan.guess_flavour({import_scan.LANG_FIELD: "zh"}) == "guess:zh"
    assert import_scan.guess_flavour({import_scan.LANG_FIELD: "fr"}) == "guess"
    assert import_scan.guess_flavour({}) == "guess"
    assert import_scan.LANG_FIELD not in import_scan.pending_fields(None, lang="xx")
    flat = _flat(_section(tmp_path, cold_start="guess:zh"))
    assert "Reply to the user in Chinese" in flat and op.SHARED_MEMORY_LINE_ZH in flat
    assert "Reply to the user in English" in _flat(_section(tmp_path, cold_start="guess"))


def test_skeleton_leaves_out_orbitals_own_folder(tmp_path):
    """orbital/ exists before the scan now (§5.1); its files are not the
    project and must not crowd the inventory the scan plans from."""
    from agent_os.agent.workspace_scan import scan_workspace

    (tmp_path / "README.md").write_text("hi")
    (tmp_path / "orbital" / "history").mkdir(parents=True)
    (tmp_path / "orbital" / "INDEX.md").write_text("x")
    (tmp_path / "orbital" / "history" / "blob").write_text("x" * 50)
    (tmp_path / "orbitals.md").write_text("kept: only the folder named orbital is ours")
    text = scan_workspace(str(tmp_path))
    assert "README.md" in text and "orbitals.md" in text
    assert "orbital/" not in text and "orbital\\" not in text
    assert text.startswith("[WORKSPACE SKELETON] 2 files")


# ---------------------------------------------------------------------------
# Round 2: keyless scan on the pinned worker, one queue for both runners
# ---------------------------------------------------------------------------


def test_one_queue_both_runners_two_slots(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a", activity="2026-11-03T00:00:00Z")
    b, _ = _imported(store, tmp_path, "b", activity="2026-11-02T00:00:00Z")
    c, _ = _imported(store, tmp_path, "c", activity="2026-11-01T00:00:00Z")

    async def go():
        runner = FakeRunner(store, ready=lambda pid: True if pid == a else "worker")
        q = ImportScanQueue(store, runner)
        q.kick("key_skipped")
        await q.idle()
        return runner

    runner = _run(go())
    # The two slots fill together (the runner pick runs off the loop, so
    # their start order may interleave); the oldest waits for a free slot.
    assert set(runner.started[:2]) == {a, b} and runner.started[2] == c
    assert dict(zip(runner.started, runner.runners)) == {a: "orbital", b: "worker", c: "worker"}
    assert runner.max_running <= 2
    assert all(store.get_project(p)[import_scan.STATE_FIELD] == DONE for p in (a, b, c))


class _Mgr:
    def __init__(self, *, key=False, skipped=False, worker="claude-code"):
        self.key, self.worker = key, worker
        self._settings_store = SimpleNamespace(get=lambda: SimpleNamespace(
            setup_api_key_skipped=skipped, sub_agents_hidden=[]))
        self._registry = SimpleNamespace(get=lambda slug: SimpleNamespace(name="Claude Code")
                                         if slug == "claude-code" else None)

    def manager_ready(self, pid):
        return self.key

    def effective_pinned_target(self, pid, ready=None):
        return self.worker

    def available_worker_slugs(self, pid, project=None):
        return [self.worker] if self.worker else []


def test_runner_choice_per_run():
    from agent_os.onboarding.import_scan import ScanRunner

    assert ScanRunner(_Mgr(key=True)).pick("p") == "orbital"
    # Round 3: keyless → the ready pinned worker, whether or not the wizard's
    # Skip was ever pressed (the wizard's own timing rides on defer_scan).
    assert ScanRunner(_Mgr(key=False, skipped=False)).pick("p") == "worker"
    assert ScanRunner(_Mgr(key=False, skipped=True)).pick("p") == "worker"
    assert ScanRunner(_Mgr(key=False, worker=None)).pick("p") is None
    # A key always wins over the worker.
    assert ScanRunner(_Mgr(key=True, skipped=True)).pick("p") == "orbital"


def test_scan_agent_names_the_product(store, tmp_path):
    from agent_os.onboarding.import_scan import scan_agent

    _imported(store, tmp_path, "a")
    assert scan_agent(_Mgr(), store) == {"slug": "claude-code", "name": "Claude Code"}
    # No unscanned project: the worker a new import would get.
    empty = ProjectStore(str(tmp_path / "empty"))
    assert scan_agent(_Mgr(), empty) == {"slug": "claude-code", "name": "Claude Code"}
    assert scan_agent(_Mgr(worker=None), store) is None
    assert scan_agent(None, store) is None


def test_settings_get_carries_the_scan_agent(store, tmp_path, monkeypatch):
    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import settings as settings_routes

    _imported(store, tmp_path, "a")
    fake = MagicMock()
    fake.get_masked = MagicMock(return_value={"llm": {"api_key_set": False}})
    settings_routes.configure(fake, project_store=store)
    monkeypatch.setattr(agents_v2, "_agent_manager", _Mgr())
    app = FastAPI()
    app.include_router(settings_routes.router)
    body = TestClient(app).get("/api/v2/settings").json()
    assert body["unscanned_imports"] == 1
    assert body["import_scan_agent"] == {"slug": "claude-code", "name": "Claude Code"}


def test_wizard_skip_kicks(tmp_path, monkeypatch, kicks):
    client = _settings_app(tmp_path, monkeypatch)
    client.put("/api/v2/settings", json={"sub_agents_hidden": []})
    assert kicks == []
    r = client.put("/api/v2/settings", json={"setup_api_key_skipped": True})
    assert r.status_code == 200, r.text
    assert kicks == [("key_skipped", True)]


def test_worker_guess_prompt_is_the_manager_text_behind_the_preamble(tmp_path):
    from agent_os.agent import onboarding_prompts as op
    from agent_os.daemon_v2.worker_onboarding import build_instruction

    ws = str(tmp_path)
    skel = "[WORKSPACE SKELETON] 1 files, ~5 bytes shown.\nREADME.md\t5"
    text = build_instruction("claude-code", {"workspace": ws}, "scan", skel,
                             guess_lang="zh", multi_agent=True)
    assert text.startswith(op.worker_preamble("claude-code", ws, scan=True))
    assert op.guess_first_instruction(ws, lang="zh", multi_agent=True) in text
    assert text.rstrip().endswith("README.md\t5")
    flat = _flat(text)
    assert "这样理解对吗" not in flat and op.SHARED_MEMORY_LINE_ZH in flat
    assert "Write NOTHING until the user confirms" not in text
    # Round 5: the worker gets the same next-steps rule (it writes the file
    # itself, so the card reads it the same way).
    assert "`## Next steps` heading (exactly that heading" in flat
    one = build_instruction("claude-code", {"workspace": ws}, "scan", skel,
                            guess_lang="en", multi_agent=False)
    assert op.SHARED_MEMORY_LINE_EN not in _flat(one)
    # No guess language: the card's propose-first worker scan, unchanged.
    assert build_instruction("claude-code", {"workspace": ws}, "scan", skel) == \
        op.worker_cold_start_instruction("claude-code", ws, skel)


def test_scan_title_follows_the_import_language(monkeypatch):
    from agent_os.onboarding import deferred_seed

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")
    assert import_scan.scan_title(import_scan.guess_lang({import_scan.LANG_FIELD: "zh"})) == "项目梳理"
    assert import_scan.scan_title(import_scan.guess_lang({})) == "Project overview"


def test_only_noise_reads_worker_sessions():
    from agent_os.onboarding.import_scan import _only_noise

    assert _only_noise([{"role": "system", "source": "daemon", "content": "[Sub-agent] x could not start"}])
    assert _only_noise([{"role": "user", "source": "daemon", "_kind": "memory_record", "content": "x"}])
    assert not _only_noise([{"role": "assistant", "source": "sub_agent", "content": "这是一个…"}])
    assert not _only_noise([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Round 3: the scan session's title is final; every other session unchanged
# ---------------------------------------------------------------------------


def _session(tmp_path, origin="cold_start"):
    from agent_os.agent.session import Session
    return Session.new("s1", str(tmp_path), session_id="s1", origin=origin)


def test_scan_title_survives_the_users_first_reply(tmp_path):
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.session_index import derive_disk_entry

    s = _session(tmp_path)
    s.set_name("项目梳理")                      # what the scan's mint does
    s.append({"role": "system", "source": "daemon", "content": "[WORKSPACE SKELETON]"})
    s.append({"role": "assistant", "content": "这是一个解析库。这样理解对吗？"})
    s.append({"role": "user", "content": "对"})
    assert s.name == "项目梳理"
    assert Session.load(s.filepath).name == "项目梳理"
    assert derive_disk_entry(s.filepath, "s1")["name"] == "项目梳理"
    s.set_name("我的解析库")                    # a manual rename still works
    assert Session.load(s.filepath).name == "我的解析库"


def test_every_other_session_still_titles_from_its_first_message(tmp_path):
    """Byte-identical auto-title for non-scan sessions: nothing set at mint,
    the first user message names it, later ones don't."""
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.session_index import derive_disk_entry

    s = _session(tmp_path, origin="chat")
    assert s.name is None
    s.append({"role": "user", "content": "帮我把 README 改短一点"})
    s.append({"role": "user", "content": "second"})
    assert s.name == "帮我把 README 改短一点"
    assert Session.load(s.filepath).name == "帮我把 README 改短一点"
    assert derive_disk_entry(s.filepath, "s1")["name"] == "帮我把 README 改短一点"


def test_the_card_cold_start_scan_is_not_titled(tmp_path, monkeypatch):
    """Only the import scan names its session; the card's scan (and every
    other start) passes no title."""
    import inspect

    from agent_os.api.routes import agents_v2
    src = inspect.getsource(agents_v2.cold_start_scan)
    assert "title" not in src


def test_import_seeds_the_checked_map_empty_so_the_scan_reply_is_unread(tmp_path, monkeypatch, kicks):
    """§4 (a), found live in round 2: the checked store seeds a project on its
    first list, counting replies then present as checked. If nobody listed an
    imported project before its background scan replied, that reply was
    swallowed. The import seeds the map (empty) at creation."""
    from datetime import datetime, timezone

    from agent_os.api.routes import agents_v2, workbench
    from agent_os.api.routes import onboarding as onboarding_routes
    from agent_os.daemon_v2.checked_store import CheckedStore

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    store = ProjectStore(str(tmp_path / "data"))
    agents_v2.configure(project_store=store, agent_manager=MagicMock(), ws_manager=MagicMock())
    checked = CheckedStore.in_dir(str(tmp_path / "data"))
    monkeypatch.setattr(workbench, "_checked_store", checked)
    scanner = MagicMock()
    scanner.analysis = MagicMock(return_value=None)
    scanner.is_ready = MagicMock(return_value=False)
    onboarding_routes.configure(project_store=store, scanner=scanner)
    ws = tmp_path / "folder"
    ws.mkdir()
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    pid = TestClient(app).post("/api/v2/onboarding/import",
                               json={"folders": [str(ws)]}).json()["created"][0]["project_id"]
    assert checked.project(pid) == {}
    # Later the scan replies; the first list comes after it.
    reply_at = datetime.now(timezone.utc).isoformat()
    rows = [{"session_id": "s1", "session_uuid": "s1", "status": "idle",
             "last_reply_at": reply_at, "last_user_at": None}]
    checked.annotate(pid, rows)
    assert rows[0]["checked_reply_at"] is None
    assert workbench.is_unchecked(rows[0], datetime.now(timezone.utc))
    onboarding_routes.configure()


# ---------------------------------------------------------------------------
# Round 2: the opt-out switch inside the sign (per batch, never lasting)
# ---------------------------------------------------------------------------


def test_turn_off_waiting_covers_pending_and_retry_left_only(store, tmp_path):
    from agent_os.onboarding.import_scan import OFF, turn_off_waiting

    pend, _ = _imported(store, tmp_path, "pend")
    retry, _ = _imported(store, tmp_path, "retry", state=FAILED, attempts=1)
    gave_up, _ = _imported(store, tmp_path, "gave_up", state=FAILED, attempts=2)
    running, _ = _imported(store, tmp_path, "running", state=RUNNING)
    done, _ = _imported(store, tmp_path, "done", state=DONE)
    legacy, _ = _imported(store, tmp_path, "legacy", state=None)
    assert sorted(turn_off_waiting(store)) == sorted([pend, retry])
    st = lambda p: store.get_project(p).get(import_scan.STATE_FIELD)
    assert st(pend) == OFF and st(retry) == OFF
    assert st(gave_up) == FAILED and st(running) == RUNNING and st(done) == DONE
    assert st(legacy) is None
    assert unscanned_count(store) == 0


def test_off_projects_are_never_scanned_by_a_later_key_event(store, tmp_path):
    from agent_os.onboarding.import_scan import OFF

    a, _ = _imported(store, tmp_path, "a", state=OFF)

    async def go():
        runner = FakeRunner(store)
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        q.resume()
        await q.idle()
        return runner

    assert _run(go()).started == []
    assert store.get_project(a)[import_scan.STATE_FIELD] == OFF


def test_a_queued_project_turned_off_before_its_turn_is_not_started(store, tmp_path):
    from agent_os.onboarding.import_scan import OFF, turn_off_waiting

    a, _ = _imported(store, tmp_path, "a", activity="2026-11-03T00:00:00Z")
    b, _ = _imported(store, tmp_path, "b", activity="2026-11-02T00:00:00Z")
    c, _ = _imported(store, tmp_path, "c", activity="2026-11-01T00:00:00Z")

    async def go():
        runner = FakeRunner(store, hold=True)
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        for _ in range(30):
            await asyncio.sleep(0)
        # a and b take the two slots together; their start order is not
        # deterministic (it differs on Windows), only that c waits.
        assert sorted(runner.started) == sorted([a, b])
        turn_off_waiting(store)                  # running a/b untouched, c off
        runner.release.set()
        await q.idle()
        return runner

    runner = _run(go())
    assert sorted(runner.started) == sorted([a, b])
    assert store.get_project(c)[import_scan.STATE_FIELD] == OFF
    assert store.get_project(a)[import_scan.STATE_FIELD] == DONE


def test_import_with_the_switch_off_is_a_pre160_import(tmp_path, monkeypatch, kicks):
    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import onboarding as onboarding_routes

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    store = ProjectStore(str(tmp_path / "data"))
    agents_v2.configure(project_store=store, agent_manager=MagicMock(), ws_manager=MagicMock())
    scanner = MagicMock()
    scanner.analysis = MagicMock(return_value=None)
    scanner.is_ready = MagicMock(return_value=False)
    onboarding_routes.configure(project_store=store, scanner=scanner)
    ws = tmp_path / "folder"
    ws.mkdir()
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    r = TestClient(app).post("/api/v2/onboarding/import",
                             json={"folders": [str(ws)], "import_scan": False})
    p = store.get_project(r.json()["created"][0]["project_id"])
    assert import_scan.STATE_FIELD not in p and p["imported"] is True
    assert kicks == []
    assert unscanned_count(store) == 0
    onboarding_routes.configure()


def test_card_save_and_skip_with_the_switch_off(tmp_path, monkeypatch, kicks):
    from agent_os.api.routes import settings as settings_routes
    from agent_os.onboarding.import_scan import OFF

    client = _settings_app(tmp_path, monkeypatch)
    store = settings_routes._project_store
    a, _ = _imported(store, tmp_path, "a")
    # Default: the switch is on; nothing changes.
    client.post("/api/v2/settings/cards", json={"provider": "opencode-go", "model": "m", "api_key": "k"})
    assert store.get_project(a)[import_scan.STATE_FIELD] == PENDING
    client.post("/api/v2/settings/cards", json={"provider": "opencode-go", "model": "m",
                                                 "api_key": "k", "import_scan": False})
    assert store.get_project(a)[import_scan.STATE_FIELD] == OFF
    b, _ = _imported(store, tmp_path, "b")
    client.put("/api/v2/settings/cards/c1", json={"api_key": "k2", "import_scan": False})
    assert store.get_project(b)[import_scan.STATE_FIELD] == OFF
    c, _ = _imported(store, tmp_path, "c")
    client.put("/api/v2/settings", json={"setup_api_key_skipped": True, "import_scan": False})
    assert store.get_project(c)[import_scan.STATE_FIELD] == OFF
    # The action's own trigger still fires; it simply finds nothing.
    assert [k for k, _ in kicks] == ["card_created", "card_created", "card_updated", "key_skipped"]
    # import_scan is never written onto the card.
    for call in settings_routes._settings_store.update_card.call_args_list:
        assert "import_scan" not in call.kwargs



def test_the_wizards_import_stamps_but_defers_the_trigger(tmp_path, monkeypatch, kicks):
    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import onboarding as onboarding_routes

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    store = ProjectStore(str(tmp_path / "data"))
    agents_v2.configure(project_store=store, agent_manager=MagicMock(), ws_manager=MagicMock())
    scanner = MagicMock()
    scanner.analysis = MagicMock(return_value=None)
    scanner.is_ready = MagicMock(return_value=False)
    onboarding_routes.configure(project_store=store, scanner=scanner)
    ws = tmp_path / "folder"
    ws.mkdir()
    app = FastAPI()
    app.include_router(onboarding_routes.router)
    r = TestClient(app).post("/api/v2/onboarding/import",
                             json={"folders": [str(ws)], "defer_scan": True})
    p = store.get_project(r.json()["created"][0]["project_id"])
    assert p[import_scan.STATE_FIELD] == import_scan.DEFERRED   # waits for the key step
    assert kicks == []                             # ...which is the trigger
    onboarding_routes.configure()


# ---------------------------------------------------------------------------
# Round 4: the daemon-posted confirm card
# ---------------------------------------------------------------------------


def test_the_confirm_card_is_posted_on_success_only(store, tmp_path):
    a, _ = _imported(store, tmp_path, "a")
    b, _ = _imported(store, tmp_path, "b")

    async def go():
        runner = FakeRunner(store, outcomes={b: "fail"})
        q = ImportScanQueue(store, runner)
        q.kick("card_created")
        await q.idle()
        return runner

    runner = _run(go())
    assert len(runner.confirmed) == 1 and runner.confirmed[0].startswith(f"sess-{a}")


def test_deferred_wizard_imports_wait_for_a_trigger_not_a_restart(store, tmp_path):
    from agent_os.onboarding.import_scan import DEFERRED

    w, _ = _imported(store, tmp_path, "wiz", state=DEFERRED)
    p, _ = _imported(store, tmp_path, "pend")

    async def go():
        runner = FakeRunner(store)
        q = ImportScanQueue(store, runner)
        q.resume()                       # daemon start
        await q.idle()
        first = list(runner.started)
        q.kick("key_skipped")            # the wizard's Skip
        await q.idle()
        return first, runner

    first, runner = _run(go())
    assert first == [p]                  # the deferred one did not start
    assert w in runner.started           # the key step's action started it
    assert unscanned_count(ProjectStore(str(tmp_path / "data"))) == 0


def _confirm_env(tmp_path, *, keyed, state=None, lang="zh"):
    """A real AgentManager behind agents_v2 (the keyless harness), one
    imported project whose scan wrote goals, and the confirm card posted.
    ``state``: the PROJECT_STATE.md the scan wrote (round 5: its Next steps
    section feeds the card)."""
    import asyncio as _asyncio
    import json as _json

    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import workbench as wb
    from tests.unit.keyless_harness import build_client, create_project

    tc, deps = build_client(tmp_path, keyed=keyed)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("# lib\n")
    pid = create_project(tc, ws)["project_id"]
    deps["project_store"].update_project(pid, {import_scan.LANG_FIELD: lang})
    _write_goals(ws)
    if state is not None:
        (ws / "orbital" / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
    sid = "scan_sess_1"
    # The scan session exists on disk (its first row).
    from agent_os.agent.session import Session
    s = Session.new(sid, str(ws), session_id=sid, origin="cold_start")
    s.append({"role": "assistant", "content": "这是一个解析库。"})
    ask_id = _asyncio.run(agents_v2.post_scan_confirm(pid, sid))
    wb.configure(deps["project_store"], deps["agent_manager"], MagicMock(),
                 ws_manager=deps["ws_manager"])
    rows = [_json.loads(l) for l in open(s.filepath, encoding="utf-8")]
    return tc, deps, pid, sid, ws, ask_id, rows


def test_post_scan_confirm_opens_the_ask_and_its_session_row(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, rows = _confirm_env(tmp_path, keyed=True)
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.text == "这样理解对吗？" and ask.options == ("对", "需要改") and ask.is_open
    card = rows[-1]
    assert card["_kind"] == "ask_card" and card["_meta"]["event"] == "ask_opened"
    assert card["_meta"]["ask_id"] == ask_id and card["_meta"]["import_scan_confirm"] is True
    assert deps["project_store"].get_project(pid)[import_scan.ASK_FIELD] == {
        "id": ask_id, "session_id": sid, "yes": "对", "change": "需要改"}


def _answer(tc, pid, sid, ask_id, words, target=None):
    body = {"content": f'[Answer to ask {ask_id}, already closed: "这样理解对吗？"]\n{words}',
            "session_id": sid, "ask_id": ask_id, "ask_answer": words, "nonce": "n1"}
    if target:
        body["target"] = target
    return tc.post(f"/api/v2/agents/{pid}/inject", json=body)


def test_yes_closes_the_ask_records_the_row_and_dispatches_nothing(tmp_path, monkeypatch):
    import json as _json

    from agent_os.agent import asks
    from agent_os.agent.session import Session

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    mgr = deps["agent_manager"]
    calls = []

    async def spy(*a, **k):
        calls.append(a)
        return "delivered"

    monkeypatch.setattr(mgr, "inject_message", spy)
    monkeypatch.setattr(mgr, "start_agent", spy)
    s = Session.load(str(ws / "orbital" / "sessions" / f"{sid}.jsonl"))
    s.set_onboarding_gate(True)                       # the scan armed it
    r = _answer(tc, pid, sid, ask_id, "对")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "recorded"
    assert calls == []                                 # no run, no LLM call
    deps["sub_agent_manager"].send.assert_not_awaited()
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open
    rows = [_json.loads(l) for l in open(s.filepath, encoding="utf-8")]
    user = [r for r in rows if r.get("role") == "user"]
    assert user[-1]["content"].endswith("\n对") and "<orbital_turn_note>" not in user[-1]["content"]
    # The gate stays armed: the user's NEXT real message is the go-ahead.
    assert Session.load(s.filepath).onboarding_gate is True


def test_yes_on_a_worker_session_dispatches_nothing_either(tmp_path):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=False)
    r = _answer(tc, pid, sid, ask_id, "对", target="claude-code")
    assert r.status_code == 200 and r.json()["status"] == "recorded"
    deps["sub_agent_manager"].send.assert_not_awaited()


def test_needs_changes_runs_one_turn_with_the_note_in_that_message_only(tmp_path, monkeypatch):
    """The 需要改 option runs one guarded turn (round 4). Round 5: only that
    option carries the note — a typed reply is a normal message (below)."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(deps["agent_manager"], "inject_message", spy)
    r = _answer(tc, pid, sid, ask_id, "需要改")
    assert r.status_code == 200, r.text
    assert len(sent) == 1
    assert sent[0].startswith(f'[Answer to ask {ask_id}, already closed:')
    assert "需要改\n\n<orbital_turn_note>" in sent[0]
    assert "只改 orbital/ 下的文件" in sent[0] and sent[0].endswith("</orbital_turn_note>")


def test_a_typed_reply_is_a_normal_message(tmp_path, monkeypatch):
    """Round 5: the daemon cannot tell a typed correction from a typed "做第
    2 步", so typed text goes through as the user's own message (no note, so
    the gate opens on it as on any reply)."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(deps["agent_manager"], "inject_message", spy)
    r = _answer(tc, pid, sid, ask_id, "不对，这是给团队用的配置工具")
    assert r.status_code == 200, r.text
    assert len(sent) == 1
    assert sent[0].endswith("\n不对，这是给团队用的配置工具")
    assert "<orbital_turn_note>" not in sent[0] and "<orbital_picked_step>" not in sent[0]


def test_a_worker_needs_changes_sends_the_note_but_stores_only_the_words(tmp_path):
    import json as _json

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=False)
    r = _answer(tc, pid, sid, ask_id, "需要改", target="claude-code")
    assert r.status_code == 200, r.text
    args, _k = deps["sub_agent_manager"].send.await_args
    assert args[2].endswith("</orbital_turn_note>") and "需要改" in args[2]
    rows = [_json.loads(l) for l in open(ws / "orbital" / "sessions" / f"{sid}.jsonl", encoding="utf-8")]
    user = [r for r in rows if r.get("role") == "user"]
    assert user[-1]["content"].endswith("需要改")
    assert "<orbital_turn_note>" not in user[-1]["content"]


def test_the_note_reaches_its_turn_only_and_the_gate_holds_through_it(tmp_path):
    from agent_os.agent.context import ContextManager, strip_turn_notes
    from agent_os.agent.prompt_builder import PromptContext
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.models import Autonomy

    class _B:
        def build(self, context):
            return ("sys", "semi", "dyn")

    ctx = PromptContext(workspace=str(tmp_path), model="m", autonomy=Autonomy.HANDS_OFF,
                        enabled_agents=[], tool_names=["read"], os_type="macos",
                        datetime_now="2026-11-07T00:00:00", project_name="p",
                        project_instructions="", agent_name="p", project_id="x")
    s = Session.new("g1", str(tmp_path), session_id="g1", origin="cold_start")
    s.append({"role": "assistant", "content": "summary"})
    s.set_onboarding_gate(True)
    note = import_scan.turn_note_block("zh")
    s.append({"role": "user", "source": "user", "content": "不对，改成连载\n\n" + note})
    assert s.onboarding_gate is True                 # held through the guarded turn
    mgr = ContextManager(s, _B(), ctx, model_context_limit=200_000)
    sent = mgr.prepare()
    assert any("<orbital_turn_note>" in str(m.get("content")) for m in sent)
    s.append({"role": "assistant", "content": "已改好。"})
    s.append({"role": "user", "source": "user", "content": "好，接下来写第三篇"})
    assert s.onboarding_gate is False                # the next real message opens it
    later = mgr.prepare()
    assert not any("<orbital_turn_note>" in str(m.get("content")) for m in later)
    assert "不对，改成连载" in " ".join(str(m.get("content")) for m in later)
    # The session row itself keeps the note (restart-safe), only the copy
    # bound for the provider drops it.
    assert any("<orbital_turn_note>" in str(m.get("content")) for m in s.get_messages())
    assert strip_turn_notes([{"role": "user", "content": "a" + note}])[0]["content"].endswith(note)


def test_an_open_scan_ask_is_still_handled_after_a_restart(tmp_path, monkeypatch):
    """The record lives on the project and the ask in ASKS.md: a fresh daemon
    (new routes, new manager) still answers 对 without a run."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    from agent_os.daemon_v2.project_store import ProjectStore as _PS
    from tests.unit.keyless_harness import build_client
    # "Restart": a new client over the same data dir and workspace.
    tc2, deps2 = build_client(tmp_path, keyed=True)
    assert _PS(str(tmp_path / "orbital-data")).get_project(pid)[import_scan.ASK_FIELD]["id"] == ask_id
    called = []

    async def spy(*a, **k):
        called.append(a)
        return "delivered"

    monkeypatch.setattr(deps2["agent_manager"], "inject_message", spy)
    r = _answer(tc2, pid, sid, ask_id, "对")
    assert r.status_code == 200 and r.json()["status"] == "recorded" and called == []


# ---------------------------------------------------------------------------
# Round 5: the scan proposes next steps; the confirm card offers step 1
# ---------------------------------------------------------------------------

_STATE_WITH_STEPS = (
    "<!--format PROJECT_STATE is what is true NOW-->\n"
    "# Project State\n\n"
    "## Current focus\n- tinyparse v0.3.0\n\n"
    "## Next steps\n"
    "- Implement `dumps()` so it round-trips with `loads()`\n"
    "<!--mem id=a1 created=2026-11-07-->\n"
    "  detail line under the bullet, not a step\n"
    "- Add line numbers to `ParseError`\n"
    "- Publish 0.4.0 to PyPI\n"
    "- A fourth one past the cap\n\n"
    "## Notes\n- not a step\n"
)


def _rows(ws, sid):
    import json as _json
    return [_json.loads(l) for l in open(ws / "orbital" / "sessions" / f"{sid}.jsonl",
                                         encoding="utf-8")]


def test_next_steps_reads_the_state_files_section(tmp_path):
    from agent_os.onboarding.import_scan import next_steps

    orb = tmp_path / "orbital"
    orb.mkdir()
    assert next_steps(str(tmp_path)) == []                     # no state file
    (orb / "PROJECT_STATE.md").write_text(_STATE_WITH_STEPS, encoding="utf-8")
    assert next_steps(str(tmp_path)) == [
        "Implement `dumps()` so it round-trips with `loads()`",
        "Add line numbers to `ParseError`",
        "Publish 0.4.0 to PyPI",
    ]
    # The heading as earlier scans wrote it, in either language; numbered items.
    for heading in ("## Likely next steps (guess)", "## Next steps (best guess)",
                    "## 下一步", "### Next Steps"):
        (orb / "PROJECT_STATE.md").write_text(
            f"# S\n\n{heading}\n1. 续写第二篇\n2) 动笔《散步》\n", encoding="utf-8")
        assert next_steps(str(tmp_path)) == ["续写第二篇", "动笔《散步》"], heading
    # Nothing clearly next: no section, or only a placeholder.
    for body in ("# S\n\n## Current focus\n- x\n", "# S\n\n## Next steps\n- (none yet)\n",
                 "# S\n\n## Next steps\n- 无\n"):
        (orb / "PROJECT_STATE.md").write_text(body, encoding="utf-8")
        assert next_steps(str(tmp_path)) == [], body


def test_the_card_with_steps_offers_step_one(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, rows = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.text == "这样理解对吗？" and ask.is_open
    assert ask.options == ("对，做第 1 步", "对，先不做", "需要改")
    assert ask.detail == (
        "1. Implement `dumps()` so it round-trips with `loads()`\n"
        "2. Add line numbers to `ParseError`\n"
        "3. Publish 0.4.0 to PyPI")
    assert rows[-1]["_meta"]["import_scan_confirm"] is True
    rec = deps["project_store"].get_project(pid)[import_scan.ASK_FIELD]
    assert rec["id"] == ask_id and rec["yes"] == "对，先不做"
    assert rec["go"] == "对，做第 1 步" and rec["change"] == "需要改"
    assert rec["steps"][0] == "Implement `dumps()` so it round-trips with `loads()`"
    # English import.
    tc2, deps2, pid2, sid2, ws2, ask2, _ = _confirm_env(
        tmp_path / "en", keyed=True, state=_STATE_WITH_STEPS, lang="en")
    (a2,) = [a for a in asks.read_asks(str(ws2 / "orbital")) if a.id == ask2]
    assert a2.text == "Did I get this right?"
    assert a2.options == ("Yes, do step 1", "Yes, not now", "Needs changes")


def test_the_card_without_steps_is_unchanged(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state="# S\n\n## Current focus\n- x\n")
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.options == ("对", "需要改") and ask.detail == ""
    rec = deps["project_store"].get_project(pid)[import_scan.ASK_FIELD]
    assert rec["yes"] == "对" and "go" not in rec and "steps" not in rec


def _spy_inject(monkeypatch, mgr):
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(mgr, "inject_message", spy)
    monkeypatch.setattr(mgr, "start_agent", spy)
    return sent


def test_do_step_one_runs_once_with_an_explicit_approval(tmp_path, monkeypatch):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    r = _answer(tc, pid, sid, ask_id, "对，做第 1 步")
    assert r.status_code == 200, r.text
    assert len(sent) == 1                                   # exactly one run
    msg = sent[0]
    # The user's tap text stays the visible answer; the approval names step 1.
    assert msg.startswith(f'[Answer to ask {ask_id}, already closed: "这样理解对吗？"]\n对，做第 1 步')
    assert ("<orbital_picked_step>对，按第 1 步开始：Implement `dumps()` so it "
            "round-trips with `loads()`</orbital_picked_step>") in msg
    assert "<orbital_turn_note>" not in msg                 # not a guarded turn
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open


def test_do_step_one_opens_the_gate_and_the_approval_stays_in_context(tmp_path):
    from agent_os.agent.context import strip_turn_notes
    from agent_os.agent.session import Session

    s = Session.new("g2", str(tmp_path), session_id="g2", origin="cold_start")
    s.append({"role": "assistant", "content": "summary\n1. a\n2. b"})
    s.set_onboarding_gate(True)
    block = import_scan.picked_step_block("zh", "a")
    s.append({"role": "user", "source": "user", "content": "对，做第 1 步\n\n" + block})
    assert s.onboarding_gate is False                       # the user's go
    later = strip_turn_notes(s.get_messages() + [{"role": "user", "content": "next"}])
    assert block in later[-2]["content"]                    # kept on later turns


def test_do_step_one_on_a_worker_sends_the_approval_stores_the_tap(tmp_path):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=False, state=_STATE_WITH_STEPS)
    r = _answer(tc, pid, sid, ask_id, "对，做第 1 步", target="claude-code")
    assert r.status_code == 200, r.text
    assert deps["sub_agent_manager"].send.await_count == 1
    args, _k = deps["sub_agent_manager"].send.await_args
    assert "<orbital_picked_step>对，按第 1 步开始：Implement" in args[2]
    user = [r for r in _rows(ws, sid) if r.get("role") == "user"]
    assert user[-1]["content"].endswith("\n对，做第 1 步")


def test_not_now_is_zero_call_and_keeps_the_gate(tmp_path, monkeypatch):
    from agent_os.agent import asks
    from agent_os.agent.session import Session

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    s = Session.load(str(ws / "orbital" / "sessions" / f"{sid}.jsonl"))
    s.set_onboarding_gate(True)
    r = _answer(tc, pid, sid, ask_id, "对，先不做")
    assert r.status_code == 200 and r.json()["status"] == "recorded"
    assert sent == []
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open
    assert Session.load(s.filepath).onboarding_gate is True


@pytest.mark.parametrize("typed", ["对", "对。", " 对！", "对的", "Yes", "yes."])
def test_a_typed_yes_is_still_zero_call(tmp_path, monkeypatch, typed):
    """The round-3 bug: a bare 对 must never become the go-ahead, on either
    card, typed or tapped."""
    for state in (_STATE_WITH_STEPS, None):
        root = tmp_path / ("steps" if state else "plain")
        tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(root, keyed=True, state=state)
        sent = _spy_inject(monkeypatch, deps["agent_manager"])
        r = _answer(tc, pid, sid, ask_id, typed)
        assert r.status_code == 200 and r.json()["status"] == "recorded", (typed, state)
        assert sent == []


def test_typed_do_step_two_is_a_normal_run(tmp_path, monkeypatch):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    r = _answer(tc, pid, sid, ask_id, "做第 2 步")
    assert r.status_code == 200, r.text
    assert len(sent) == 1 and sent[0].endswith("\n做第 2 步")
    assert "<orbital_turn_note>" not in sent[0] and "<orbital_picked_step>" not in sent[0]


def test_needs_changes_on_the_steps_card_is_the_guarded_turn(tmp_path, monkeypatch):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    _answer(tc, pid, sid, ask_id, "需要改")
    assert len(sent) == 1 and "<orbital_turn_note>" in sent[0]
    assert "<orbital_picked_step>" not in sent[0]


def test_a_round_four_record_still_answers(tmp_path, monkeypatch):
    """An ask posted by the round-4 daemon (record without go/change): 对 is
    zero-call, 需要改 the guarded turn."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    deps["project_store"].update_project(pid, {import_scan.ASK_FIELD: {
        "id": ask_id, "session_id": sid, "yes": "对"}})
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    _answer(tc, pid, sid, ask_id, "需要改")
    assert len(sent) == 1 and "<orbital_turn_note>" in sent[0]


# ---------------------------------------------------------------------------
# Spec 155 merge: no project slot any more
# ---------------------------------------------------------------------------


class _BusyRunner(FakeRunner):
    def __init__(self, store, message, **kw):
        super().__init__(store, **kw)
        self.message = message

    async def start(self, pid, who):
        raise RuntimeError(self.message)


def test_a_busy_start_goes_back_to_pending_without_an_attempt(store, tmp_path):
    """A start refused because its runner is already busy (a keyless worker
    mid-turn) is not a failure: back to pending for the next trigger."""
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        q = ImportScanQueue(store, _BusyRunner(store, "Error: worker is already running a task"))
        q.kick("card_created")
        await q.idle()

    _run(go())
    p = store.get_project(a)
    assert p[import_scan.STATE_FIELD] == PENDING
    assert not p.get(import_scan.ATTEMPTS_FIELD)


def test_the_retired_slot_refusal_is_an_ordinary_failure(store, tmp_path):
    """Spec 155 lifted the project slot: the scan mints its own session and
    runs beside the user's, so no "Slot held by session" refusal exists. If
    such a text ever came back it must not park the scan in pending forever."""
    a, _ = _imported(store, tmp_path, "a")

    async def go():
        q = ImportScanQueue(store, _BusyRunner(store, "Slot held by session s1"))
        q.kick("card_created")
        await q.idle()

    _run(go())
    p = store.get_project(a)
    assert p[import_scan.STATE_FIELD] == FAILED and p[import_scan.ATTEMPTS_FIELD] == 1

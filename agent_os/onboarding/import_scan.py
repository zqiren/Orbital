# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Imported projects: Orbital guesses first, the user confirms (spec 160 §5.2).

``POST /onboarding/import`` stamps every project it creates
``import_scan: "pending"``. With a key, a daemon-side queue reads each such
project once through the cold-start mint path, in its guess-first flavour:
the manager writes its best guess at goals, ``PROJECT_STATE.md`` and
``INDEX.md`` and leaves one unread reply ending with "is this right?".

* **Only stamped projects.** A project with no ``import_scan`` field (every
  project imported before spec 160, every project created by hand) is never
  scanned, whatever event arrives (open decision 1: only new imports).
* **Decidable triggers only** (:func:`kick`): the import finishing, a card
  created or re-keyed, the default card changed, the legacy key saved. Never
  a startup sweep: :meth:`ImportScanQueue.resume` on startup picks up only
  projects already ``pending`` (or ``running`` when the daemon died mid-scan).
* **2 at a time, newest activity first** (``import_scan_activity``, the
  folder's last session from the history scan; unknown sorts last, then
  creation order).
* **Skip** when ``project_goals.md`` exists by the time a project's turn
  comes (the user got there first): ``skipped``.
* **Opt-out** (round 2): the sign's switch, per batch. Off on the sidebar
  import → those projects are never stamped (pre-160 behaviour). Off on a
  key Save or the wizard's Skip → everything still waiting turns ``off``.
* **Failure** (the run ends without goals, or the start fails): ``failed``,
  retried once on the next trigger, then left to today's first-use
  behaviour. A failed run's session is removed when nothing in it is the
  user's or a real reply, so a provider error leaves no litter.
* **Runner, chosen per run** (round 3): the manager when it has a key; else
  the project's effective pinned worker when one is ready, through the
  existing keyless path; else nobody, and the project stays ``pending``
  until a key or a worker arrives. The wizard imports before its key step,
  so its import is marked ``defer_scan`` and stamps without triggering: the
  key step's Save or Skip is the trigger there. One queue, the same 2 slots,
  for both runners.
* The scan session is titled 项目梳理 / Project overview (the import's UI
  language), set at mint like a rename: the user's first reply does not
  retitle it; a manual rename still does.

The fields are additive: an older daemon ignores them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATE_FIELD = "import_scan"
AT_FIELD = "import_scan_at"
ATTEMPTS_FIELD = "import_scan_attempts"
ACTIVITY_FIELD = "import_scan_activity"
# The UI language at import (``zh`` | ``en``), for the guess reply.
LANG_FIELD = "import_scan_lang"
LANGS = ("zh", "en")

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
# Round 2 opt-out: the user switched the sign off for the batch behind a
# button. Terminal, like done: the queue never picks it, nothing re-arms it.
OFF = "off"
# Round 4: a wizard import (``defer_scan``) waits for the key step's Save or
# Skip — a daemon restart in between must not start it. Picked by a real
# trigger, never by the startup resume.
DEFERRED = "deferred"

# One try plus the retry on the next trigger.
MAX_ATTEMPTS = 2
CONCURRENCY = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pending_fields(activity: str | None = None, *, lang: str | None = None,
                   deferred: bool = False) -> dict:
    """The stamp ``POST /onboarding/import`` writes on a project it creates
    (``deferred``: the wizard's, waiting for its key step)."""
    out = {STATE_FIELD: DEFERRED if deferred else PENDING, AT_FIELD: _now(),
           ACTIVITY_FIELD: activity}
    if lang in LANGS:
        out[LANG_FIELD] = lang
    return out


SCAN_TITLES = {"zh": "项目梳理", "en": "Project overview"}

# Round 4: the scan's confirmation is a daemon-posted ask card, never the
# model's own question, so "对" can never be read as a go-ahead.
# ``(question, yes, needs-changes)``; the yes option is answered by the daemon
# alone (no run), anything else runs one guarded turn.
CONFIRM_ASK = {
    "zh": ("这样理解对吗？", "对", "需要改"),
    "en": ("Did I get this right?", "Yes", "Needs changes"),
}
# Round 5: the scan recorded next steps (PROJECT_STATE's ``## Next steps``):
# the card lists them as its detail and offers step 1.
# ``(question, go, not-now, needs-changes)``: go runs step 1 (one run, the
# user's go), not-now is the zero-call close, needs-changes the guarded turn.
CONFIRM_ASK_STEPS = {
    "zh": ("这样理解对吗？", "对，做第 1 步", "对，先不做", "需要改"),
    "en": ("Did I get this right?", "Yes, do step 1", "Yes, not now", "Needs changes"),
}
# The words step 1's run carries after the user's tap (hidden from the
# bubble, kept in context): an explicit approval naming the step.
PICKED_STEP_OPEN = "<orbital_picked_step>"
PICKED_STEP_CLOSE = "</orbital_picked_step>"
PICKED_STEP = {
    "zh": "对，按第 1 步开始：{step}",
    "en": "Yes, go ahead with step 1: {step}",
}
# A bare yes typed (or tapped) on either card is always the zero-call close
# (the round-3 bug: a bare 对 became the go-ahead). Compared after
# ``_norm_answer``.
BARE_YES = frozenset({"对", "对的", "是的", "没错", "yes", "correct"})
MAX_STEPS = 3
# On the project: {"id": <ask id>, "session_id": <scan session>,
# "yes": <zero-call option>, "change": <needs-changes option>} plus, with
# steps, "go": <do-step-1 option> and "steps": [<text>, ...].
ASK_FIELD = "import_scan_ask"

# The one-turn instruction a non-yes answer carries (folded into that user
# message; ``context.strip_turn_notes`` hides it from every later turn and the
# chat never shows it).
TURN_NOTE_OPEN = "<orbital_turn_note>"
TURN_NOTE_CLOSE = "</orbital_turn_note>"
TURN_NOTES = {
    "zh": ("用户在回答你对这个项目的理解。如果是在纠正，只改 orbital/ 下的文件"
           "（目标、PROJECT_STATE.md、INDEX.md）让它们一致；如果只点了「需要改」"
           "却没说哪里不对，就问哪里不对。这一轮不要开始任何任务，也不要动 orbital/ 以外的文件。"),
    "en": ("The user is answering your understanding of this project. If they're "
           "correcting it, edit only the files under orbital/ (goals, PROJECT_STATE.md, "
           "INDEX.md) to match; if they only tapped Needs changes without saying what, "
           "ask what's wrong. Do not start any task or touch files outside orbital/ in "
           "this turn."),
}


def confirm_ask(lang: str) -> tuple[str, str, str]:
    return CONFIRM_ASK.get(lang, CONFIRM_ASK["en"])


def confirm_ask_steps(lang: str) -> tuple[str, str, str, str]:
    return CONFIRM_ASK_STEPS.get(lang, CONFIRM_ASK_STEPS["en"])


def picked_step_block(lang: str, step: str) -> str:
    text = PICKED_STEP.get(lang, PICKED_STEP["en"]).format(step=" ".join(step.split()))
    return f"{PICKED_STEP_OPEN}{text}{PICKED_STEP_CLOSE}"


_STEPS_HEADING_RE = re.compile(r"^#{2,6}\s+.*(next\s+steps?|下一步)", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s")
_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*\S)")
_PLACEHOLDER_RE = re.compile(r"^[(（].*[)）]$|^(none|无|暂无|n/a)[.。]?$", re.IGNORECASE)


def next_steps(workspace: str) -> list[str]:
    """Round 5: the next steps the scan recorded — the items under the first
    ``## Next steps`` (or ``下一步``) heading of ``orbital/PROJECT_STATE.md``,
    top-level bullets or numbered lines only (indented detail and the
    daemon's mem-comments are not steps), placeholders like "(none yet)"
    dropped, at most 3. ``[]`` when the file or the section is missing.
    A file the run wrote is the record: no chat text is parsed, and a
    worker runner writes it the same way."""
    if not workspace:
        return []
    from agent_os.agent.project_paths import ProjectPaths
    try:
        with open(ProjectPaths(workspace).project_state, encoding="utf-8",
                  errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    steps: list[str] = []
    inside = False
    for line in lines:
        if _HEADING_RE.match(line):
            if inside:
                break
            inside = bool(_STEPS_HEADING_RE.match(line))
            continue
        if not inside:
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        text = " ".join(m.group(1).split())
        if _PLACEHOLDER_RE.match(text):
            continue
        steps.append(text)
        if len(steps) >= MAX_STEPS:
            break
    return steps


def _norm_answer(text: str | None) -> str:
    return " ".join((text or "").split()).strip(" 。.!！~～,，、").casefold()


def classify_answer(rec: dict, words: str | None, lang: str) -> str:
    """What the answer to the scan's confirm card asks for:

    * ``"yes"`` — the zero-call close: the card's not-now / 对 option, or a
      bare yes typed on either card;
    * ``"go"`` — the do-step-1 option: one run, the user's go;
    * ``"change"`` — the 需要改 option: one guarded turn;
    * ``"message"`` — anything else typed: the user's own message, sent as
      is (the daemon can't tell a correction from "做第 2 步").

    Options are matched as exact text (normalised for spaces, closing
    punctuation and case), so a tap and the same words typed are one answer.
    A round-4 record (no ``change``) falls back to the language's label."""
    said = _norm_answer(words)
    if not said:
        return "message"
    if said == _norm_answer(rec.get("yes")) or said in BARE_YES:
        return "yes"
    if rec.get("go") and rec.get("steps") and said == _norm_answer(rec.get("go")):
        return "go"
    change = rec.get("change") or confirm_ask(lang)[2]
    if said == _norm_answer(change):
        return "change"
    return "message"


def turn_note_block(lang: str) -> str:
    return f"{TURN_NOTE_OPEN}{TURN_NOTES.get(lang, TURN_NOTES['en'])}{TURN_NOTE_CLOSE}"


def scan_ask_of(project: dict | None, ask_id: str | None) -> dict | None:
    """The project's open scan-confirm record when ``ask_id`` is it."""
    rec = (project or {}).get(ASK_FIELD)
    if isinstance(rec, dict) and ask_id and rec.get("id") == (ask_id or "").lower():
        return rec
    return None


def guess_lang(project: dict | None) -> str:
    """The reply language of a project's scan: the import's UI language,
    else the system's."""
    lang = (project or {}).get(LANG_FIELD)
    if lang in LANGS:
        return lang
    from agent_os.onboarding.deferred_seed import default_lang
    return default_lang()


def scan_title(lang: str) -> str:
    return SCAN_TITLES.get(lang, SCAN_TITLES["en"])


def guess_flavour(project: dict | None) -> str:
    """The ``cold_start`` value of a project's guess run: ``"guess"``, with
    the import's UI language when it was known (``"guess:zh"``)."""
    lang = (project or {}).get(LANG_FIELD)
    return f"guess:{lang}" if lang in LANGS else "guess"


def has_goals(workspace: str) -> bool:
    """The test the prompt uses: an empty goals file is still onboarding."""
    if not workspace:
        return False
    from agent_os.agent.project_paths import ProjectPaths
    try:
        with open(ProjectPaths(workspace).project_goals, encoding="utf-8",
                  errors="replace") as fh:
            return bool(fh.read().strip())
    except OSError:
        return False


def wants_scan(project: dict | None, *, include_failed: bool = True) -> bool:
    """Whether ``project``'s stamp still asks for a scan (goals aside).
    ``include_failed=False`` is the startup resume: neither a failed project
    (its retry waits for a trigger) nor a deferred wizard import."""
    if not isinstance(project, dict) or project.get("is_scratch"):
        return False
    state = project.get(STATE_FIELD)
    if state in (PENDING, RUNNING):
        return True
    if state == DEFERRED:
        return include_failed
    if state == FAILED and include_failed:
        return int(project.get(ATTEMPTS_FIELD) or 0) < MAX_ATTEMPTS
    return False


def turn_off_waiting(project_store) -> list[str]:
    """The sign's switch was off when the user pressed its button: every
    project still waiting for its read (``pending``, or ``failed`` with the
    retry left) becomes ``off`` in that same action, so the trigger the
    action fires finds nothing. A scan already ``running`` is left alone (the
    session's own stop handles it); queued-but-not-started ones are
    ``pending`` and so are covered. Returns the ids turned off."""
    if project_store is None:
        return []
    out = []
    try:
        projects = list(project_store.list_projects())
    except Exception:
        logger.warning("import scan: could not list projects", exc_info=True)
        return []
    for p in projects:
        if p.get(STATE_FIELD) == RUNNING or not wants_scan(p):
            continue
        pid = p.get("project_id")
        try:
            project_store.update_project(pid, {STATE_FIELD: OFF, AT_FIELD: _now()})
            out.append(pid)
        except Exception:
            logger.warning("import scan: could not turn off %s", pid, exc_info=True)
    if out:
        logger.info("import scan: switched off for %d project(s): %s", len(out), ", ".join(out))
    return out


def unscanned_count(project_store) -> int:
    """Spec 160 §5.3: imported projects a key saved now would read, for the
    token sign on the key form. ``running`` ones are being read already and
    cost nothing more on save, so they are not counted."""
    if project_store is None:
        return 0
    n = 0
    try:
        projects = list(project_store.list_projects())
    except Exception:
        logger.warning("unscanned_imports: could not list projects", exc_info=True)
        return 0
    for p in projects:
        if p.get(STATE_FIELD) == RUNNING or not wants_scan(p):
            continue
        workspace = p.get("workspace") or ""
        if not workspace or not os.path.isdir(workspace) or has_goals(workspace):
            continue
        n += 1
    return n


def _activity_key(project: dict) -> float:
    value = project.get(ACTIVITY_FIELD)
    if not isinstance(value, str) or not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class ImportScanQueue:
    """The one daemon-side queue. ``runner`` does the model side:

    * ``pick(pid) -> "orbital" | "worker" | None`` — who can read this
      project now (blocking: it may read the agent setup cache);
    * ``async start(pid, who) -> session_id`` — mint and start the scan;
    * ``async wait(pid, session_id, who)`` — return when that run has ended;
    * ``async discard(pid, session_id)`` — drop a failed run's empty session;
    * ``async confirm(pid, session_id)`` — post the confirm card (round 4).
    """

    def __init__(self, project_store, runner, *, concurrency: int = CONCURRENCY):
        self._store = project_store
        self._runner = runner
        self._concurrency = concurrency
        self._slots: asyncio.Semaphore | None = None
        self._waiting: list[str] = []
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._pump: asyncio.Task | None = None

    # -- triggers ------------------------------------------------------------

    def kick(self, reason: str, *, include_failed: bool = True) -> None:
        """Queue every stamped project that still wants a scan and make sure
        the pump runs. Sync; needs a running event loop (a route handler)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("import scan: kick(%s) outside an event loop ignored", reason)
            return
        added = self._collect(include_failed=include_failed)
        if added:
            logger.info("import scan: %s queued %d project(s): %s", reason,
                        len(added), ", ".join(added))
        if self._waiting and (self._pump is None or self._pump.done()):
            self._pump = loop.create_task(self._run_pump())

    def resume(self) -> None:
        """Daemon start: carry on with projects already ``pending`` (or left
        ``running`` by a crash). A ``failed`` one waits for a real trigger."""
        self.kick("resume", include_failed=False)

    async def idle(self) -> None:
        """Return once nothing is queued or running (tests, shutdown)."""
        while True:
            pending = [t for t in (self._pump, *self._tasks) if t is not None and not t.done()]
            if not pending:
                return
            await asyncio.wait(pending)

    # -- internals -----------------------------------------------------------

    def _collect(self, *, include_failed: bool) -> list[str]:
        try:
            projects = list(self._store.list_projects())
        except Exception:
            logger.warning("import scan: could not list projects", exc_info=True)
            return []
        order = {p.get("project_id"): (i, p) for i, p in enumerate(projects)}
        added = []
        for pid, (_i, p) in order.items():
            if not pid or pid in self._inflight or pid in self._waiting:
                continue
            if not wants_scan(p, include_failed=include_failed):
                continue
            self._waiting.append(pid)
            added.append(pid)
        # Newest activity first; unknown last; creation order breaks ties.
        self._waiting.sort(key=lambda pid: (
            -_activity_key(order[pid][1]) if pid in order else float("inf"),
            order[pid][0] if pid in order else 0,
        ))
        return added

    async def _run_pump(self) -> None:
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._concurrency)
        while self._waiting:
            await self._slots.acquire()
            if not self._waiting:
                self._slots.release()
                break
            pid = self._waiting.pop(0)
            self._inflight.add(pid)
            task = asyncio.get_running_loop().create_task(self._run_one(pid))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _mark(self, pid: str, state: str, **extra) -> None:
        try:
            self._store.update_project(pid, {STATE_FIELD: state, AT_FIELD: _now(), **extra})
        except Exception:
            logger.warning("import scan: could not record %s for %s", state, pid,
                           exc_info=True)

    def _fail(self, pid: str) -> None:
        project = self._store.get_project(pid) or {}
        attempts = int(project.get(ATTEMPTS_FIELD) or 0) + 1
        self._mark(pid, FAILED, **{ATTEMPTS_FIELD: attempts})
        if attempts >= MAX_ATTEMPTS:
            logger.info("import scan: %s failed twice; first use takes over", pid)

    async def _run_one(self, pid: str) -> None:
        try:
            project = self._store.get_project(pid)
            if not wants_scan(project):
                return
            workspace = project.get("workspace") or ""
            if not workspace or not os.path.isdir(workspace):
                return
            if await asyncio.to_thread(has_goals, workspace):
                self._mark(pid, SKIPPED)
                return
            who = await asyncio.to_thread(self._runner.pick, pid)
            if who is None:
                return  # nobody can read it yet: stays as it is
            self._mark(pid, RUNNING)
            try:
                session_id = await self._runner.start(pid, who)
            except Exception as exc:
                if _runner_busy(exc):
                    # The runner is mid-turn (a keyless worker): not a
                    # failure, try again on the next trigger.
                    self._mark(pid, PENDING)
                    return
                logger.warning("import scan: could not start %s: %s", pid, exc)
                self._fail(pid)
                return
            logger.info("import scan: reading %s with %s (session %s)", pid, who, session_id)
            try:
                await self._runner.wait(pid, session_id, who)
            except Exception:
                logger.warning("import scan: waiting on %s failed", pid, exc_info=True)
            if await asyncio.to_thread(has_goals, workspace):
                self._mark(pid, DONE)
                logger.info("import scan: %s done", pid)
                try:
                    await self._runner.confirm(pid, session_id)
                except Exception:
                    logger.warning("import scan: could not post the confirm card for %s",
                                   pid, exc_info=True)
                return
            self._fail(pid)
            try:
                await self._runner.discard(pid, session_id)
            except Exception:
                logger.warning("import scan: could not discard %s/%s", pid, session_id,
                               exc_info=True)
        except Exception:
            logger.exception("import scan: %s crashed", pid)
        finally:
            self._inflight.discard(pid)
            if self._slots is not None:
                self._slots.release()


def _runner_busy(exc: BaseException) -> bool:
    """A start refused only because its runner is already mid-turn.

    Spec 155 lifted the project slot: the scan mints its own session and
    runs beside the user's sessions, so the manager path never refuses for
    another session. What is left is a keyless worker still busy with a turn
    ("already running"), which the next trigger retries. Anything else is a
    failed attempt."""
    detail = getattr(exc, "detail", None)
    text = detail if isinstance(detail, str) else str(exc)
    return "already running" in text


# ---------------------------------------------------------------------------
# The production runner: the manager, through the cold-start mint path
# ---------------------------------------------------------------------------


_WAIT_POLL_SECONDS = 1.0
_LIVE_STATUSES = ("running", "pending_approval", "waiting")


class ScanRunner:
    """The model side of the queue, on the daemon's agent manager."""

    def __init__(self, agent_manager, *, poll_seconds: float = _WAIT_POLL_SECONDS):
        self._mgr = agent_manager
        self._poll = poll_seconds

    def pick(self, pid: str) -> str | None:
        try:
            if self._mgr.manager_ready(pid):
                return "orbital"
            if self._mgr.effective_pinned_target(pid, ready=False):
                return "worker"
        except Exception:
            logger.warning("import scan: could not pick a runner for %s", pid,
                           exc_info=True)
        return None

    async def start(self, pid: str, who: str) -> str:
        from agent_os.api.routes import agents_v2
        return await agents_v2.start_import_scan(pid, who)

    async def wait(self, pid: str, session_id: str, who: str) -> None:
        if who == "orbital":
            # The loop task exists once start_agent returned; poll the same
            # run status the session list reports until it rests.
            while self._mgr.get_run_status(pid, session_id=session_id) in _LIVE_STATUSES:
                await asyncio.sleep(self._poll)
            _end_guess_flavour(self._mgr, pid, session_id)
            return
        # A worker's turn ends with its terminal row on disk (completed or
        # error); then nothing of it is still running.
        while True:
            rows = await asyncio.to_thread(_session_rows, self._mgr, pid, session_id)
            if any(_is_terminal(r) for r in rows) and not _worker_running(
                    self._mgr, pid, session_id):
                return
            await asyncio.sleep(self._poll)

    async def confirm(self, pid: str, session_id: str) -> None:
        from agent_os.api.routes import agents_v2
        await agents_v2.post_scan_confirm(pid, session_id)

    async def discard(self, pid: str, session_id: str) -> None:
        rows = await asyncio.to_thread(_session_rows, self._mgr, pid, session_id)
        if rows is None or not _only_noise(rows):
            return
        try:
            await self._mgr.delete_session(pid, session_id)
        except FileNotFoundError:
            pass


# Round 1's name, kept for callers that construct it directly.
ManagerScanRunner = ScanRunner


def _end_guess_flavour(mgr, pid: str, session_id: str) -> None:
    """The scan run ended: a later turn in the same live handle (a hot resume)
    gets the plain prompt every other session gets, not the scan section."""
    from agent_os.daemon_v2.models import make_session_key
    handle = getattr(mgr, "_handles", {}).get(make_session_key(pid, session_id))
    loop = getattr(handle, "loop", None)
    ctx = getattr(getattr(loop, "_context_manager", None), "_base_ctx", None)
    if ctx is not None and isinstance(getattr(ctx, "cold_start", None), str):
        ctx.cold_start = False


def _is_terminal(row: dict) -> bool:
    meta = row.get("_meta") if isinstance(row, dict) else None
    return isinstance(meta, dict) and meta.get("event") == "sub_agent_terminal"


def _worker_running(mgr, pid: str, session_id: str) -> bool:
    sam = getattr(mgr, "_sub_agent_manager", None)
    if sam is None:
        return False
    try:
        return any(a.get("status") == "running"
                   for a in sam.list_active(pid, session_id=session_id))
    except Exception:
        return False


def _session_rows(mgr, pid: str, session_id: str) -> list[dict] | None:
    """The session's rows: the live handle's, else read from its JSONL."""
    import json
    from agent_os.daemon_v2.models import make_session_key
    handle = getattr(mgr, "_handles", {}).get(make_session_key(pid, session_id))
    session = getattr(handle, "session", None)
    if session is not None:
        return list(session.get_messages())
    try:
        resolved = mgr._resolve_session_uuid_on_disk(pid, session_id)
    except Exception:
        resolved = None
    if not resolved:
        return []
    rows = []
    try:
        with open(resolved[1], encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("role") != "meta":
                    rows.append(rec)
    except OSError:
        return None
    return rows


def _only_noise(rows: list[dict]) -> bool:
    """True when nothing in the session is the user's or a real reply (the
    agent's final text, or a worker's answer), so removing it loses nothing
    anyone wrote."""
    for row in rows:
        role = row.get("role")
        if role == "user" and row.get("source") != "daemon":
            return False
        if role == "assistant" and not row.get("tool_calls"):
            content = row.get("content")
            if content.strip() if isinstance(content, str) else content:
                return False
    return True


def scan_agent(agent_manager, project_store) -> dict | None:
    """Spec 160 round 2: the worker a keyless scan would run on, for the
    cost sign ({"slug", "name"} with the product name, never a raw id) —
    the first unscanned project's effective pinned worker, else the worker a
    new project would get. None when no worker is ready."""
    if agent_manager is None:
        return None
    slug = None
    try:
        for p in (project_store.list_projects() if project_store is not None else []):
            if wants_scan(p) and p.get(STATE_FIELD) != RUNNING:
                slug = agent_manager.effective_pinned_target(p.get("project_id"), ready=False)
                if slug:
                    break
        if slug is None:
            hidden = []
            store = getattr(agent_manager, "_settings_store", None)
            if store is not None:
                hidden = list(getattr(store.get(), "sub_agents_hidden", None) or [])
            workers = agent_manager.available_worker_slugs(
                "", project={"disabled_sub_agents": hidden})
            slug = workers[0] if workers else None
    except Exception:
        logger.warning("import scan: could not resolve the keyless runner", exc_info=True)
        return None
    if not slug:
        return None
    name = slug
    registry = getattr(agent_manager, "_registry", None)
    try:
        manifest = registry.get(slug) if registry is not None else None
        if manifest is not None and getattr(manifest, "name", None):
            name = manifest.name
    except Exception:
        pass
    return {"slug": slug, "name": name}


# ---------------------------------------------------------------------------
# Module singleton (configured by the app factory)
# ---------------------------------------------------------------------------


_queue: ImportScanQueue | None = None


def configure(queue) -> None:
    global _queue
    _queue = queue


def get_queue():
    return _queue


def kick(reason: str, *, include_failed: bool = True) -> None:
    """Fire a decidable trigger. A no-op until the app wires a queue (unit
    tests of the routes) and never raises into a route."""
    if _queue is None:
        return
    try:
        _queue.kick(reason, include_failed=include_failed)
    except Exception:
        logger.warning("import scan: kick(%s) failed", reason, exc_info=True)

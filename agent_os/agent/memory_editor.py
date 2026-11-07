# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The memory pass (spec 124 §3.7): the deterministic shell around ONE turn
of the project's Memory session.

The memory editor this replaced was a separate read-only agent loop on the
utility model that returned choices by id and paid a full prompt miss on
every run. The Memory session is a real session of the project (source
``memory``, visible in the session list, the user can type into it): same
system prompt, same tool registry, Layer-1 rows byte-identical to the
triggering run's snapshot, so its first call rides the main loop's prefix.
It works on the files directly with edit/write (confined to ``orbital/*.md``
by ``memory_tool_guard``); the daemon stamps ids, keeps the archive contents
tables, and bounds the turn (``EDITOR_MAX_CALLS`` tool calls).

What stays deterministic, in order (``run_pass``):

1. ``backup_live_files`` — the four live files into a rolling backup (last 5);
2. ``workspace_files.prepass`` — normalise, stamp, contents tables, INDEX
   archive lines;
3. the task message (``build_task_message``) appended to the Memory session
   as a user row, then one bounded turn (the runner the agent manager
   supplies);
4. ``postpass_conservation`` — every id the pass started with is still live
   or in its archive, else it comes back from the snapshot; INDEX keeps its
   ``*_ARCHIVE.md`` lines;
5. the outcome (``edited`` / ``no_change`` / ``index_only`` / ``restored`` /
   ``skipped`` / ``failed``), the run record next to the backup, and one
   summary log line with the turn's token numbers and cost.

Nothing is deleted: an entry only leaves a live file through its archive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent_os.agent import memory_entries as _mem
from agent_os.utils.text_io import read_text_lenient

logger = logging.getLogger(__name__)

LIVE_KEYS = ("state", "decisions", "lessons", "index")
EDITABLE_KEYS = ("state", "decisions", "lessons")

# The Memory turn's tool cap: the daemon ends the turn when it is reached.
EDITOR_MAX_CALLS = 12
# The whole pass (one turn) is bounded; a hung provider never holds the
# single-flight slot for good.
EDITOR_TOTAL_TIMEOUT_S = 900.0
# Automatic (over-budget) runs are at most this often per project. An
# explicit checkpoint_state request is not bound by it.
EDITOR_AUTO_INTERVAL_S = 30 * 60

BACKUP_DIRNAME = "backups"
BACKUP_PREFIX = "memory-editor-"
BACKUP_KEEP = 5

# Key in orbital/.memory_cleanup.json holding the last successful run (UTC ISO).
WATERMARK_KEY = "editor_last_run"
_MARKER_FILE = ".memory_cleanup.json"
_SESSIONS_FALLBACK_DAYS = 7
_MAX_SESSIONS_LISTED = 30

FILE_BY_KEY = {
    "state": "PROJECT_STATE.md",
    "decisions": "DECISIONS.md",
    "lessons": "LESSONS.md",
    "index": "INDEX.md",
}
_KEY_BY_NAME = {v: k for k, v in FILE_BY_KEY.items()}


# ---------------------------------------------------------------------------
# Single flight per project (every caller: loop scheduler, pinned coordinator)
# ---------------------------------------------------------------------------

_RUNNING: set[str] = set()


def claim(workspace: str) -> bool:
    """Take the project's pass slot. Synchronous, so it is race-free."""
    key = os.path.realpath(workspace)
    if key in _RUNNING:
        return False
    _RUNNING.add(key)
    return True


def release(workspace: str) -> None:
    _RUNNING.discard(os.path.realpath(workspace))


def is_running(workspace: str) -> bool:
    return os.path.realpath(workspace) in _RUNNING


# ---------------------------------------------------------------------------
# Spec 155: the record lock. Sessions of one project run in parallel; their
# round-end record turns take this lock before the [MEMORY] row is built, so
# the second turn's row is the file as the first one left it. In process: one
# daemon owns a project, and the holder is always a live coroutine whose
# ``finally`` releases it, so a crash or a quit cannot leave it stuck.
# ---------------------------------------------------------------------------

_RECORD_LOCKS: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}


def record_lock(workspace: str) -> asyncio.Lock:
    """The project's record lock, keyed like ``claim`` by the workspace's
    realpath (two projects on one folder share it). One per workspace for the
    daemon's lifetime (the daemon runs one event loop; a lock made on another
    loop, which only tests do, is replaced)."""
    key = os.path.realpath(workspace)
    loop = asyncio.get_running_loop()
    entry = _RECORD_LOCKS.get(key)
    if entry is not None and entry[0] is loop:
        return entry[1]
    lock = asyncio.Lock()
    _RECORD_LOCKS[key] = (loop, lock)
    return lock


# ---------------------------------------------------------------------------
# Watermark + the automatic-run gate
# ---------------------------------------------------------------------------

def _marker_path(orbital_dir: str) -> str:
    return os.path.join(orbital_dir, _MARKER_FILE)


def read_marker(orbital_dir: str) -> dict:
    text = read_text_lenient(_marker_path(orbital_dir))
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def watermark(orbital_dir: str) -> datetime | None:
    """When the pass last finished successfully.

    A project upgraded from before the editor has a marker without the key;
    the marker file's own mtime is when its last successful cleanup ran,
    which is the same moment by the old definition.
    """
    stamp = _parse_iso(read_marker(orbital_dir).get(WATERMARK_KEY))
    if stamp is not None:
        return stamp
    try:
        return datetime.fromtimestamp(
            os.stat(_marker_path(orbital_dir)).st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _layer1_paths(orbital_dir: str) -> dict[str, str]:
    return {k: os.path.join(orbital_dir, FILE_BY_KEY[k]) for k in LIVE_KEYS}


def _mtimes(orbital_dir: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for key, path in _layer1_paths(orbital_dir).items():
        try:
            out[key] = os.stat(path).st_mtime_ns
        except OSError:
            out[key] = None
    return out


def _byte_sizes(orbital_dir: str) -> dict[str, int | None]:
    """Byte sizes beside the mtimes: two writes inside one filesystem
    timestamp tick (Windows NTFS rounds LastWriteTime to the system timer)
    leave ``st_mtime_ns`` identical, so a rewrite that changed the length
    is still a delta."""
    out: dict[str, int | None] = {}
    for key, path in _layer1_paths(orbital_dir).items():
        try:
            out[_SIZE_KEY.format(key)] = os.stat(path).st_size
        except OSError:
            out[_SIZE_KEY.format(key)] = None
    return out


_SIZE_KEY = "{}_size"


def has_delta(orbital_dir: str) -> bool:
    """True if any live file changed since the last successful pass."""
    stored = read_marker(orbital_dir)
    if not stored:
        return True
    current = _mtimes(orbital_dir)
    if any(str(stored.get(k)) != str(current.get(k)) for k in LIVE_KEYS):
        return True
    sizes = _byte_sizes(orbital_dir)
    # Size keys were added later; a marker without them (older daemon) falls
    # back to the mtime-only answer above.
    return any(
        k in stored and str(stored.get(k)) != str(v) for k, v in sizes.items()
    )


def write_marker(orbital_dir: str, *, editor_ran_at: datetime | None) -> None:
    """Record post-pass mtimes (the no-delta gate) and the watermark.

    Older daemons read only the four mtime keys and ignore the extra one; a
    marker they rewrite simply loses the watermark, which falls back to the
    marker's mtime.
    """
    data: dict = dict(_mtimes(orbital_dir))
    data.update(_byte_sizes(orbital_dir))
    previous = read_marker(orbital_dir).get(WATERMARK_KEY)
    if editor_ran_at is not None:
        data[WATERMARK_KEY] = editor_ran_at.astimezone(timezone.utc).isoformat()
    elif previous:
        data[WATERMARK_KEY] = previous
    try:
        os.makedirs(orbital_dir, exist_ok=True)
        tmp = _marker_path(orbital_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _marker_path(orbital_dir))
    except OSError as e:
        logger.warning("memory pass: could not write the cleanup marker (%s)", e)


def auto_run_due(orbital_dir: str, *, now: datetime | None = None) -> bool:
    """Gate for the AUTOMATIC over-budget trigger (cheap: stats + one JSON).

    Due only when a live file changed since the last pass (the pass already
    saw these exact bytes otherwise) and the last successful run is at least
    ``EDITOR_AUTO_INTERVAL_S`` old — a file that stays over budget because its
    content is recent and live must not buy a pass on every write.
    """
    if not has_delta(orbital_dir):
        return False
    last = _parse_iso(read_marker(orbital_dir).get(WATERMARK_KEY))
    if last is None:
        return True
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds() >= EDITOR_AUTO_INTERVAL_S


# ---------------------------------------------------------------------------
# Rolling backup
# ---------------------------------------------------------------------------

def backup_live_files(orbital_dir: str, *, now: datetime | None = None) -> str | None:
    """Copy the four live files into ``orbital/backups/memory-editor-<ts>/``.

    Keeps the newest ``BACKUP_KEEP`` backups (only this pass's own dirs are
    ever pruned). Returns the directory, or None when there was nothing to
    back up or the copy failed — in which case the caller must not edit.
    """
    now = now or datetime.now(timezone.utc)
    root = os.path.join(orbital_dir, BACKUP_DIRNAME)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    dest = os.path.join(root, BACKUP_PREFIX + stamp)
    try:
        os.makedirs(dest, exist_ok=True)
        copied = 0
        for path in _layer1_paths(orbital_dir).values():
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(dest, os.path.basename(path)))
                copied += 1
        if not copied:
            shutil.rmtree(dest, ignore_errors=True)
            return None
    except OSError as e:
        logger.warning("memory pass: backup failed (%s); not editing", e)
        return None
    try:
        mine = sorted(
            d for d in os.listdir(root)
            if d.startswith(BACKUP_PREFIX) and os.path.isdir(os.path.join(root, d))
        )
        for old in mine[:-BACKUP_KEEP]:
            shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    except OSError:
        pass
    return dest


# ---------------------------------------------------------------------------
# Sessions since the last pass (evidence for "finished")
# ---------------------------------------------------------------------------

def _disk_sessions(workspace: str) -> list[dict]:
    """Fallback session rows from the JSONL files (no agent manager)."""
    from agent_os.agent.project_paths import ProjectPaths
    from agent_os.daemon_v2.native_worker import is_worker_session_stem

    sessions_dir = ProjectPaths(workspace).sessions_dir
    rows = []
    try:
        names = os.listdir(sessions_dir)
    except OSError:
        return rows
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        stem = name[:-6]
        if is_worker_session_stem(stem):
            continue
        try:
            mtime = os.stat(os.path.join(sessions_dir, name)).st_mtime
        except OSError:
            continue
        rows.append({
            "session_id": stem,
            "session_uuid": stem,
            "name": None,
            "origin": "chat",
            "status": "idle",
            "last_activity_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        })
    return rows


def sessions_since(list_sessions, workspace: str, since: datetime | None) -> list[dict]:
    """Sessions active since ``since`` (newest first), the Memory session
    itself excluded — its own turns are not evidence."""
    rows_fn = list_sessions or (lambda: _disk_sessions(workspace))
    try:
        rows = [r for r in rows_fn() if isinstance(r, dict) and r.get("session_uuid")]
    except Exception:  # noqa: BLE001 — evidence is optional
        logger.warning("memory pass: could not list sessions", exc_info=True)
        return []
    floor = since or (datetime.now(timezone.utc) - timedelta(days=_SESSIONS_FALLBACK_DAYS))
    picked = []
    for r in rows:
        if r.get("origin") == "memory":
            continue
        at = _parse_iso(r.get("last_activity_at"))
        if at is not None and at >= floor:
            picked.append((at, r))
    picked.sort(key=lambda p: p[0], reverse=True)
    return [r for _at, r in picked[:_MAX_SESSIONS_LISTED]]


# ---------------------------------------------------------------------------
# The task message (spec 124 §3.7): fixed bytes, filled fields
# ---------------------------------------------------------------------------

_TASK_HEAD = (
    "Tidy this project's memory. You are in the project's own context: the system\n"
    "prompt, tools and memory files above are what the working agent sees. Work on\n"
    "the files directly with read, grep, read_session, edit and write. The daemon\n"
    "stamps ids and maintains the archive contents tables; you never touch\n"
    "<!--mem …--> lines.\n"
    "Edit these files only with edit and write; never through the shell.\n"
    "\n"
    "SIZES (bullet lines only; indented detail does not count toward soft)\n"
)

_DO_OVER = {
    "state": (
        "PROJECT_STATE.md is over: move entries that are finished or superseded to\n"
        "   PROJECT_STATE_ARCHIVE.md; merge entries that say the same thing; fold\n"
        "   indented detail into its line. The sessions listed below are your evidence\n"
        "   for \"finished\" (read_session with since=…, grep, small limits)."
    ),
    "decisions": (
        "DECISIONS.md is over: move decisions that were superseded in practice to\n"
        "   DECISIONS_ARCHIVE.md; merge entries that say the same thing; fold indented\n"
        "   detail into its line. The sessions listed below are your evidence for\n"
        "   \"superseded\" (read_session with since=…, grep, small limits)."
    ),
    "lessons": (
        "LESSONS.md is over: move lessons that no longer apply to LESSONS_ARCHIVE.md;\n"
        "   merge entries that say the same thing; fold indented detail into its line.\n"
        "   The sessions listed below are your evidence (read_session with since=…,\n"
        "   grep, small limits)."
    ),
    "index": (
        "INDEX.md is over: merge lines upward into folder or topic lines; a file gets\n"
        "   its own line only if it is the source of truth for something."
    ),
}

_DO_FOLDED = (
    "{filename}'s bullet lines do not fit the injected view, which folds it by\n"
    "   section: group its entries under `## topic` headings and merge same-topic\n"
    "   entries into one line."
)

_DO_POINTERS = (
    "These paths lost their only live mention since the last pass:\n"
    "   {paths}\n"
    "   Decide whether INDEX.md needs a line for them. Folder or topic granularity;\n"
    "   a file gets its own line only if it is the source of truth for something."
)

_DO_FORCED = (
    "This pass was requested explicitly (checkpoint_state); nothing is over its\n"
    "   soft budget. Look for entries that are finished or superseded and archive\n"
    "   or merge only those."
)

_DO_ARCHIVE_LINE = (
    "If you archive into a file, update that archive's line in INDEX.md: the\n"
    "   one-sentence description of what is in it. The daemon fills count and dates."
)

_DO_NOTHING = "If nothing is finished or superseded, say so and stop. Never force cuts."

_RULES = (
    "RULES\n"
    "- Archive only what is no longer true or no longer live, never because it is\n"
    "  old. A standing decision or lesson stays, however old.\n"
    "- Move entries intact: the bullet line, its mem comment line, its indented\n"
    "  detail, appended under `## [archived {today}]` in the archive. Write the\n"
    "  archive first, then remove from the live file. For a merge, write the merged\n"
    "  entry in the live file, then move the originals under\n"
    "  `## [merged {today} into id:<new id>]`.\n"
    "- A merged entry is shorter than its parts and says only what they said.\n"
    "  Never add facts. Never touch entries tagged tag:pinned.\n"
    "- Do not read an archive whole; grep it.\n"
    "- Close an ask in orbital/ASKS.md only with the user's own words quoted, as\n"
    "  that file's header says.\n"
    "- INDEX.md lists folders and topics, not dated files. Keep every *_ARCHIVE.md line.\n"
)

_TASK_TAIL = "At most {cap} tool calls. Finish with one line saying what moved and where."


def _minutes(value) -> str:
    at = _parse_iso(value)
    if at is None:
        return str(value or "?")
    return at.strftime("%Y-%m-%dT%H:%M")


def build_task_message(
    *, sizes: dict[str, tuple[int, int, int]], folded: list[str], over: list[str],
    forced: bool, pointer_paths: list[str], last_pass: datetime | None,
    sessions: list[dict], today: str, tool_cap: int,
) -> str:
    """The daemon-generated user row that starts a Memory turn.

    ``sizes`` maps each live file key to ``(now, soft, target)`` tokens;
    ``over`` lists the keys over their soft budget (the trigger reason);
    ``folded`` the filenames whose injected view folds; ``pointer_paths``
    the paths folded legacy pointers named (prepass). Everything but these
    fields, the date in RULES, LAST PASS and the session list is fixed
    bytes.
    """
    lines = [_TASK_HEAD]
    for key in LIVE_KEYS:
        now, soft, target = sizes[key]
        if key in over:
            lines.append(f"- {FILE_BY_KEY[key]}: {now:,} tokens (soft {soft:,}, target {target:,}) — OVER\n")
        else:
            lines.append(f"- {FILE_BY_KEY[key]}: {now:,} (soft {soft:,}) — within\n")
    lines.append(f"Folded in the injected view: {', '.join(folded) if folded else 'none'}.\n")
    lines.append("\nDO\n")
    items: list[str] = []
    for key in LIVE_KEYS:
        if key in over:
            items.append(_DO_OVER[key])
    for filename in folded:
        items.append(_DO_FOLDED.format(filename=filename))
    if pointer_paths:
        items.append(_DO_POINTERS.format(paths=", ".join(pointer_paths)))
    if forced and not over:
        items.append(_DO_FORCED)
    if any(k in over for k in EDITABLE_KEYS) or forced:
        items.append(_DO_ARCHIVE_LINE)
    items.append(_DO_NOTHING)
    for n, item in enumerate(items, 1):
        lines.append(f"{n}. {item}\n")
    lines.append("\n" + _RULES.format(today=today) + "\n")
    last = last_pass.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M") if last_pass else "never"
    lines.append(f"LAST PASS: {last}\n")
    lines.append("SESSIONS ACTIVE SINCE THEN (newest first):\n")
    if sessions:
        for s in sessions:
            name = " ".join(str(s.get("name") or "").split())[:80]
            lines.append(f"- {s.get('session_uuid')}  last_activity={_minutes(s.get('last_activity_at'))}  {name}\n")
    else:
        lines.append("(none)\n")
    lines.append("\n" + _TASK_TAIL.format(cap=tool_cap))
    return "".join(lines)


# ---------------------------------------------------------------------------
# The tool guard: edit/write confined to <workspace>/orbital/*.md
# ---------------------------------------------------------------------------

_GUARDED_TOOLS = ("edit", "write")
# A shell command that names any memory file by basename — any path form,
# the backups folder included — is refused outright. Reads are not told
# apart from writes (a heredoc or a `python -` body cannot be classified
# cheaply); the read and grep tools cover reads. Both live smoke passes had
# the agent rewrite PROJECT_STATE through `cat >>` / `python`, skipping the
# write chokepoint (normalise, stamp, conservation).
_MEMORY_FILE_IN_SHELL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:PROJECT_STATE|DECISIONS|LESSONS|INDEX|ASKS|[A-Z_]+_ARCHIVE)\.md(?![A-Za-z0-9_])"
)
_SHELL_REFUSAL = (
    "Error: Memory files are edited with the edit/write tools only (the daemon "
    "stamps ids and keeps the archive tables); run this command without them. "
    "Every shell command that names PROJECT_STATE.md, DECISIONS.md, LESSONS.md, "
    "INDEX.md, ASKS.md or an *_ARCHIVE.md is refused, reads included — use the "
    "read and grep tools to look at them."
)


def memory_tool_guard(workspace: str):
    """A deterministic check for the Memory session's loop: ``edit`` and
    ``write`` may only target ``<workspace>/orbital/*.md`` (the live files,
    their archives, ASKS.md), and ``shell`` may not name a memory file at
    all. Every other tool is unrestricted. Returns the error text to hand
    back as the tool result, or None."""
    orbital = os.path.realpath(os.path.join(workspace, "orbital"))
    root = os.path.realpath(workspace)

    def guard(tc: dict) -> str | None:
        args = tc.get("arguments") or {}
        if tc.get("name") == "shell":
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and _MEMORY_FILE_IN_SHELL_RE.search(command):
                return _SHELL_REFUSAL
            return None
        if tc.get("name") not in _GUARDED_TOOLS:
            return None
        raw = args.get("path") if isinstance(args, dict) else None
        if not raw or not isinstance(raw, str):
            return None   # the tool reports its own missing argument
        target = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(root, raw))
        if os.path.dirname(target) == orbital and target.endswith(".md"):
            return None
        return (
            f"Error: the Memory session may only edit or write orbital/*.md "
            f"(the memory files, their archives and ASKS.md); {raw!r} is outside that."
        )

    return guard


# ---------------------------------------------------------------------------
# INDEX: archive lines survive a rewrite
# ---------------------------------------------------------------------------

_ARCHIVE_NAME_RE = re.compile(r"[A-Z_]+_ARCHIVE\.md")


def _keep_archive_lines(new_index: str, old_index: str) -> str:
    """A rewritten INDEX keeps every line that was the route to an archive.

    The old line is restored verbatim when the rewrite dropped the archive's
    name altogether — an archive nothing points to is lost in practice.
    """
    text = new_index.strip() + "\n"
    restored = []
    for line in old_index.split("\n"):
        names = _ARCHIVE_NAME_RE.findall(line)
        if names and any(n not in text for n in names) and line not in restored:
            restored.append(line)
    if restored:
        text += "\n" + "\n".join(restored) + "\n"
    return text


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """What the agent manager reports back from one Memory turn."""
    mode: str                       # "snapshot" (live run's Layer-1 rows) | "disk"
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    ended: str = ""                 # the loop's exit path ("text", "tool_cap", …)
    provider: str = ""
    model: str = ""

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens, "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens, "ended": self.ended,
            "provider": self.provider, "model": self.model,
        }


@dataclass
class EditorResult:
    ok: bool                            # the turn ran (or was declined) without error
    outcome: str                        # edited | no_change | index_only | restored | skipped | failed
    calls: int = 0
    turn: TurnResult | None = None
    written: list[str] = field(default_factory=list)      # file keys that changed
    restored: list[str] = field(default_factory=list)     # ids put back by conservation
    archived: list[str] = field(default_factory=list)     # ids that left a live file for its archive
    backup_dir: str | None = None
    error: str | None = None


def _cost_usd(turn: TurnResult) -> float:
    try:
        from agent_os.agent.pricing import resolve_rates
        rates = resolve_rates(turn.provider or "custom", turn.model or "")
        fresh = max(0, turn.input_tokens - turn.cached_tokens)
        return (
            fresh * rates.input_per_1m
            + turn.cached_tokens * rates.cached_input_per_1m
            + turn.output_tokens * rates.output_per_1m
        ) / 1_000_000
    except Exception:  # noqa: BLE001 — a pricing gap never hides the pass
        return 0.0


def _sizes(workspace_files) -> dict[str, tuple[int, int, int]]:
    out = {}
    for key in LIVE_KEYS:
        raw = workspace_files.read(key) or ""
        now = int(_mem.est_tokens(_mem._budget_text(raw, key)))
        out[key] = (now, _mem.FILE_BUDGETS[key]["soft"], _mem.consolidation_target(key))
    return out


def _folded(workspace_files) -> list[str]:
    """Entry files whose injected view folds (a section count line appears)."""
    out = []
    for key in _mem.ENTRY_KEYS:
        raw = workspace_files.read(key)
        if not raw:
            continue
        view = _mem.inject_view(raw, key, _mem.FILE_BUDGETS[key]["hard"]) or ""
        if "not shown — read " in view:
            out.append(FILE_BY_KEY[key])
    return out


async def run_pass(
    workspace_files, run_turn, *, list_sessions=None, project_id: str = "",
    today: str | None = None, now: datetime | None = None,
    over_keys=(), forced: bool = False, trigger: str = "over_budget",
) -> EditorResult:
    """One pass: backup → prepass → task message → one Memory turn →
    conservation → outcome. Never raises for model/tool trouble — that comes
    back as ``ok=False``.

    ``run_turn(task_text)`` is supplied by the agent manager: it appends the
    text to the Memory session, runs one bounded turn and returns a
    ``TurnResult`` (None when it declined, e.g. the user is mid-turn there).
    """
    from agent_os.agent import workspace_files as _wf

    now = now or datetime.now(timezone.utc)
    today = today or datetime.now().date().isoformat()
    orbital_dir = workspace_files.dir

    backup_dir = backup_live_files(orbital_dir, now=now)
    if backup_dir is None and any(workspace_files.exists(k) for k in LIVE_KEYS):
        return EditorResult(ok=False, outcome="failed", error="backup failed")
    pre = _wf.prepass(workspace_files, today=today)

    snapshot = {k: workspace_files.read(k) or "" for k in LIVE_KEYS}
    since = watermark(orbital_dir)
    sessions = sessions_since(list_sessions, workspace_files.workspace, since)
    over = [k for k in LIVE_KEYS if k in set(over_keys)] or [
        k for k in LIVE_KEYS if _mem.over_soft_budget(snapshot[k], k)]
    task_text = build_task_message(
        sizes=_sizes(workspace_files), folded=_folded(workspace_files), over=over,
        forced=forced, pointer_paths=list(pre.pointer_paths), last_pass=since,
        sessions=sessions, today=today, tool_cap=EDITOR_MAX_CALLS,
    )

    result = EditorResult(ok=True, outcome="no_change", backup_dir=backup_dir)
    try:
        turn = await asyncio.wait_for(run_turn(task_text), timeout=EDITOR_TOTAL_TIMEOUT_S)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("memory session: the turn exceeded %.0fs; giving up on this pass",
                       EDITOR_TOTAL_TIMEOUT_S)
        result.ok, result.outcome, result.error = False, "failed", "turn timed out"
        turn = None
    except Exception as e:  # noqa: BLE001 — the shell reports, never crashes the caller
        logger.warning("memory session: the turn failed (%s)", e)
        result.ok, result.outcome, result.error = False, "failed", str(e)
        turn = None
    result.turn = turn
    result.calls = turn.llm_calls if turn else 0

    # Post-pass, whatever happened in the turn: INDEX keeps its archive
    # lines, the archive lines' counts/dates are refreshed, and every id the
    # pass started with is still live or archived.
    current_index = workspace_files.read("index") or ""
    if current_index != snapshot["index"]:
        kept = _keep_archive_lines(current_index, snapshot["index"])
        if kept.strip() != current_index.strip():
            workspace_files.write("index", kept)
    _wf.refresh_index_archive_lines(workspace_files)
    changed = [k for k in EDITABLE_KEYS if (workspace_files.read(k) or "") != snapshot[k]]
    result.restored = _wf.postpass_conservation(workspace_files, snapshot, changed)
    result.written = [k for k in LIVE_KEYS if (workspace_files.read(k) or "") != snapshot[k]]
    result.archived = _archived_ids(workspace_files, snapshot, changed)

    if result.ok:
        if turn is None:
            result.outcome = "skipped"
        elif result.restored:
            result.outcome = "restored"
        elif any(k in EDITABLE_KEYS for k in result.written):
            result.outcome = "edited"
        elif "index" in result.written:
            result.outcome = "index_only"
        else:
            result.outcome = "no_change"
    if turn is not None:
        logger.info(
            "memory session: mode=%s calls=%d input=%d cached=%d output=%d cost=$%.4f "
            "(project=%s trigger=%s tool_calls=%d ended=%s outcome=%s written=%s restored=%s)",
            turn.mode, turn.llm_calls, turn.input_tokens, turn.cached_tokens,
            turn.output_tokens, _cost_usd(turn), project_id, trigger, turn.tool_calls,
            turn.ended, result.outcome, result.written, result.restored,
        )
    else:
        logger.info("memory pass: %s (project=%s trigger=%s written=%s restored=%s error=%s)",
                    result.outcome, project_id, trigger, result.written, result.restored,
                    result.error)
    _write_run_record(backup_dir, result, trigger=trigger)
    return result


def _archived_ids(workspace_files, snapshot: dict[str, str], keys) -> list[str]:
    """The ids the pass moved out of a live file and into its archive (after
    conservation, so an id that left without reaching the archive is back in
    the live file and not counted). Parsed with the same block parser the
    conservation check uses; nothing new is judged here."""
    from agent_os.agent import state_blocks
    out: list[str] = []
    for key in keys:
        before = snapshot.get(key) or ""
        if not before:
            continue
        _, before_blocks = state_blocks.parse(before)
        _, after_blocks = state_blocks.parse(workspace_files.read(key) or "")
        live_now = {b.id for b in after_blocks if b.id}
        in_archive = _mem.archive_ids(workspace_files.read(_mem.ARCHIVE_OF[key]))
        out.extend(b.id for b in before_blocks
                   if b.id and b.id not in live_now and b.id in in_archive)
    return out


def _write_run_record(backup_dir: str | None, result: EditorResult, *, trigger: str = "") -> None:
    """Keep the pass's outcome next to its backup, for audit."""
    if not backup_dir:
        return
    try:
        with open(os.path.join(backup_dir, "editor-run.json"), "w", encoding="utf-8") as f:
            json.dump({
                "outcome": result.outcome,
                "calls": result.calls,
                "trigger": trigger,
                "turn": result.turn.as_dict() if result.turn else None,
                "written": result.written,
                "restored": result.restored,
                "error": result.error,
            }, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

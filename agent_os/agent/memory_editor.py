# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The memory editor (spec 089 §4.2): tidies Layer-1 memory by id.

It replaced the whole-file LLM merge, which regenerated PROJECT_STATE /
DECISIONS / LESSONS from a size-capped view and so silently dropped whatever
it never saw. The editor never writes a file whole (INDEX, small and
regenerable, is the one exception). It is a short read-only agent loop on the
project's utility model that returns CHOICES by id:

- ``archive`` — move one entry, byte-exact, to the file's archive and leave
  ``[archived DATE id:X] <pointer> → ARCHIVE`` in its place;
- ``merge`` — replace several entries with one condensed entry; the originals
  go to the archive byte-exact, with a pointer naming their ids;
- ``index`` — a complete new INDEX.md.

Sessions since the last run are evidence of whether a line is still live,
done or replaced — never a source of new facts. Nothing is deleted.

Safety: the four live files are copied into a rolling backup (last 5) before
anything is touched; every file is written only if it is unchanged since the
editor read it (OCC — an agent write during the run makes the editor skip
that file); archives are written before the live file shrinks; the apply
step has no awaits, so a cancelled run never half-applies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent_os.agent import memory_entries as _mem
from agent_os.agent import state_blocks

logger = logging.getLogger(__name__)

LIVE_KEYS = ("state", "decisions", "lessons", "index")
EDITABLE_KEYS = ("state", "decisions", "lessons")

# At most this many model calls per run (tool rounds + the final answer).
EDITOR_MAX_CALLS = 12
EDITOR_CALL_TIMEOUT_S = 240.0
EDITOR_TOTAL_TIMEOUT_S = 900.0
# A tool result is evidence, not a document: cap what one call can add.
EDITOR_TOOL_RESULT_CHARS = 12_000
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
    """Take the project's editor slot. Synchronous, so it is race-free."""
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
# Watermark + the automatic-run gate
# ---------------------------------------------------------------------------

def _marker_path(orbital_dir: str) -> str:
    return os.path.join(orbital_dir, _MARKER_FILE)


def read_marker(orbital_dir: str) -> dict:
    try:
        with open(_marker_path(orbital_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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
    """When the editor last finished successfully.

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


def has_delta(orbital_dir: str) -> bool:
    """True if any live file changed since the last successful pass."""
    stored = read_marker(orbital_dir)
    if not stored:
        return True
    current = _mtimes(orbital_dir)
    return any(str(stored.get(k)) != str(current.get(k)) for k in LIVE_KEYS)


def write_marker(orbital_dir: str, *, editor_ran_at: datetime | None) -> None:
    """Record post-pass mtimes (the no-delta gate) and the watermark.

    Older daemons read only the four mtime keys and ignore the extra one; a
    marker they rewrite simply loses the watermark, which falls back to the
    marker's mtime.
    """
    data: dict = dict(_mtimes(orbital_dir))
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
        logger.warning("memory editor: could not write the cleanup marker (%s)", e)


def auto_run_due(orbital_dir: str, *, now: datetime | None = None) -> bool:
    """Gate for the AUTOMATIC over-budget trigger (cheap: stats + one JSON).

    Due only when a live file changed since the last pass (the editor already
    saw these exact bytes otherwise) and the last successful run is at least
    ``EDITOR_AUTO_INTERVAL_S`` old — a file that stays over budget because its
    content is recent and live must not buy an editor run on every write.
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

    Keeps the newest ``BACKUP_KEEP`` backups (only this editor's own dirs are
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
        logger.warning("memory editor: backup failed (%s); not editing", e)
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
# Ids everywhere before the editor looks
# ---------------------------------------------------------------------------

def normalise_ids(workspace_files, *, today: str) -> list[str]:
    """Give every entry an id so it can be chosen, archived and recalled.

    PROJECT_STATE: a system write runs the chokepoint, which stamps every
    bullet (lines written outside the tools — external agents, pinned
    workers — arrive unstamped). DECISIONS/LESSONS: ``stamp`` fills in ids
    for unstamped entries and leaves stamped ones byte-identical. A new stamp
    carries today's date, so an unstamped line counts as recent. Returns the
    keys it rewrote.
    """
    touched: list[str] = []
    state = workspace_files.read("state")
    if state and any(not b.id for b in state_blocks.parse(state)[1]):
        workspace_files.write("state", state)
        touched.append("state")
    for key in ("decisions", "lessons"):
        content = workspace_files.read(key)
        if not content:
            continue
        _pre, raws = _mem._split_entries(content, _mem.ENTRY_MARKERS[key])
        if all(_mem.entry_id(r) for r in raws):
            continue
        stamped, _warns = _mem.stamp(content, content, key, today=today)
        if stamped != content:
            workspace_files.write(key, stamped)
            touched.append(key)
    return touched


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are the memory editor for an AI agent's project. You tidy its "
    "memory files by choosing entries to archive or merge, BY ID. You never "
    "add facts. When done, reply with ONLY the JSON object described."
)


def _size_lines(contents: dict[str, str]) -> list[str]:
    out = []
    for key in LIVE_KEYS:
        raw = contents.get(key) or ""
        now = int(_mem.est_tokens(_mem._budget_text(raw, key)))
        soft = _mem.FILE_BUDGETS[key]["soft"]
        target = _mem.consolidation_target(key)
        if now > soft:
            verdict = f"OVER its soft budget — bring it to about {target} (cut ~{now - target})"
        elif now > target:
            verdict = "under the soft budget; tidy only what is clearly stale"
        else:
            verdict = "within target — leave it unless something is plainly done"
        out.append(
            f"- {FILE_BY_KEY[key]}: {now} tokens (soft budget {soft}, target {target}) — {verdict}"
        )
    return out


def _pointer_summary(contents: dict[str, str]) -> list[str]:
    out = []
    for key in EDITABLE_KEYS:
        n = sum(
            1 for ln in (contents.get(key) or "").split("\n")
            if state_blocks.pointer_date(ln)
        )
        if n:
            out.append(f"- {FILE_BY_KEY[key]}: {n} pointer line(s) to archived entries")
    return out


def build_prompt(
    contents: dict[str, str],
    *,
    today: str,
    since: datetime | None,
    sessions: list[dict],
) -> str:
    since_txt = since.isoformat(timespec="minutes") if since else "never"
    if sessions:
        session_lines = "\n".join(
            f"- {s.get('session_uuid')}  last_activity={s.get('last_activity_at')}  "
            f"{(s.get('name') or '')[:80]}"
            for s in sessions
        )
    else:
        session_lines = "(none)"
    cutoff = (
        datetime.fromisoformat(today) - timedelta(days=_mem.STATE_PROBATION_DAYS)
    ).date().isoformat()
    files = []
    for key in LIVE_KEYS:
        raw = _mem.strip_format_header(contents.get(key) or "").strip()
        files.append(f"=== {FILE_BY_KEY[key]} ===\n{raw or '(empty)'}")
    pointers = _pointer_summary(contents)
    pointer_block = (
        "\nEXISTING POINTERS (already archived — not entries, leave them):\n"
        + "\n".join(pointers) + "\n"
    ) if pointers else ""
    # INDEX is the one file the editor may rewrite, so its format drift rides
    # along here (report-only lint; never a trigger of its own).
    drift = _mem.shape_report(contents.get("index"), "index")
    formatting_block = (
        f"\nINDEX FORMATTING TO FIX (if you rewrite INDEX): {drift}\n"
        f"  Contract: {_mem.FORMAT_HEADERS['index']}\n"
    ) if drift else ""

    return f"""Today is {today}. Tidy this project's memory so each file gets back toward its target.

HOW EDITS WORK
The daemon moves the exact bytes by id; you only choose. Every choice leaves a
pointer line `[archived {today} id:<id>] <your pointer> → <ARCHIVE_FILE>` in the
live file, so nothing is lost and a future session can recall the entry by id.
Ids: PROJECT_STATE bullets carry `<!--mem id:… created:…-->` on the line under
them; DECISIONS/LESSONS entries carry `<!--mem id:…-->` on their title line.

SIZES (exact counts, do not estimate):
{chr(10).join(_size_lines(contents))}
{pointer_block}{formatting_block}
RULES
1. Archive what is no longer live: finished work, superseded status, dormant
   threads, decisions long superseded in practice. Keep what a future session
   needs now — including every line that names a file as the source of truth
   for a kind of output that is still produced.
2. The sessions below are EVIDENCE of whether a line is still live, done, or
   replaced. They are never a source of new facts: add nothing that is not
   already written in these files.
3. Nothing is ever deleted — every removal is an archive move with a pointer.
   A pointer is one short line telling a future reader what left and when it
   is worth going back for (e.g. "v0.6 signing decisions — read before
   installer work"). Give the pointer text only; the daemon adds the date,
   the id and the archive file.
4. merge: only for entries that say the same thing twice or where one
   supersedes another. The merged text may only condense what the originals
   say and must be shorter than them combined. For PROJECT_STATE it is one
   bullet (`- …`); for DECISIONS one `## …` entry; for LESSONS one `N. …`
   entry.
5. PROJECT_STATE bullets created on or after {cutoff} are recent: archive or
   merge one only when a session shows it is done or replaced.
6. Never choose an entry tagged `tag:pinned`.
7. INDEX.md is a navigation map. Return a complete new INDEX.md as "index"
   only if it is over budget, stale, or has formatting to fix; keep every
   *_ARCHIVE.md line.
8. If nothing needs to change, return {{}}.

TOOLS (read-only; at most {EDITOR_MAX_CALLS - 1} tool rounds): read, grep,
list_sessions, read_session. The four live files are already below in full,
with their ids — do not read them again. Use the tools for evidence: the
sessions (read_session with `since`, `grep` and a small `limit`) and the
archives. Never read an archive file whole — grep it for an id or a topic and
read only the matching lines with offset/limit. Fewer, targeted calls are
better; answer as soon as you can decide.

LAST EDITOR RUN: {since_txt}
SESSIONS ACTIVE SINCE THEN (newest first):
{session_lines}

OUTPUT — ONLY this JSON (omit keys you do not use):
{{"archive": [{{"file": "PROJECT_STATE.md", "id": "…", "pointer": "…"}}],
 "merge": [{{"file": "DECISIONS.md", "ids": ["…", "…"], "text": "…", "pointer": "…"}}],
 "index": "<complete INDEX.md>"}}

--- THE LIVE FILES, IN FULL ---
{chr(10).join(files)}
"""


# ---------------------------------------------------------------------------
# The read-only tool set
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


def build_tools(workspace: str, list_sessions) -> dict:
    from agent_os.agent.tools.grep_tool import GrepTool
    from agent_os.agent.tools.read import ReadTool
    from agent_os.agent.tools.sessions_tool import ListSessionsTool, ReadSessionTool

    rows = list_sessions or (lambda: _disk_sessions(workspace))
    tools = [
        ReadTool(workspace=workspace),
        GrepTool(workspace=workspace),
        ListSessionsTool(list_sessions=rows, current_session_id=None),
        ReadSessionTool(workspace=workspace, list_sessions=rows),
    ]
    return {t.name: t for t in tools}


def sessions_since(list_sessions, workspace: str, since: datetime | None) -> list[dict]:
    rows_fn = list_sessions or (lambda: _disk_sessions(workspace))
    try:
        rows = [r for r in rows_fn() if isinstance(r, dict) and r.get("session_uuid")]
    except Exception:  # noqa: BLE001 — evidence is optional
        logger.warning("memory editor: could not list sessions", exc_info=True)
        return []
    floor = since or (datetime.now(timezone.utc) - timedelta(days=_SESSIONS_FALLBACK_DAYS))
    picked = []
    for r in rows:
        at = _parse_iso(r.get("last_activity_at"))
        if at is not None and at >= floor:
            picked.append((at, r))
    picked.sort(key=lambda p: p[0], reverse=True)
    return [r for _at, r in picked[:_MAX_SESSIONS_LISTED]]


def _raw_message(response) -> dict:
    raw = getattr(response, "raw_message", None)
    return raw if isinstance(raw, dict) else {}


def _normalize_calls(response) -> list[dict]:
    raw = _raw_message(response).get("tool_calls") or []
    if not raw:
        fallback = getattr(response, "tool_calls", None)
        raw = fallback if isinstance(fallback, list) else []
    calls = []
    for tc in raw:
        if not isinstance(tc, dict):
            tc = tc.model_dump() if hasattr(tc, "model_dump") else {
                "id": getattr(tc, "id", ""),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                },
            }
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        args = fn.get("arguments", tc.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (json.JSONDecodeError, ValueError):
                args = {}
        calls.append({
            "id": tc.get("id") or f"call_{secrets.token_hex(4)}",
            "name": name,
            "arguments": args if isinstance(args, dict) else {},
        })
    return calls


# File tools (ripgrep, disk reads) run off the event loop.
_THREAD_SAFE_TOOLS = frozenset({"read", "grep"})


async def _run_tool(tools: dict, name: str, arguments: dict) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"Error: '{name}' is not available to the memory editor (read-only tools: {', '.join(tools)})."
    try:
        if name in _THREAD_SAFE_TOOLS:
            result = await asyncio.to_thread(tool.execute, **arguments)
        else:
            # The session tools read the agent manager's live registries,
            # which only the event loop may touch.
            result = tool.execute(**arguments)
    except Exception as e:  # noqa: BLE001 — a tool error is just a result
        return f"Error: {e}"
    content = result.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)[:EDITOR_TOOL_RESULT_CHARS]
    if len(content) > EDITOR_TOOL_RESULT_CHARS:
        content = (
            content[:EDITOR_TOOL_RESULT_CHARS]
            + f"\n[... cut at {EDITOR_TOOL_RESULT_CHARS} chars — narrow the query ...]"
        )
    return content


def parse_choices(text: str | None) -> dict | None:
    """The editor's JSON reply, tolerant of fences and surrounding prose."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


# ---------------------------------------------------------------------------
# Applying choices (pure planning + an await-free write step)
# ---------------------------------------------------------------------------

@dataclass
class FilePlan:
    key: str
    content: str                       # new live content
    archived: str                      # text appended to the archive ("" = none)
    applied: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


# Models echo the pointer format back: "<text> → PROJECT_STATE_ARCHIVE.md".
_TRAILING_ARCHIVE_RE = re.compile(r"\s*(?:→|->)\s*[A-Z_]+_ARCHIVE\.md\s*$")


def _clean_pointer(pointer, fallback: str) -> str:
    text = _TRAILING_ARCHIVE_RE.sub("", " ".join(str(pointer or "").split()))[:200]
    return text or _mem.pointer_summary(fallback)


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _key_for(file_value) -> str | None:
    name = str(file_value or "").strip()
    name = os.path.basename(name)
    if name in _KEY_BY_NAME:
        return _KEY_BY_NAME[name]
    return name.lower() if name.lower() in FILE_BY_KEY else None


def group_choices(choices: dict) -> tuple[dict[str, list[dict]], list[str]]:
    """Split raw choices per editable file; returns (per_key, rejected)."""
    per_key: dict[str, list[dict]] = {k: [] for k in EDITABLE_KEYS}
    rejected: list[str] = []
    for kind in ("merge", "archive"):
        for c in _as_list(choices.get(kind)):
            if not isinstance(c, dict):
                rejected.append(f"{kind}: not an object")
                continue
            key = _key_for(c.get("file"))
            if key not in EDITABLE_KEYS:
                rejected.append(f"{kind}: file {c.get('file')!r} cannot be edited by id")
                continue
            per_key[key].append({**c, "_kind": kind})
    return per_key, rejected


def _state_merge_lines(text: str, prefix: str) -> list[str]:
    lines = [ln.rstrip() for ln in str(text).strip().split("\n") if ln.strip()]
    first = lines[0].lstrip()
    if not re.match(r"^(?:-|\d+[.)])\s+\S", first):
        first = prefix + first
    rest = [ln if ln[:1] in (" ", "\t") else "  " + ln.lstrip() for ln in lines[1:]]
    return [first, *rest]


def plan_state(content: str, items: list[dict], *, today: str, archive_filename: str) -> FilePlan:
    lines, blocks = state_blocks.parse(content)
    by_id = {b.id: b for b in blocks if b.id}
    plan = FilePlan("state", content, "")
    claimed: set[str] = set()
    replaced: dict[int, tuple[int, list[str]]] = {}
    archived_parts: list[tuple[int, str]] = []

    for item in sorted(items, key=lambda c: c["_kind"] != "merge"):
        if item["_kind"] == "merge":
            ids = [str(i) for i in _as_list(item.get("ids"))]
            text = str(item.get("text") or "").strip()
            origs = [by_id.get(i) for i in ids]
            if not ids or not text or any(b is None for b in origs) or claimed & set(ids) \
                    or len(set(ids)) != len(ids):
                plan.rejected.append(f"merge {ids}: unknown, repeated or already-used id, or empty text")
                continue
            total = sum(len(b.text) for b in origs)
            if len(" ".join(text.split())) > total * 1.1 + 20:
                plan.rejected.append(f"merge {ids}: merged text is longer than the originals")
                continue
            origs.sort(key=lambda b: b.start)
            ptr = _mem.pointer_line(
                today, [b.id for b in origs],
                _clean_pointer(item.get("pointer"), "merged into the line above"),
                archive_filename,
            )
            first = origs[0]
            replaced[first.start] = (first.end, [*_state_merge_lines(text, first.prefix), ptr])
            for b in origs[1:]:
                replaced[b.start] = (b.end, [])
            summary = _mem.pointer_summary(text, 80)
            archived_parts.append((
                first.start,
                f"[merged {today} into: {summary}]\n" + "\n".join(b.raw for b in origs),
            ))
            claimed.update(ids)
            plan.applied.append({"kind": "merge", "file": FILE_BY_KEY["state"], "ids": ids})
        else:
            eid = str(item.get("id") or "")
            b = by_id.get(eid)
            if b is None or eid in claimed:
                plan.rejected.append(f"archive {eid!r}: unknown or already-used id")
                continue
            ptr = _mem.pointer_line(
                today, [eid], _clean_pointer(item.get("pointer"), b.text), archive_filename)
            replaced[b.start] = (b.end, [ptr])
            archived_parts.append((b.start, b.raw))
            claimed.add(eid)
            plan.applied.append({"kind": "archive", "file": FILE_BY_KEY["state"], "id": eid})

    if not replaced:
        return plan
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in replaced:
            end, repl = replaced[i]
            out.extend(repl)
            i = end + 1
            continue
        out.append(lines[i])
        i += 1
    plan.content = "\n".join(out)
    archived_parts.sort(key=lambda p: p[0])
    plan.archived = "\n".join(t for _pos, t in archived_parts)
    return plan


def _fresh_durable_id(title: str, key: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", _mem._norm_title(title, key)).strip("-")[:32] or "entry"
    while True:
        eid = f"{base}-m{secrets.token_hex(2)}"
        if eid not in taken:
            return eid


def plan_durable(key: str, content: str, items: list[dict], *, today: str,
                 archive_filename: str) -> FilePlan:
    marker = _mem.ENTRY_MARKERS[key]
    pre, raws = _mem._split_entries(content, marker)
    plan = FilePlan(key, content, "")
    by_id: dict[str, int] = {}
    pinned: set[str] = set()
    for i, raw in enumerate(raws):
        first_line, _rest = _mem._first_line_split(raw)
        _title, meta = _mem._parse_meta(first_line)
        if meta.get("id"):
            by_id[meta["id"]] = i
            if meta.get("tag") == "pinned":
                pinned.add(meta["id"])
    new_raws = list(raws)
    claimed: set[str] = set()
    archived_parts: list[tuple[int, str]] = []

    def _leave(i: int, head_lines: list[str]) -> str:
        """Put ``head_lines`` (+ the chunk's own pointer lines) where entry
        ``i`` was; return the entry's body for the archive."""
        body, pointers = _mem.split_out_pointer_lines(raws[i])
        trailing = raws[i][len(raws[i].rstrip("\n")):] or "\n"
        new_raws[i] = "\n".join([*head_lines, *[p for p in pointers if p.strip()]]) + trailing
        return body.rstrip("\n") + "\n"

    for item in sorted(items, key=lambda c: c["_kind"] != "merge"):
        if item["_kind"] == "merge":
            ids = [str(i) for i in _as_list(item.get("ids"))]
            text = str(item.get("text") or "").strip()
            idxs = [by_id.get(i) for i in ids]
            if not ids or not text or any(x is None for x in idxs) or claimed & set(ids) \
                    or pinned & set(ids) or len(set(ids)) != len(ids):
                plan.rejected.append(f"merge {ids}: unknown, pinned, repeated or already-used id, or empty text")
                continue
            m_pre, m_raws = _mem._split_entries(text + "\n", marker)
            if m_pre.strip() or len(m_raws) != 1:
                plan.rejected.append(f"merge {ids}: text is not exactly one {FILE_BY_KEY[key]} entry")
                continue
            total = sum(len(_mem.split_out_pointer_lines(raws[x])[0]) for x in idxs)
            if len(text) > total * 1.1 + 20:
                plan.rejected.append(f"merge {ids}: merged text is longer than the originals")
                continue
            idxs.sort()
            first_line, rest = _mem._first_line_split(m_raws[0].rstrip("\n") + "\n")
            clean, _meta = _mem._parse_meta(first_line)
            if key == "lessons":
                num = re.match(r"^(\d+)\.", raws[idxs[0]])
                if num:
                    clean = re.sub(r"^\d+\.", f"{num.group(1)}.", clean)
            new_id = _fresh_durable_id(clean, key, set(by_id))
            header = f"{clean} " + _mem._meta_comment(
                {"id": new_id, "created": today, "touched": today})
            merged_entry = (header + rest).rstrip("\n")
            ptr = _mem.pointer_line(
                today, ids, _clean_pointer(item.get("pointer"), "merged into the entry above"),
                archive_filename,
            )
            bodies = [_leave(idxs[0], [merged_entry, "", ptr])]
            bodies += [_leave(x, []) for x in idxs[1:]]
            archived_parts.append((
                idxs[0],
                f"[merged {today} into id:{new_id}]\n\n" + "\n".join(bodies),
            ))
            claimed.update(ids)
            plan.applied.append({"kind": "merge", "file": FILE_BY_KEY[key], "ids": ids, "new_id": new_id})
        else:
            eid = str(item.get("id") or "")
            i = by_id.get(eid)
            if i is None or eid in claimed or eid in pinned:
                plan.rejected.append(f"archive {eid!r}: unknown, pinned or already-used id")
                continue
            title = _mem._parse_meta(_mem._first_line_split(raws[i])[0])[0]
            ptr = _mem.pointer_line(
                today, [eid], _clean_pointer(item.get("pointer"), title), archive_filename)
            archived_parts.append((i, _leave(i, [ptr])))
            claimed.add(eid)
            plan.applied.append({"kind": "archive", "file": FILE_BY_KEY[key], "id": eid})

    if not archived_parts:
        return plan
    plan.content = pre + "".join(new_raws)
    archived_parts.sort(key=lambda p: p[0])
    plan.archived = "\n".join(t for _pos, t in archived_parts)
    return plan


def _index_ok(new_index, old_index: str) -> bool:
    return isinstance(new_index, str) and bool(new_index.strip()) and new_index.strip() != (old_index or "").strip()


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


@dataclass
class EditorResult:
    ok: bool                            # the editor produced parseable choices
    outcome: str                        # "edited" | "no_change" | "failed"
    calls: int = 0
    choices: dict | None = None
    applied: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    backup_dir: str | None = None
    error: str | None = None


def apply_choices(
    workspace_files, choices: dict, *, baselines: dict[str, int | None],
    snapshot: dict[str, str], today: str, project_id: str = "",
) -> EditorResult:
    """Apply the editor's choices. NO awaits in here — all or per-file nothing.

    Per file: plan against the exact bytes the editor saw; if the file changed
    since (OCC), skip it; otherwise append the moved entries to the archive
    FIRST, then write the smaller live file.
    """
    from agent_os.agent import workspace_files as _wf

    result = EditorResult(ok=True, outcome="no_change", choices=choices)
    per_key, rejected = group_choices(choices)
    result.rejected.extend(rejected)

    plans: list[FilePlan] = []
    for key in EDITABLE_KEYS:
        if not per_key[key]:
            continue
        archive_key = _mem.ARCHIVE_OF[key]
        archive_filename = _wf.FILE_NAMES[archive_key]
        if key == "state":
            plan = plan_state(snapshot.get(key) or "", per_key[key],
                              today=today, archive_filename=archive_filename)
        else:
            plan = plan_durable(key, snapshot.get(key) or "", per_key[key],
                                today=today, archive_filename=archive_filename)
        result.rejected.extend(plan.rejected)
        if plan.archived:
            plans.append(plan)

    # INDEX first: the moves below append archive pointers to INDEX, and a
    # rewrite checked after them would read the editor's own write as a
    # concurrent one.
    new_index = choices.get("index")
    if new_index is not None:
        if not _index_ok(new_index, snapshot.get("index") or ""):
            result.rejected.append("index: empty or unchanged")
        elif not _wf._occ_unchanged(workspace_files, "index", baselines.get("index"),
                                    project_id=project_id):
            result.skipped_files.append(FILE_BY_KEY["index"])
        else:
            workspace_files.write("index", _keep_archive_lines(
                new_index, snapshot.get("index") or ""))
            for archive_key in ("decisions_archive", "lessons_archive", "state_archive"):
                if workspace_files.exists(archive_key):
                    _wf._ensure_index_archive_pointer(
                        workspace_files, _wf.FILE_NAMES[archive_key])
            result.applied.append({"kind": "index", "file": FILE_BY_KEY["index"]})

    for plan in plans:
        if not _wf._occ_unchanged(workspace_files, plan.key, baselines.get(plan.key),
                                  project_id=project_id):
            result.skipped_files.append(FILE_BY_KEY[plan.key])
            continue
        archive_key = _mem.ARCHIVE_OF[plan.key]
        try:
            _wf._append_archive_section(
                workspace_files, archive_key, f"## [archived {today}]", plan.archived)
            workspace_files.write(plan.key, plan.content)
        except OSError as e:
            logger.warning("memory editor: could not apply %s (%s)", plan.key, e)
            result.skipped_files.append(FILE_BY_KEY[plan.key])
            continue
        _wf._ensure_index_archive_pointer(workspace_files, _wf.FILE_NAMES[archive_key])
        result.applied.extend(plan.applied)

    if result.applied:
        result.outcome = "edited"
    return result


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

# Provider errors worth one more try: gateway hiccups (5xx, 429, dropped
# connections). An auth or request error would fail the same way again.
_RETRY_DELAY_S = 2.0


async def _complete_once_or_retry(llm, messages, tools, deadline: float):
    """One model call, bounded by the per-call and whole-run deadlines, retried
    once on a transient provider error (a single gateway 500 used to discard
    every call the run had already made)."""
    from agent_os.agent.providers.types import ErrorCategory, LLMError

    for attempt in (1, 2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("editor exceeded its total budget")
        try:
            return await asyncio.wait_for(
                llm.complete(messages, tools=tools),
                timeout=min(EDITOR_CALL_TIMEOUT_S, remaining),
            )
        except LLMError as e:
            if attempt == 2 or e.category == ErrorCategory.ABORT:
                raise
            logger.info("memory editor: transient provider error (%s); retrying once", e)
            await asyncio.sleep(_RETRY_DELAY_S)


async def run_editor(
    workspace_files, llm, *, list_sessions=None, project_id: str = "",
    today: str | None = None, now: datetime | None = None,
) -> EditorResult:
    """One editor pass: backup → ids → read-only loop → apply. Never raises
    for model/tool trouble — that comes back as ``ok=False``."""
    now = now or datetime.now(timezone.utc)
    today = today or datetime.now().date().isoformat()
    orbital_dir = workspace_files.dir

    backup_dir = backup_live_files(orbital_dir, now=now)
    if backup_dir is None and any(workspace_files.exists(k) for k in LIVE_KEYS):
        return EditorResult(ok=False, outcome="failed", error="backup failed")
    normalise_ids(workspace_files, today=today)

    from agent_os.agent import workspace_files as _wf
    snapshot = {k: workspace_files.read(k) or "" for k in LIVE_KEYS}
    baselines = {k: _wf._stat_mtime_ns(workspace_files._file_path(k)) for k in LIVE_KEYS}
    since = watermark(orbital_dir)
    sessions = sessions_since(list_sessions, workspace_files.workspace, since)
    tools = build_tools(workspace_files.workspace, list_sessions)
    schemas = [t.schema() for t in tools.values()]

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": build_prompt(
            snapshot, today=today, since=since, sessions=sessions)},
    ]
    calls = 0
    choices: dict | None = None
    deadline = time.monotonic() + EDITOR_TOTAL_TIMEOUT_S
    try:
        while calls < EDITOR_MAX_CALLS:
            last = calls == EDITOR_MAX_CALLS - 1
            if last:
                messages.append({"role": "user", "content": (
                    "Tool budget used up. Reply now with ONLY the JSON choices.")})
            response = await _complete_once_or_retry(
                llm, messages, None if last else schemas, deadline)
            calls += 1
            tool_calls = [] if last else _normalize_calls(response)
            if tool_calls:
                assistant = {
                    "role": "assistant",
                    "content": response.text if isinstance(response.text, str) else "",
                    "tool_calls": [{
                        "id": c["id"], "type": "function",
                        "function": {"name": c["name"],
                                     "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                    } for c in tool_calls],
                }
                reasoning = _raw_message(response).get("reasoning_content")
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
                for c in tool_calls:
                    logger.info("memory editor tool call: %s %s", c["name"],
                                json.dumps(c["arguments"], ensure_ascii=False)[:300])
                    messages.append({
                        "role": "tool", "tool_call_id": c["id"],
                        "content": await _run_tool(tools, c["name"], c["arguments"]),
                    })
                continue
            text = response.text if isinstance(response.text, str) else None
            choices = parse_choices(text)
            if choices is not None:
                break
            messages.append({"role": "assistant", "content": text or ""})
            messages.append({"role": "user", "content": (
                "That was not a JSON object. Reply with ONLY the JSON choices.")})
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — the floor still runs; report and move on
        logger.warning("memory editor: model call failed after %d call(s) (%s)", calls, e)
        return EditorResult(ok=False, outcome="failed", calls=calls,
                            backup_dir=backup_dir, error=str(e))

    if choices is None:
        logger.warning("memory editor: no usable JSON after %d call(s)", calls)
        return EditorResult(ok=False, outcome="failed", calls=calls,
                            backup_dir=backup_dir, error="no JSON choices")

    result = apply_choices(workspace_files, choices, baselines=baselines,
                           snapshot=snapshot, today=today, project_id=project_id)
    result.calls = calls
    result.backup_dir = backup_dir
    logger.info(
        "memory editor: %s in %d call(s) — applied=%s rejected=%s skipped=%s",
        result.outcome, calls, json.dumps(result.applied, ensure_ascii=False),
        result.rejected, result.skipped_files,
    )
    _write_run_record(backup_dir, result)
    return result


def _write_run_record(backup_dir: str | None, result: EditorResult) -> None:
    """Keep the run's choices next to its backup, for audit."""
    if not backup_dir:
        return
    try:
        with open(os.path.join(backup_dir, "editor-run.json"), "w", encoding="utf-8") as f:
            json.dump({
                "outcome": result.outcome,
                "calls": result.calls,
                "choices": result.choices,
                "applied": result.applied,
                "rejected": result.rejected,
                "skipped_files": result.skipped_files,
            }, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 180: the on-disk half of moving a session to another project.

The owner's definition of a move: the session's transcript moves, its
sub-agent transcripts move, and so does whatever the transcript needs to
render (its images). The prompt is rebuilt from the new project. No artifact,
workspace file or memory file is touched in either project.

Order of writes (each step is harmless if the daemon dies right after it,
and moving again finishes the job):

1. read A's log under its file lock (held to the end);
2. copy the image blobs and the sub-agent transcripts the rows name into B
   (add-only: an existing file in B is never overwritten);
3. write B's log whole (temp file + ``os.replace``), then the session's own
   fanout worker logs; a B log that is this very move's earlier copy is kept;
4. retire A's log (and the worker logs) into ``A/orbital/sessions/.moved/``
   under a name that says where it went: ``{stem}.to-{B}.{UTC}.jsonl``.

Nothing is ever deleted. ``.moved/`` is not listed by any version (listing
takes ``*.jsonl`` directly in ``sessions/``), so a v0.22.0 daemon sees the
session in B only.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MOVED_DIRNAME = ".moved"

# A session id that can be a file stem: stems carry the sanitized project
# name, which may be any script (``alpha_预算_e571b947``), so only path
# separators, glob characters, control characters and ".." are refused.
_UNSAFE_ID_RE = re.compile(r"[\\/*?\[\]\x00-\x1f]")


class MoveError(Exception):
    """A refused move; ``code`` is the route's error detail."""

    code = "move_failed"
    status = 400

    def __init__(self, code: str | None = None, status: int | None = None):
        super().__init__(code or self.code)
        if code:
            self.code = code
        if status:
            self.status = status


def safe_session_id(session_id: str | None) -> bool:
    return (bool(session_id) and not _UNSAFE_ID_RE.search(session_id)
            and ".." not in session_id and not session_id.startswith("."))


def utc_stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")


def retired_name(stem: str, to_project_id: str, stamp: str) -> str:
    return f"{stem}.to-{to_project_id}.{stamp}.jsonl"


def moved_destination(sessions_dir: str, stem: str) -> str | None:
    """Where a session that left this project went: the project id named by
    the newest ``.moved/{stem}.to-{B}.{UTC}.jsonl``, else None. Only
    meaningful when ``sessions_dir`` has no live ``{stem}.jsonl``."""
    if not safe_session_id(stem):
        return None
    moved_dir = os.path.join(sessions_dir, MOVED_DIRNAME)
    if not os.path.isdir(moved_dir):
        return None
    prefix = f"{stem}.to-"
    best: tuple[str, str] | None = None
    for path in glob.glob(os.path.join(glob.escape(moved_dir), prefix + "*.jsonl")):
        rest = os.path.basename(path)[len(prefix):-len(".jsonl")]
        project_id, _, stamp = rest.rpartition(".")
        if not project_id or not stamp:
            continue
        if best is None or stamp > best[0]:
            best = (stamp, project_id)
    return best[1] if best else None


def _copy_add_only(src: str, dst: str) -> bool:
    """Copy ``src`` to ``dst`` unless ``dst`` exists (never overwritten).
    Atomic: a temp file beside ``dst``, then ``os.replace``. True if copied."""
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".move-tmp"
    try:
        shutil.copyfile(src, tmp)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        if os.path.exists(dst):
            os.remove(tmp)
            return False
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return True


def _is_prefix(short: str, long: str) -> bool:
    """True when file ``short``'s bytes are the start of file ``long``'s."""
    try:
        if os.path.getsize(short) > os.path.getsize(long):
            return False
        with open(short, "rb") as a, open(long, "rb") as b:
            while True:
                chunk = a.read(1 << 20)
                if not chunk:
                    return True
                if b.read(len(chunk)) != chunk:
                    return False
    except OSError:
        return False


def _copy_transcript(src: str, dst: str, *, tag: str) -> str | None:
    """Copy a sub-agent transcript into B and return the path the rows must
    name, or None when there is nothing to point at.

    A per-handle transcript is shared by every session of a project and keeps
    growing, so B may already hold an older copy (an earlier move) or a newer
    one (a move back). Nothing is ever overwritten with less:
    * missing in B → copied;
    * B's copy is the start of A's file (A grew since) → replaced by A's;
    * A's file is the start of B's (or equal) → B's is kept;
    * they differ otherwise → A's is copied beside it under a new name."""
    if not os.path.isfile(src):
        return dst if os.path.isfile(dst) else None
    if not os.path.exists(dst):
        _copy_add_only(src, dst)
        return dst
    if _is_prefix(src, dst):
        return dst
    if _is_prefix(dst, src):
        tmp = dst + ".move-tmp"
        shutil.copyfile(src, tmp)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)  # strictly more of the same file: nothing lost
        return dst
    root, ext = os.path.splitext(dst)
    alt = f"{root}.{tag}{ext}"
    _copy_add_only(src, alt)
    return alt


def _under(path: str, root: str) -> str | None:
    """``path`` relative to ``root`` when it lies inside it (realpath on both
    sides, trailing separator), else None."""
    try:
        real = os.path.realpath(path)
        base = os.path.realpath(root)
    except (OSError, ValueError):
        return None
    if real == base or not real.startswith(base.rstrip(os.sep) + os.sep):
        return None
    return os.path.relpath(real, base)


def _worker_logs_of(sessions_dir: str, ids: set[str]) -> list[str]:
    """The session's own fanout worker logs: ``session_kind: worker`` meta
    naming one of ``ids`` as ``parent_session_id`` (head read only)."""
    out: list[str] = []
    try:
        names = sorted(os.listdir(sessions_dir))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(sessions_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head = [fh.readline() for _ in range(4)]
        except OSError:
            continue
        for raw in head:
            if '"session_kind"' not in raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (isinstance(rec, dict) and rec.get("event") == "session_kind"
                    and rec.get("kind") == "worker"
                    and rec.get("parent_session_id") in ids):
                out.append(path)
                break
    return out


def checkpoint(step: str) -> None:
    """Test seam: called after each write boundary (``copied``, ``written``,
    ``workers_written``, ``retired``). A no-op in the product."""


def transfer_session_files(*, a_workspace: str, b_workspace: str, a_path: str,
                           stem: str, ids: set[str], from_info: dict,
                           to_info: dict, now: datetime | None = None) -> dict:
    """Steps 4-7 of spec 180 §5.4 for an already-stopped session. Returns
    ``{"b_path", "retired_path", "resumed": bool}``. Raises ``MoveError``
    (``destination_exists``, 409) when B holds a different log of this id."""
    from agent_os.agent import blob_store
    from agent_os.agent.project_paths import ProjectPaths
    from agent_os.agent.session import (
        SESSION_MOVED_EVENT, image_blob_refs_in, move_session_lines,
        read_session_start, transcript_paths_in, write_new_session_file,
    )
    from agent_os.utils.file_lock import retry_session_lock_sync, session_lock
    from agent_os.utils.text_io import read_text_lenient_raising

    now = now or datetime.now(timezone.utc)
    a_paths, b_paths = ProjectPaths(a_workspace), ProjectPaths(b_workspace)
    a_sessions, b_sessions = a_paths.sessions_dir, b_paths.sessions_dir
    a_orbital = os.path.dirname(a_sessions)
    b_orbital = os.path.dirname(b_sessions)
    a_sub, b_sub = a_paths.sub_agents_dir, b_paths.sub_agents_dir
    b_path = os.path.join(b_sessions, f"{stem}.jsonl")
    stamp = utc_stamp(now)
    lock = session_lock(a_path)
    retry_session_lock_sync(lock.acquire)
    try:
        text = read_text_lenient_raising(a_path)

        # Step 5: what the transcript needs, before B's log can name it.
        mapping: dict[str, str] = {}
        for path in transcript_paths_in(text):
            rel = _under(path, a_sub)
            if rel is None:
                continue  # not A's own sub-agent transcript: left as it is
            dst = _copy_transcript(path, os.path.join(b_sub, rel),
                                   tag=f"from-{from_info['project_id']}-{stamp}")
            if dst is not None:
                mapping[path] = dst
        for sha256, ext in image_blob_refs_in(text):
            data = blob_store.read_verified(a_orbital, sha256, ext)
            if data is not None:
                blob_store.put_bytes(b_orbital, data, ext)
        checkpoint("copied")

        # Step 6: B's log, whole.
        move_row = {
            "role": "system",
            "source": "daemon",
            "timestamp": now.isoformat(),
            "session_id": (read_session_start(a_path) or {}).get("session_id") or stem,
            "session_uuid": stem,
            "content": _move_note(from_info, to_info),
            "_meta": {
                "event": SESSION_MOVED_EVENT,
                "from": {"project_id": from_info["project_id"],
                         "name": from_info.get("name") or "",
                         "workspace": from_info.get("workspace") or "",
                         # The UI names Quick Tasks through its catalog.
                         **({"is_scratch": True} if from_info.get("is_scratch") else {})},
                "to": {"project_id": to_info["project_id"],
                       "name": to_info.get("name") or "",
                       **({"is_scratch": True} if to_info.get("is_scratch") else {})},
            },
        }
        lines = move_session_lines(
            text, moved_from={"project_id": from_info["project_id"],
                              "at": now.isoformat()},
            move_row=move_row, rewrite_transcript=lambda p: mapping.get(p),
            session_uuid=stem)
        resumed = False
        try:
            write_new_session_file(b_sessions, stem, lines)
        except FileExistsError:
            start = read_session_start(b_path) or {}
            moved_from = start.get("moved_from")
            if not (isinstance(moved_from, dict)
                    and moved_from.get("project_id") == from_info["project_id"]):
                raise MoveError("destination_exists", 409)
            resumed = True  # an interrupted earlier move: finish it
            logger.info("move %s: %s already holds this move's copy; finishing",
                        stem, to_info["project_id"])
        checkpoint("written")

        workers = _worker_logs_of(a_sessions, ids | {stem})
        for wpath in workers:
            wstem = os.path.basename(wpath)[:-len(".jsonl")]
            try:
                wtext = read_text_lenient_raising(wpath)
                write_new_session_file(b_sessions, wstem, [
                    ln.strip() + "\n" for ln in wtext.splitlines() if ln.strip()])
            except FileExistsError:
                pass  # copied by an interrupted earlier run
        checkpoint("workers_written")

        # Step 7: retire A's log (same directory tree: atomic rename).
        moved_dir = os.path.join(a_sessions, MOVED_DIRNAME)
        os.makedirs(moved_dir, exist_ok=True)
        retired = os.path.join(moved_dir, retired_name(stem, to_info["project_id"], stamp))
        os.replace(a_path, retired)
        for wpath in workers:
            wstem = os.path.basename(wpath)[:-len(".jsonl")]
            try:
                os.replace(wpath, os.path.join(
                    moved_dir, retired_name(wstem, to_info["project_id"], stamp)))
            except OSError:
                logger.warning("move %s: worker log %s not retired", stem, wpath,
                               exc_info=True)
        checkpoint("retired")
    finally:
        try:
            lock.release()
        except Exception:
            pass
    return {"b_path": b_path, "retired_path": retired, "resumed": resumed}


def _move_note(from_info: dict, to_info: dict) -> str:
    """The model-facing tail message (English; not UI, never translated)."""
    a_name = from_info.get("name") or from_info["project_id"]
    b_name = to_info.get("name") or to_info["project_id"]
    a_ws = from_info.get("workspace") or ""
    b_ws = to_info.get("workspace") or ""
    return (
        f"The user moved this conversation from the project \"{a_name}\" "
        f"(folder: {a_ws}) to the project \"{b_name}\" (folder: {b_ws}). "
        f"Work in {b_ws} from now on. Files this conversation changed earlier "
        f"stay in {a_ws}, and paths in earlier messages are relative to that "
        f"folder. Do not copy or move them unless the user asks.")

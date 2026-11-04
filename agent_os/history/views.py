# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-side shapes for the history routes (spec 065 §5.5, §5.6).

Everything the UI paints is computed here: versions newest first with
session names, turn grouping and the current flag; the tinted line view;
the list of files one turn changed with the hash check undo needs.
"""
from __future__ import annotations

import bisect
import json
import os

from agent_os.history import diffing, rules
from agent_os.history.store import HistoryStore, iso_to_epoch

# ── sessions / turns ──────────────────────────────────────────────────


class SessionIndex:
    """Session names and turn starts, resolved once per request.

    ``sessions`` maps the F1 session id to ``{name, session_uuid}`` (from
    ``AgentManager.list_sessions``). A turn of session S is the span from
    one user row to the next in S's JSONL (§5.5)."""

    def __init__(self, workspace: str, sessions: dict[str, dict] | None = None):
        self._workspace = workspace
        self._sessions = sessions or {}
        self._starts: dict[str, list[float]] = {}

    def name(self, sid: str) -> str | None:
        return (self._sessions.get(sid) or {}).get("name")

    def turn_starts(self, sid: str) -> list[float]:
        if sid in self._starts:
            return self._starts[sid]
        starts: list[float] = []
        uuid = (self._sessions.get(sid) or {}).get("session_uuid")
        if uuid:
            from agent_os.agent.project_paths import ProjectPaths
            path = os.path.join(ProjectPaths(self._workspace).sessions_dir, f"{uuid}.jsonl")
            try:
                with open(path, "rb") as f:
                    for raw in f:
                        if b'"user"' not in raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except ValueError:
                            continue
                        if not _is_turn_start(rec):
                            continue
                        t = iso_to_epoch(rec.get("timestamp"))
                        if t is not None:
                            starts.append(t)
            except OSError:
                pass
        starts.sort()
        self._starts[sid] = starts
        return starts

    def turn_of(self, sid: str, ts: float) -> dict | None:
        starts = self.turn_starts(sid)
        if not starts:
            return None
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0:
            return None
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        return {"session_id": sid, "from_ts": starts[i], "to_ts": nxt}


def _is_turn_start(rec: dict) -> bool:
    if rec.get("role") != "user" or rec.get("_kind") == "memory_record":
        return False
    return rec.get("source") in (None, "user")


# ── attribution helpers ───────────────────────────────────────────────


def sessions_of(rec: dict) -> list[dict]:
    return list((rec.get("attrib") or {}).get("sessions") or [])


def sole_session(rec: dict) -> str | None:
    ss = sessions_of(rec)
    return ss[0].get("session_id") if len(ss) == 1 else None


def is_ambiguous(rec: dict) -> bool:
    return bool((rec.get("attrib") or {}).get("ambiguous")) or len(sessions_of(rec)) > 1


def present_attrib(rec: dict, idx: SessionIndex) -> dict:
    attrib = dict(rec.get("attrib") or {})
    if attrib.get("sessions"):
        attrib["sessions"] = [
            {**s, "name": idx.name(s.get("session_id"))} for s in attrib["sessions"]
        ]
    if is_ambiguous(rec):
        attrib["ambiguous"] = True
    return attrib


def editor_of(rec: dict, idx: SessionIndex) -> dict:
    """Version-level attribution for the tint (§5.8): ambiguous, outside
    and closed versions are neutral (no agents)."""
    attrib = rec.get("attrib") or {}
    kind = attrib.get("kind")
    agents: list[str] = []
    if kind in ("tool", "running") and not is_ambiguous(rec):
        for s in sessions_of(rec):
            agents.extend(s.get("agents") or [])
    elif kind == "orbital":
        agents = ["orbital"]
    sid = sole_session(rec)
    return {
        "v": rec.get("v"), "ts": rec.get("ts"), "kind": kind,
        "agents": agents, "ambiguous": is_ambiguous(rec),
        "session_id": sid, "session_name": idx.name(sid) if sid else None,
        "neutral": not agents,
    }


# ── list ──────────────────────────────────────────────────────────────


def file_versions(store: HistoryStore, rel: str, idx: SessionIndex) -> list[dict]:
    """Versions of ``rel`` newest first, with ``current``, ``name``,
    ``turn`` and ``group`` (consecutive versions of the file in one turn of
    one session share a group)."""
    recs = store.versions_for(rel)
    out: list[dict] = []
    prev_key = None
    group = None
    for rec in recs:
        sid = sole_session(rec)
        turn = None
        if sid and rec.get("kind") != "baseline":
            t = iso_to_epoch(rec.get("ts"))
            turn = idx.turn_of(sid, t) if t is not None else None
        key = (sid, turn["from_ts"]) if turn else None
        if key is None or key != prev_key:
            group = f"g{rec.get('v')}"
        prev_key = key
        added, removed = rec.get("added"), rec.get("removed")
        out.append({
            "v": rec.get("v"), "ts": rec.get("ts"), "kind": rec.get("kind"),
            "sha": rec.get("sha"), "prev_sha": rec.get("prev_sha"), "size": rec.get("size"),
            "added": added, "removed": removed,
            "lines": (added or 0) + (removed or 0) if added is not None else None,
            "attrib": present_attrib(rec, idx), "ambiguous": is_ambiguous(rec),
            "name": store.names.get((rel, rec.get("sha"))) if rec.get("sha") else None,
            "turn": turn, "group": group, "current": False,
        })
    if out:
        out[-1]["current"] = True
    out.reverse()
    return out


# ── view ──────────────────────────────────────────────────────────────


def _blob(store: HistoryStore, sha: str | None) -> bytes | None:
    if not sha:
        return None
    try:
        return store.get_blob(sha)
    except OSError:
        return None


def version_view(store: HistoryStore, rel: str, idx: SessionIndex, *, sha: str | None = None,
                 v: int | None = None, from_sha: str | None = None,
                 from_v: int | None = None) -> dict | None:
    recs = store.versions_for(rel)
    if not recs:
        return None
    ti = None
    if v is not None:
        ti = next((i for i, r in enumerate(recs) if r.get("v") == v), None)
    elif sha is not None:
        ti = next((i for i in range(len(recs) - 1, -1, -1) if recs[i].get("sha") == sha), None)
    else:
        ti = len(recs) - 1
    if ti is None:
        return None
    target = recs[ti]
    bi = None
    if from_v is not None:
        bi = next((i for i in range(ti) if recs[i].get("v") == from_v), None)
    elif from_sha is not None:
        bi = next((i for i in range(ti - 1, -1, -1) if recs[i].get("sha") == from_sha), None)
    if bi is None:
        bi = ti - 1 if ti > 0 else None
    chain_recs = recs[(bi + 1 if bi is not None else 0):ti + 1]
    base_rec = recs[bi] if bi is not None else None
    if target.get("kind") == "baseline" and base_rec is None:
        base_rec, chain_recs = target, []
    editors = [editor_of(r, idx) for r in chain_recs]
    out = {
        "path": rel, "v": target.get("v"), "sha": target.get("sha"),
        "from_v": base_rec.get("v") if base_rec else None,
        "from_sha": base_rec.get("sha") if base_rec else None,
        "editors": editors, "whole_file_only": rules.is_office(rel),
        "size": target.get("size"),
        "from_size": base_rec.get("size") if base_rec else None,
    }
    if rules.is_office(rel):
        out.update({"lines": [], "first_change": None})
        return out
    base = _blob(store, base_rec.get("sha")) if base_rec else None
    if not chain_recs:
        lines = [{"text": t, "op": "same", "editor": None} for t in diffing.split_lines(base)]
    else:
        lines = diffing.compose_view(base, [_blob(store, r.get("sha")) for r in chain_recs])
    out["lines"] = lines
    out["first_change"] = next((i for i, ln in enumerate(lines) if ln["op"] != "same"), None)
    return out


# ── turn ──────────────────────────────────────────────────────────────


def turn_files(store: HistoryStore, idx: SessionIndex, session_id: str, from_ts: float,
               to_ts: float | None, skip_events=()) -> dict:
    """Files one turn of ``session_id`` changed (§5.6): per path the
    ``before_sha`` undo restores, the turn's last ``turn_sha``, the live
    ``current_sha`` and who changed it later."""
    end = to_ts if to_ts is not None else float("inf")
    by_path: dict[str, list[dict]] = {}
    with store.lock:
        recs = list(store.versions)
        index = {k: dict(v) for k, v in store.index.items()}
    for rec in recs:
        if rec.get("kind") == "baseline":
            continue
        t = iso_to_epoch(rec.get("ts"))
        if t is None or t < from_ts or t >= end:
            continue
        if not any(s.get("session_id") == session_id for s in sessions_of(rec)):
            continue
        by_path.setdefault(rec["path"], []).append(rec)
    files = []
    for path, turn in by_path.items():
        first, last = turn[0], turn[-1]
        before_sha = first.get("prev_sha")
        turn_sha = last.get("sha")
        current_sha = (index.get(path) or {}).get("sha")
        later = [r for r in recs if r["path"] == path and (r.get("v") or 0) > (last.get("v") or 0)
                 and r.get("kind") != "baseline"]
        later_editor = editor_of(later[-1], idx) if later and current_sha != turn_sha else None
        if before_sha is None and turn_sha is not None:
            kind = "create"
        elif turn_sha is None:
            kind = "delete"
        else:
            kind = "modify"
        office = rules.is_office(path)
        if office:
            added = removed = None
        else:
            added, removed = diffing.count_changes(_blob(store, before_sha), _blob(store, turn_sha))
        before_rec = next((r for r in reversed(recs)
                           if r["path"] == path and (r.get("v") or 0) < (first.get("v") or 0)), None)
        files.append({
            "path": path, "kind": kind, "added": added, "removed": removed,
            "lines": None if office else added + removed,
            "before_sha": before_sha, "turn_sha": turn_sha, "current_sha": current_sha,
            "before_size": before_rec.get("size") if before_rec else None,
            "turn_size": last.get("size"),
            "clean": current_sha == turn_sha,
            "later_editor": later_editor,
            "ambiguous": any(is_ambiguous(r) for r in turn),
            "whole_file_only": office,
            "first_v": first.get("v"), "last_v": last.get("v"),
            "editor": editor_of(last, idx),
        })
    files.sort(key=lambda f: f["first_v"] or 0)
    no_history = set()
    for ev in skip_events:
        t = iso_to_epoch(ev.get("ts"))
        if t is None or t < from_ts or t >= end:
            continue
        if any(s.get("session_id") == session_id for s in (ev.get("attrib") or {}).get("sessions") or []):
            no_history.add(ev["path"])
    return {"session_id": session_id, "session_name": idx.name(session_id),
            "from_ts": from_ts, "to_ts": to_ts, "files": files,
            "no_history": len(no_history)}

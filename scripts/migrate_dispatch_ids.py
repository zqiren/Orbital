#!/usr/bin/env python3
"""migrate_dispatch_ids.py — one-shot backfill of dispatch_ids into legacy data.

Commit 6da5f98 (TASK-dispatch-id-pairing) replaced positional pairing between
a sub-agent dispatch marker (in a management session's JSONL) and its
transcript turn with an explicit identity join: the marker's ``_meta``
carries ``dispatch_id`` + ``handle`` + ``transcript_path``, and the
transcript's closing ``turn_complete`` boundary row carries the same
``dispatch_id``. Data written before that commit has neither — the new
renderer (``_interleave_sub_agent_summaries`` in
``agent_os/api/routes/agents_v2.py``) honestly degrades those markers to no
bubble rather than guessing.

This script recovers the (handle, transcript_path) pair from the marker's
prose via ``_SUB_AGENT_DISPATCH_RE`` (still kept in agents_v2.py for exactly
this purpose), then reconstructs the pairing the SAME way the old positional
code did — a deterministic chronological zip of every legacy marker for a
transcript against every unstamped turn boundary in that transcript, in
order — and mints and stamps a dispatch_id onto both sides.

Safety model: fail CLOSED per transcript. If the counts don't reconcile, or
any chronology sanity check fails, NOTHING is stamped for that transcript
and the reason is reported — an unstamped legacy turn renders no bubble
(the pre-migration status quo), but a mis-stamped one would render the
WRONG bubble, which is the exact bug TASK-dispatch-id-pairing fixed. This
script never guesses past a reconciliation failure.

Usage:
    migrate_dispatch_ids.py [--dry-run] [--workspace PATH ...] \\
                             [--projects-json PATH]

    --workspace may be given multiple times. --projects-json points at an
    Orbital ``projects.json`` (a dict of project_id -> project dict, each
    carrying a "workspace" field) to derive workspaces from in bulk. Both
    may be combined; workspaces are de-duplicated.

    --dry-run performs the full scan/reconcile and prints the report but
    writes nothing. Without it, reconciled pairs are stamped and the same
    report is printed afterward. Idempotent: a transcript/marker already
    carrying a dispatch_id is never touched again, so a second run reports
    nothing left to do.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4

from agent_os.api.routes.agents_v2 import _SUB_AGENT_DISPATCH_RE

# Small allowance for clock jitter between the process that wrote the
# dispatch marker and the process that wrote the turn boundary (normally
# the same daemon, but not guaranteed across restarts/host clock drift).
SLACK_SECONDS = 5


def _parse_ts(value) -> "datetime | None":
    """Best-effort ISO-8601 parse; returns None on anything unparseable.

    Mirrors ``sub_agent_transcript._parse_ts`` — kept as a local copy since
    this script must never import daemon runtime modules that could have
    side effects; only the small, already-test-covered regex constant is
    imported from agents_v2.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class MarkerHit:
    """One legacy (no ``_meta.dispatch_id``) dispatch marker found on disk."""

    session_file: str
    session_uuid: str
    line_no: int
    handle: str
    transcript_path: str
    timestamp: "datetime | None"
    raw_timestamp: str
    record: dict


@dataclass
class BoundaryHit:
    """One ``turn_complete`` boundary row found in a transcript file."""

    line_no: int
    timestamp: "datetime | None"
    raw_timestamp: str
    record: dict
    stamped: bool


@dataclass
class TranscriptReport:
    """Outcome of reconciling one transcript's legacy markers vs its turns."""

    transcript_path: str
    status: str = "skipped"  # "stamped" | "skipped"
    reason: str = ""
    markers_considered: int = 0
    unstamped_turns: int = 0
    pairs_stamped: int = 0


@dataclass
class WorkspaceResult:
    workspace: str
    dry_run: bool
    session_files_scanned: int = 0
    legacy_markers_found: int = 0
    already_migrated_markers: int = 0
    transcripts: list = field(default_factory=list)  # list[TranscriptReport]
    total_pairs_stamped: int = 0


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def find_session_files(workspace: str) -> list[str]:
    """All ``orbital/sessions/*.jsonl`` files under a workspace, sorted."""
    sessions_dir = os.path.join(workspace, "orbital", "sessions")
    if not os.path.isdir(sessions_dir):
        return []
    return sorted(glob.glob(os.path.join(sessions_dir, "*.jsonl")))


def scan_legacy_markers(session_files: list[str]) -> tuple[list[MarkerHit], int]:
    """Scan every session file for dispatch markers.

    Returns ``(legacy_hits, already_migrated_count)``. A marker matching
    ``_SUB_AGENT_DISPATCH_RE`` whose ``_meta.dispatch_id`` is already set is
    counted in ``already_migrated_count`` and skipped (idempotency); one
    with no ``_meta.dispatch_id`` at all is a legacy hit needing backfill.
    """
    hits: list[MarkerHit] = []
    already_migrated = 0
    for session_file in session_files:
        stem = os.path.splitext(os.path.basename(session_file))[0]
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line_no, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("role") != "system":
                continue
            content = rec.get("content") or ""
            m = _SUB_AGENT_DISPATCH_RE.match(content)
            if not m:
                continue
            meta = rec.get("_meta") or {}
            if meta.get("dispatch_id"):
                already_migrated += 1
                continue
            raw_ts = rec.get("timestamp") or ""
            hits.append(MarkerHit(
                session_file=session_file,
                session_uuid=stem,
                line_no=line_no,
                handle=m.group(1),
                transcript_path=m.group(2),
                timestamp=_parse_ts(raw_ts),
                raw_timestamp=raw_ts,
                record=rec,
            ))
    return hits, already_migrated


def scan_transcript_boundaries(transcript_path: str) -> "list[BoundaryHit] | None":
    """All ``turn_complete`` rows in a transcript, in file order.

    Returns ``None`` if the file is missing/unreadable (distinct from an
    empty list, which means the file exists but has zero boundaries — a
    flat legacy transcript).
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    boundaries: list[BoundaryHit] = []
    for line_no, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("chunk_type") != "turn_complete":
            continue
        raw_ts = rec.get("timestamp") or ""
        boundaries.append(BoundaryHit(
            line_no=line_no,
            timestamp=_parse_ts(raw_ts),
            raw_timestamp=raw_ts,
            record=rec,
            stamped=bool(rec.get("dispatch_id")),
        ))
    return boundaries


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile_transcript(
    transcript_path: str, markers: list[MarkerHit],
) -> tuple[list[tuple[MarkerHit, BoundaryHit, str]], TranscriptReport]:
    """Reconcile one transcript's legacy markers against its unstamped turns.

    All-or-nothing per transcript: either every marker reconciles with
    exactly one turn boundary (counts match, chronology sanity checks pass),
    in which case a dispatch_id is minted for each pair, or NONE of them are
    stamped and the report explains why. See module docstring for the
    rationale (a mis-stamped turn is worse than an unstamped one).

    Returns ``(pairs, report)`` where ``pairs`` is a list of
    ``(marker, boundary, dispatch_id)`` — empty unless ``report.status ==
    "stamped"``.
    """
    report = TranscriptReport(
        transcript_path=transcript_path, markers_considered=len(markers),
    )

    boundaries = scan_transcript_boundaries(transcript_path)
    if boundaries is None:
        report.reason = "transcript file missing or unreadable"
        return [], report
    if not boundaries:
        report.reason = (
            "flat legacy transcript (no turn_complete boundaries) — "
            "left untouched"
        )
        return [], report

    # Mixed-era guard: dispatch_id stamping ships as a code deploy, so once
    # live every turn from that point on carries one. Unstamped turns must
    # therefore be a strict PREFIX of the file — a stamped turn followed
    # later by an unstamped one is an inconsistency this script refuses to
    # guess through.
    first_stamped_idx = next(
        (i for i, b in enumerate(boundaries) if b.stamped), len(boundaries),
    )
    if any(not b.stamped for b in boundaries[first_stamped_idx:]):
        report.reason = (
            "inconsistent interleaving: an unstamped turn appears after an "
            "already-stamped turn"
        )
        return [], report

    unstamped = boundaries[:first_stamped_idx]
    report.unstamped_turns = len(unstamped)

    if any(m.timestamp is None for m in markers):
        report.reason = "one or more legacy markers have an unparseable timestamp"
        return [], report
    if any(b.timestamp is None for b in unstamped):
        report.reason = "one or more turn boundaries have an unparseable timestamp"
        return [], report

    sorted_markers = sorted(markers, key=lambda m: m.timestamp)

    if len(sorted_markers) != len(unstamped):
        report.reason = (
            f"count mismatch: {len(sorted_markers)} legacy marker(s) vs "
            f"{len(unstamped)} unstamped turn(s)"
        )
        return [], report

    slack = timedelta(seconds=SLACK_SECONDS)
    prev_boundary_ts: "datetime | None" = None
    for idx, (marker, boundary) in enumerate(zip(sorted_markers, unstamped)):
        if marker.timestamp > boundary.timestamp:
            report.reason = (
                f"pair {idx}: marker timestamp {marker.raw_timestamp!r} is "
                f"after its turn's boundary timestamp "
                f"{boundary.raw_timestamp!r}"
            )
            return [], report
        if prev_boundary_ts is not None and marker.timestamp < prev_boundary_ts - slack:
            report.reason = (
                f"pair {idx}: marker timestamp {marker.raw_timestamp!r} "
                f"precedes the previous turn's boundary timestamp "
                f"({prev_boundary_ts.isoformat()}) by more than the "
                f"{SLACK_SECONDS}s jitter allowance"
            )
            return [], report
        prev_boundary_ts = boundary.timestamp

    pairs = [
        (marker, boundary, f"{marker.session_uuid}:{uuid4().hex[:8]}")
        for marker, boundary in zip(sorted_markers, unstamped)
    ]
    report.status = "stamped"
    report.pairs_stamped = len(pairs)
    return pairs, report


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def _apply_edits(edits: dict) -> None:
    """Rewrite the specific lines in each file, atomically, in place.

    ``edits`` maps ``file_path -> {line_no: new_record}``. Every line NOT in
    the edit set is preserved byte-for-byte (read raw, only the targeted
    lines are re-serialized). One read-modify-write per file even when
    multiple transcripts/markers touch the same file.
    """
    for path, line_edits in edits.items():
        if not line_edits:
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line_no, new_record in line_edits.items():
            lines[line_no] = json.dumps(new_record, ensure_ascii=False) + "\n"
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


def migrate_workspace(workspace: str, *, dry_run: bool = False) -> WorkspaceResult:
    """Scan, reconcile, and (unless ``dry_run``) stamp one workspace.

    Deterministic and read-only when ``dry_run=True``: analysis runs in
    full (so the report is identical either way) but ``_apply_edits`` is
    never called.
    """
    session_files = find_session_files(workspace)
    legacy_hits, already_migrated = scan_legacy_markers(session_files)

    by_transcript: dict[str, list[MarkerHit]] = {}
    for hit in legacy_hits:
        by_transcript.setdefault(hit.transcript_path, []).append(hit)

    session_edits: dict[str, dict[int, dict]] = {}
    transcript_edits: dict[str, dict[int, dict]] = {}
    reports: list[TranscriptReport] = []

    for transcript_path in sorted(by_transcript):
        markers = by_transcript[transcript_path]
        pairs, report = reconcile_transcript(transcript_path, markers)
        reports.append(report)
        if not pairs:
            continue

        t_edits = transcript_edits.setdefault(transcript_path, {})
        for marker, boundary, dispatch_id in pairs:
            new_boundary = dict(boundary.record)
            new_boundary["dispatch_id"] = dispatch_id
            t_edits[boundary.line_no] = new_boundary

            new_marker = dict(marker.record)
            merged_meta = dict(new_marker.get("_meta") or {})
            merged_meta["dispatch_id"] = dispatch_id
            merged_meta["handle"] = marker.handle
            merged_meta["transcript_path"] = marker.transcript_path
            new_marker["_meta"] = merged_meta
            s_edits = session_edits.setdefault(marker.session_file, {})
            s_edits[marker.line_no] = new_marker

    total_pairs = sum(r.pairs_stamped for r in reports)

    if not dry_run:
        _apply_edits(session_edits)
        _apply_edits(transcript_edits)

    return WorkspaceResult(
        workspace=workspace,
        dry_run=dry_run,
        session_files_scanned=len(session_files),
        legacy_markers_found=len(legacy_hits),
        already_migrated_markers=already_migrated,
        transcripts=reports,
        total_pairs_stamped=total_pairs,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(result: WorkspaceResult) -> str:
    mode = "DRY RUN" if result.dry_run else "APPLIED"
    lines = [f"== {result.workspace} [{mode}] =="]
    lines.append(
        f"  session files scanned: {result.session_files_scanned}; "
        f"legacy markers found: {result.legacy_markers_found}; "
        f"already migrated: {result.already_migrated_markers}"
    )
    if not result.transcripts:
        lines.append("  no legacy dispatch transcripts to reconcile.")
    for r in result.transcripts:
        if r.status == "stamped":
            lines.append(f"  [stamped] {r.transcript_path}: {r.pairs_stamped} pair(s)")
        else:
            lines.append(
                f"  [skipped] {r.transcript_path}: {r.reason} "
                f"({r.markers_considered} marker(s), {r.unstamped_turns} unstamped turn(s))"
            )
    lines.append(f"  total pairs stamped: {result.total_pairs_stamped}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_workspaces(workspaces: list[str], projects_json: "str | None") -> list[str]:
    resolved = list(workspaces or [])
    if projects_json:
        with open(projects_json, "r", encoding="utf-8") as f:
            projects = json.load(f)
        if isinstance(projects, dict):
            values = list(projects.values())
        elif isinstance(projects, list):
            values = projects
        else:
            values = []
        for p in values:
            ws = (p or {}).get("workspace")
            if ws:
                resolved.append(ws)
    seen: set = set()
    deduped = []
    for ws in resolved:
        if ws not in seen:
            seen.add(ws)
            deduped.append(ws)
    return deduped


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot backfill of dispatch_ids into legacy session/transcript data.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and report only; write nothing to disk.",
    )
    parser.add_argument(
        "--workspace", action="append", dest="workspaces", default=[],
        metavar="PATH", help="Workspace root to migrate (repeatable).",
    )
    parser.add_argument(
        "--projects-json", default=None, metavar="PATH",
        help="Orbital projects.json to derive workspaces from in bulk "
             "(each project's 'workspace' field).",
    )
    args = parser.parse_args(argv)

    workspaces = _resolve_workspaces(args.workspaces, args.projects_json)
    if not workspaces:
        parser.error("no workspaces given — pass --workspace and/or --projects-json")

    any_stamped = False
    for ws in workspaces:
        result = migrate_workspace(ws, dry_run=args.dry_run)
        print(format_report(result))
        if result.total_pairs_stamped:
            any_stamped = True

    if not any_stamped:
        print("Nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

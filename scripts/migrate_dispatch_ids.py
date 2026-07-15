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
this purpose). Before pairing, legacy markers for a transcript are first
collapsed into LOGICAL DISPATCHES: a genuine @mention double-marker (one
"message_routed"-flavored marker + one "user_mention"-flavored marker for
the same handle, minted together within a few seconds of each other — see
``group_logical_dispatches``) counts as ONE dispatch, exactly mirroring what
the live write path does (one dispatch_id, stamped onto both markers).
Logical dispatches are then zipped chronologically, one-to-one and in
order, against every unstamped turn boundary in that transcript — and a
dispatch_id is minted and stamped onto the boundary plus every marker in
the dispatch.

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
# Also reused as the @mention double-marker collapse window (see
# group_logical_dispatches) — the two markers of one physical dispatch are
# minted back-to-back in the same request, well under this.
SLACK_SECONDS = 5

# The two content shapes LifecycleObserver.on_message_routed produces (see
# lifecycle_observer.py). A genuine @mention dispatch fires BOTH — one of
# each — for the SAME physical turn; that's the axis group_logical_dispatches
# collapses on.
_USER_MENTION_PREFIX = '[Sub-agent] User sent @'
_MESSAGE_ROUTED_PREFIX = '[Sub-agent] Message sent to '


def _classify_flavor(content: str) -> str:
    """Which of the two dispatch-marker content shapes this is.

    Only meaningful for content that already matched ``_SUB_AGENT_DISPATCH_RE``
    (both shapes share the same regex, differing only in this leading
    prose). Anything else reports "unknown" — grouping never guesses that
    an unknown-flavored marker is safely mergeable with anything.
    """
    if content.startswith(_USER_MENTION_PREFIX):
        return "user_mention"
    if content.startswith(_MESSAGE_ROUTED_PREFIX):
        return "message_routed"
    return "unknown"


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
    flavor: str  # "user_mention" | "message_routed" | "unknown"
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
class LogicalDispatch:
    """One or two ``MarkerHit`` records that represent the SAME physical
    dispatch — see ``group_logical_dispatches``.

    ``markers`` has length 1 (the common case) or 2 (a collapsed @mention
    double-marker); ``markers[0]`` is always the earlier of the two.
    ``anchor_timestamp`` is that earlier marker's timestamp — the one the
    zip against turn boundaries pairs on.
    """

    markers: list  # list[MarkerHit]
    anchor_timestamp: "datetime"
    handle: str


@dataclass
class TranscriptReport:
    """Outcome of reconciling one transcript's legacy markers vs its turns."""

    transcript_path: str
    status: str = "skipped"  # "stamped" | "skipped"
    reason: str = ""
    markers_considered: int = 0
    logical_dispatches: int = 0
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
                flavor=_classify_flavor(content),
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
# Logical-dispatch grouping
# ---------------------------------------------------------------------------

def group_logical_dispatches(
    markers: list[MarkerHit],
) -> tuple[list[LogicalDispatch], str]:
    """Collapse legacy markers into logical dispatches before pairing them
    against turn boundaries.

    Root cause this fixes: the known @mention double-marker
    (``SubAgentManager.send()``'s own "message_routed"-flavored marker plus
    the API route's separate "user_mention"-flavored marker for the SAME
    physical dispatch — see the docstring on ``_interleave_sub_agent_
    summaries`` in agents_v2.py) is TWO session records but ONE turn. Zipping
    raw markers 1:1 against turn boundaries silently shifts every marker
    after the double-marker by one slot, mis-stamping them with the WRONG
    turn's id — exactly the bug TASK-dispatch-id-pairing fixed on the live
    write path (which mints a single id and stamps it onto both markers).
    This function reproduces that: one "message_routed" + one "user_mention"
    marker, same handle, minted within ``SLACK_SECONDS`` of each other,
    collapse into one ``LogicalDispatch`` anchored at the EARLIER timestamp.
    Every other marker is its own singleton LogicalDispatch.

    Fails closed — returns ``([], reason)`` — the moment a grouping can't be
    proven to be a clean, UNAMBIGUOUS one-of-each pairing. The exclusivity
    rule (code review round 2): a marker's "candidates" are every OTHER
    marker with the same handle, the OPPOSITE flavor, within
    ``SLACK_SECONDS`` of it (either direction — jitter can go either way).
    A marker may only be paired when it has EXACTLY ONE candidate (and,
    since the candidate relation is symmetric, that candidate then also has
    only it). A marker with 2+ candidates means a genuine chain — e.g.
    message_routed@t0, user_mention@t0+4s, message_routed@t0+8s: the middle
    marker has TWO opposite-flavor candidates within a 5s window, and
    picking either one over the other (a naive greedy left-to-right scan
    picks whichever comes first) can silently pair the WRONG two markers —
    the Critical bug's own failure class, just one level up. So ANY marker
    with 2+ candidates fails the WHOLE transcript closed, not just that
    marker's pairing: a chain's ambiguity isn't locally contained to one
    marker, and guessing past it is exactly what this script must never do.
    A marker with ZERO candidates stays its own singleton LogicalDispatch.

    Precondition: every marker's ``timestamp`` is a non-None datetime, and
    all of them share the same tz-awareness (callers — reconcile_transcript
    — must check this first; this function does raw datetime arithmetic
    with no further guarding).
    """
    # Stable sort: exact-timestamp ties fall back to scan order (session
    # files in glob/sorted-filename order, then line order within each
    # file — see scan_legacy_markers/find_session_files). Deterministic
    # across runs on the same data, not otherwise semantically meaningful.
    ordered = sorted(markers, key=lambda m: m.timestamp)
    slack = timedelta(seconds=SLACK_SECONDS)
    n = len(ordered)

    def candidate_indices(i: int) -> list[int]:
        m = ordered[i]
        result = []
        for j in range(n):
            if j == i:
                continue
            other = ordered[j]
            if other.handle != m.handle or other.flavor == m.flavor:
                continue
            if abs(other.timestamp - m.timestamp) <= slack:
                result.append(j)
        return result

    candidates = [candidate_indices(i) for i in range(n)]

    for i, cands in enumerate(candidates):
        if len(cands) >= 2:
            m = ordered[i]
            return [], (
                f"ambiguous marker chain: handle {m.handle!r} marker at "
                f"{m.raw_timestamp!r} has {len(cands)} opposite-flavor "
                f"candidates within {SLACK_SECONDS}s — cannot prove a "
                f"unique double-marker pairing"
            )

    used = [False] * n
    dispatches: list[LogicalDispatch] = []
    for i, marker in enumerate(ordered):
        if used[i]:
            continue
        cands = candidates[i]
        if cands:
            j = cands[0]
            # Mutual exclusivity is guaranteed by the global "no marker has
            # 2+ candidates" check above (the candidate relation is
            # symmetric: j is i's only candidate iff i is j's only
            # candidate) — asserted defensively rather than silently
            # trusted, so a violation reports instead of mis-pairing.
            if used[j] or candidates[j] != [i]:
                return [], (
                    f"ambiguous marker chain: handle {marker.handle!r} "
                    f"pairing at {marker.raw_timestamp!r} failed a mutual-"
                    f"exclusivity check"
                )
            used[i] = True
            used[j] = True
            # ordered is sorted ascending and i < j here (candidates are
            # symmetric, so if j < i the pair would already have been
            # formed when the loop reached j) — marker is the earlier one.
            dispatches.append(LogicalDispatch(
                markers=[marker, ordered[j]],
                anchor_timestamp=marker.timestamp,
                handle=marker.handle,
            ))
        else:
            used[i] = True
            dispatches.append(LogicalDispatch(
                markers=[marker], anchor_timestamp=marker.timestamp,
                handle=marker.handle,
            ))

    # Already ascending by construction (a dispatch's anchor is always its
    # first/earliest marker, appended while walking `ordered` left to
    # right) — re-sort defensively so the zip in reconcile_transcript never
    # relies on that being an accident of the loop above. Same tie-break
    # note as the sort at the top of this function.
    dispatches.sort(key=lambda d: d.anchor_timestamp)
    return dispatches, ""


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile_transcript(
    transcript_path: str, markers: list[MarkerHit],
) -> tuple[list[tuple[LogicalDispatch, BoundaryHit, str]], TranscriptReport]:
    """Reconcile one transcript's legacy markers against its unstamped turns.

    All-or-nothing per transcript: either every LOGICAL DISPATCH (see
    ``group_logical_dispatches`` — a genuine @mention double-marker counts
    as one) reconciles with exactly one turn boundary (counts match,
    chronology sanity checks pass), in which case a dispatch_id is minted
    for each pair and stamped onto every marker in the dispatch, or NONE of
    them are stamped and the report explains why. See module docstring for
    the rationale (a mis-stamped turn is worse than an unstamped one).

    Returns ``(pairs, report)`` where ``pairs`` is a list of
    ``(dispatch, boundary, dispatch_id)`` — empty unless ``report.status ==
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

    # A naive/aware mix would raise TypeError the instant two of these
    # timestamps are compared (grouping, sorting, or the pairwise zip
    # below) — detect it explicitly up front so the failure is a reported
    # reason, not an uncaught exception. migrate_workspace also wraps the
    # call to this function in try/except as a second line of defense
    # against any comparison this check doesn't anticipate.
    all_timestamps = [m.timestamp for m in markers] + [b.timestamp for b in unstamped]
    if len({ts.tzinfo is not None for ts in all_timestamps}) > 1:
        report.reason = (
            "mixed timezone-aware and timezone-naive timestamps among this "
            "transcript's markers/boundaries — cannot compare safely"
        )
        return [], report

    dispatches, group_failure = group_logical_dispatches(markers)
    if group_failure:
        report.reason = group_failure
        return [], report
    report.logical_dispatches = len(dispatches)

    if len(dispatches) != len(unstamped):
        report.reason = (
            f"count mismatch: {len(dispatches)} logical dispatch(es) (from "
            f"{len(markers)} marker(s)) vs {len(unstamped)} unstamped "
            f"turn(s)"
        )
        return [], report

    slack = timedelta(seconds=SLACK_SECONDS)
    prev_boundary_ts: "datetime | None" = None
    for idx, (dispatch, boundary) in enumerate(zip(dispatches, unstamped)):
        if dispatch.anchor_timestamp > boundary.timestamp:
            report.reason = (
                f"pair {idx}: dispatch timestamp "
                f"{dispatch.anchor_timestamp.isoformat()!r} is after its "
                f"turn's boundary timestamp {boundary.raw_timestamp!r}"
            )
            return [], report
        if prev_boundary_ts is not None and dispatch.anchor_timestamp < prev_boundary_ts - slack:
            report.reason = (
                f"pair {idx}: dispatch timestamp "
                f"{dispatch.anchor_timestamp.isoformat()!r} precedes the "
                f"previous turn's boundary timestamp "
                f"({prev_boundary_ts.isoformat()}) by more than the "
                f"{SLACK_SECONDS}s jitter allowance"
            )
            return [], report
        prev_boundary_ts = boundary.timestamp

    # Stable sort ties (see group_logical_dispatches) aside, dispatches and
    # unstamped are each already in ascending-timestamp/file order, so this
    # zip pairs the i-th chronological dispatch with the i-th chronological
    # turn — the same positional contract the live write path guarantees
    # in real time, just reconstructed after the fact.
    pairs = [
        (dispatch, boundary, f"{dispatch.markers[0].session_uuid}:{uuid4().hex[:8]}")
        for dispatch, boundary in zip(dispatches, unstamped)
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
        try:
            pairs, report = reconcile_transcript(transcript_path, markers)
        except Exception as exc:
            # reconcile_transcript explicitly pre-checks the known crash
            # cause (mixed aware/naive timestamps), but this is a second
            # line of defense: ANY unanticipated exception here must fail
            # only THIS transcript, never the whole run — other
            # transcripts (and other workspaces, in main()'s loop) still
            # need to be processed.
            report = TranscriptReport(
                transcript_path=transcript_path, markers_considered=len(markers),
                reason=f"unexpected error during reconciliation: {exc!r}",
            )
            pairs = []
        reports.append(report)
        if not pairs:
            continue

        t_edits = transcript_edits.setdefault(transcript_path, {})
        for dispatch, boundary, dispatch_id in pairs:
            new_boundary = dict(boundary.record)
            new_boundary["dispatch_id"] = dispatch_id
            t_edits[boundary.line_no] = new_boundary

            for marker in dispatch.markers:
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
        # Transcript boundary FIRST, session marker SECOND. A crash between
        # the two writes must leave the MARKER side unstamped (still
        # "legacy") so the next run's scan_legacy_markers still finds it
        # and reconsiders this transcript — a stamped-boundary paired with
        # a still-legacy marker just produces a safe, reported count
        # mismatch on retry. The REVERSE order can silently orphan a
        # stamped marker (now excluded from re-scan as "already migrated")
        # whose boundary never actually got the id: that dispatch would
        # never render a bubble and the migration would never revisit it
        # to fix it — invisible data loss instead of a visible retry.
        _apply_edits(transcript_edits)
        _apply_edits(session_edits)

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

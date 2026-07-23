# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Static-analysis regression net for session_id threading.

These tests parse the backend source tree (regex + light AST) and assert
invariants that previous "rounds" of fixes (`7d08d1b`, `bf3c97c`, ...) kept
re-breaking because the bugs were *omissions* — a missing kwarg, a missing
field in a broadcast dict. Standard unit tests with mocks pass because the
mock satisfies the test even when the production wiring is missing.

Each invariant here came from a real bug observed during the audit. If a
future change reintroduces the pattern, the corresponding test fails at the
file:line of the violation, before the bug ships.

See ``REPORT-session-id-audit.md`` for the full audit table.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DAEMON_DIR = REPO_ROOT / "agent_os" / "daemon_v2"
API_DIR = REPO_ROOT / "agent_os" / "api"

# Event types that are scoped to a single session. Any WS broadcast emitting
# one of these MUST include ``session_id`` in its payload dict.
SESSION_SCOPED_EVENT_TYPES = {
    "approval.request",
    "approval.resolved",
    "chat.sub_agent_message",
    "agent.activity",
    "agent.status",
    "agent.status_summary",
    "sub_agent.started",
    "sub_agent.completed",
    "sub_agent.error",
    "sub_agent.failed",
    "workspace_claudemd_warning",
    "agent.error",
    "agent.stopped",
}


def _python_sources(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return files


# ── 1. _idle_poll_tasks keyed by SessionKey ───────────────────────────

def test_idle_poll_tasks_declared_session_keyed():
    """The declaration of ``_idle_poll_tasks`` must annotate ``SessionKey``.

    A bare ``str`` (project_id) key conflates multiple sessions in the same
    project. The current_holder_session_id docstring explicitly describes
    condition 3 as session-scoped — the runtime type must match.
    """
    text = (DAEMON_DIR / "agent_manager.py").read_text(encoding="utf-8")
    match = re.search(r"self\._idle_poll_tasks\s*:\s*dict\[(.+?)\s*,", text)
    assert match is not None, "Could not find _idle_poll_tasks declaration"
    key_type = match.group(1).strip()
    assert key_type == "SessionKey", (
        f"_idle_poll_tasks must be keyed by SessionKey, not {key_type!r}. "
        f"See REPORT-session-id-audit.md §1."
    )


def test_idle_poll_tasks_reads_use_session_key():
    """Every read/write of ``_idle_poll_tasks`` must use a SessionKey-shaped key.

    Catches a future regression to ``_idle_poll_tasks.get(project_id)`` or
    ``_idle_poll_tasks[project_id] = ...`` (bare string key).
    """
    text = (DAEMON_DIR / "agent_manager.py").read_text(encoding="utf-8")
    # Strip docstrings: triple-quoted blocks contain example syntax we don't
    # want to lint. Crude but sufficient for this file.
    text_no_docs = re.sub(r'"""[\s\S]*?"""', '', text)
    violations: list[str] = []
    pattern = re.compile(r"_idle_poll_tasks\s*(\.get|\.pop|\[)\s*\(?\s*([^,\)\]]+?)\s*[,\)\]]")
    for m in pattern.finditer(text_no_docs):
        key_expr = m.group(2).strip()
        # Accept calls like ``make_session_key(project_id, session_id)`` or a
        # named SessionKey variable (``sk``). Reject bare ``project_id``,
        # ``pid``, or any plain identifier ending in ``_id``.
        if key_expr.startswith("make_session_key("):
            continue
        if key_expr in {"sk"}:
            continue
        line_no = text_no_docs[: m.start()].count("\n") + 1
        violations.append(f"agent_manager.py:{line_no}: bare key {key_expr!r}")
    assert not violations, (
        "All _idle_poll_tasks accesses must use SessionKey. Violations:\n  "
        + "\n  ".join(violations)
    )


# ── 2. WebSocket broadcasts for session-scoped events carry session_id ──

def _find_broadcast_calls(tree: ast.AST):
    """Yield every call of the form ``X.broadcast(project_id, {...})`` whose
    second positional arg is a dict literal containing a ``type`` key."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "broadcast":
            continue
        if len(node.args) < 2:
            continue
        payload = node.args[1]
        if not isinstance(payload, ast.Dict):
            continue
        type_value: str | None = None
        for k, v in zip(payload.keys, payload.values):
            if isinstance(k, ast.Constant) and k.value == "type":
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    type_value = v.value
                break
        yield node, payload, type_value


def test_session_scoped_broadcasts_include_session_id():
    """Every WS ``broadcast(... {"type": <session-scoped>})`` must include
    a ``session_id`` key in the payload dict.

    Caught Round-3 bug: ``approval.request`` and ``chat.sub_agent_message``
    broadcasts in process_manager.py omitted session_id, so the frontend
    could not attribute them to a session.
    """
    violations: list[str] = []
    for path in _python_sources(DAEMON_DIR, API_DIR):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node, payload, type_value in _find_broadcast_calls(tree):
            if type_value not in SESSION_SCOPED_EVENT_TYPES:
                continue
            has_session_id = any(
                isinstance(k, ast.Constant) and k.value == "session_id"
                for k in payload.keys
            )
            if not has_session_id:
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{node.lineno}: broadcast type={type_value!r} missing session_id")
    assert not violations, (
        "WebSocket broadcasts for session-scoped event types must include "
        "session_id in payload. Violations:\n  " + "\n  ".join(violations)
    )


# ── 3. lifecycle_observer.on_* calls pass session_id ───────────────────

LIFECYCLE_METHODS = {"on_started", "on_message_routed", "on_completed", "on_error", "on_failed"}


def test_lifecycle_observer_calls_pass_session_id():
    """Every call to ``lifecycle_observer.on_*`` (or via attribute lookup
    such as ``self._lifecycle.on_completed``) must pass ``session_id`` as a
    keyword argument.

    Caught Round-1 (AgentMessageTool) and Round-2 (process_manager SDK
    completion) regressions. ``on_completed(..., session_id=session_id)``
    is the canonical shape.
    """
    violations: list[str] = []
    for path in _python_sources(DAEMON_DIR, API_DIR):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in LIFECYCLE_METHODS:
                continue
            # Heuristic: skip method *definitions* (handled by ast.FunctionDef
            # elsewhere) — we only care about call sites here.
            base = node.func.value
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            # Filter to plausible lifecycle-observer attribute names. This
            # avoids false positives on unrelated ``on_completed`` methods
            # (e.g. an asyncio future callback).
            if not (
                "lifecycle" in base_name.lower()
                or base_name == "_lifecycle"
                or base_name == "lifecycle_observer"
                or base_name == "_lifecycle_observer"
            ):
                continue
            kw_names = {kw.arg for kw in node.keywords}
            if "session_id" not in kw_names:
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{node.lineno}: {base_name}.{method}(...) missing session_id kwarg")
    assert not violations, (
        "All lifecycle_observer.on_* call sites must pass session_id kwarg. "
        "Violations:\n  " + "\n  ".join(violations)
    )


# ── 4. sub_agent_manager approval/dispatch calls pass session_id ───────

SUB_AGENT_MANAGER_SESSION_SCOPED = {
    "get_pending_sub_agent_approval",
    "resolve_sub_agent_approval",
    "list_active",
    "stop_all",
    "update_sub_agent_autonomy",
}


def test_sub_agent_manager_callers_pass_session_id():
    """Public ``sub_agent_manager`` methods that fall back to ``DEFAULT_SESSION_ID``
    when ``session_id`` is None must be called with an explicit ``session_id``
    kwarg from the REST and tool layers.

    Caught Round-3 bug: ``agents_v2.py:857`` called ``get_pending_sub_agent_approval(project_id)``
    (no session_id), so multi-session approvals silently resolved against the
    default-session sentinel.
    """
    violations: list[str] = []
    for path in _python_sources(API_DIR, REPO_ROOT / "agent_os" / "agent"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in SUB_AGENT_MANAGER_SESSION_SCOPED:
                continue
            base = node.func.value
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if "sub_agent_manager" not in base_name.lower():
                continue
            kw_names = {kw.arg for kw in node.keywords}
            if "session_id" not in kw_names:
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{node.lineno}: {base_name}.{method}(...) missing session_id kwarg")
    assert not violations, (
        "All session-scoped sub_agent_manager call sites must pass session_id "
        "kwarg. Violations:\n  " + "\n  ".join(violations)
    )

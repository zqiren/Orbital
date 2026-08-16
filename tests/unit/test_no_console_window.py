# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Static-analysis regression net for bug #55 — stray Windows console windows.

The packaged Windows build is a windowed-subsystem executable, so the daemon
owns no console. Any *console*-subsystem child spawned without
``CREATE_NO_WINDOW`` gets a brand-new **visible** ``conhost.exe`` window —
the black ``C:\\Windows\\SYSTEM32\\icacls.`` box a user reported flashing over
the chat UI on 2026-08-12.

This is an *omission* bug: nine call sites across six files simply never
passed the kwarg. Nothing that currently exists can catch it — the icacls
tests are ``@skip_not_windows``-gated and never run in CI, and window
visibility is invisible to headless CI even on Windows. So the net is a
source-text sweep: every process-creation site under ``agent_os/`` must
either pass the no-window flag or sit on the explicit allowlist below, with
an exact expected count so a newly-added bare call in an allowlisted file
still fails.

"Process-creation site" covers both the synchronous ``subprocess.run``/
``Popen`` family and the ``asyncio.create_subprocess_exec``/``_shell``
coroutines — asyncio forwards ``**kwds`` straight to ``subprocess.Popen``, so
those carry the identical defect and take the identical fix.

Runtime window visibility itself is NOT assertable from here — see the
manual on-Windows Task Manager checklist in the bug #55 spec §6 decision 5(b).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from agent_os.utils.subprocess_flags import CREATE_NO_WINDOW, win_no_window_flags

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_OS_DIR = REPO_ROOT / "agent_os"

#: ``subprocess`` attributes that create a process. ``run``/``Popen`` are the
#: only two in use today; the rest are listed so a future ``check_output``
#: does not slip in unflagged.
_SUBPROCESS_SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}

#: ``asyncio`` coroutines that create a process. Both forward ``**kwds`` to
#: ``subprocess.Popen``, so they carry the identical defect and take the
#: identical fix.
_ASYNCIO_SPAWNERS = {"create_subprocess_exec", "create_subprocess_shell"}

#: Substrings that make a ``creationflags=`` expression compliant.
#: ``win_no_window_flags`` is the canonical form (``agent_os/utils/subprocess_flags.py``);
#: a bare ``CREATE_NO_WINDOW`` literal is accepted because
#: ``platform/windows/sandbox.py`` predates the helper and is already correct.
_COMPLIANT_MARKERS = ("win_no_window_flags", "CREATE_NO_WINDOW")

#: Files permitted to spawn without the flag, mapped to
#: ``(exact number of bare call sites, why)``. The count is part of the
#: assertion: adding a bare call to an allowlisted file still fails the test.
_ALLOWLIST: dict[str, tuple[int, str]] = {
    "agent_os/platform/macos/provider.py": (
        4,
        "macOS-only module. MacOSPlatformProvider is instantiated only when "
        "sys.platform == 'darwin' (agent_os/platform/__init__.py), and it "
        "spawns sandbox-exec/caffeinate — never reachable on Windows. "
        "Permanent exemption.",
    ),
}


def _python_sources(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "vendor" not in p.parts
    )


def _is_spawner_ref(node: ast.AST) -> bool:
    """True for a ``subprocess.<spawner>`` / ``asyncio.<spawner>`` reference."""
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
        return False
    module, attr = node.value.id, node.attr
    return (
        (module == "subprocess" and attr in _SUBPROCESS_SPAWNERS)
        or (module == "asyncio" and attr in _ASYNCIO_SPAWNERS)
    )


def _spawn_sites(tree: ast.AST):
    """Yield ``(lineno, keywords)`` for every process-creation site.

    Two shapes exist in this codebase:

    * direct — ``subprocess.run(cmd, ...)`` or
      ``await asyncio.create_subprocess_exec(...)``: the call's own kwargs
      apply.
    * deferred — ``asyncio.to_thread(subprocess.run, cmd, ...)``: the
      *outer* call's kwargs are forwarded to ``subprocess.run``, so they are
      the ones to inspect.

    Bare ``subprocess.Popen`` references that are not calls (type
    annotations like ``self._process: subprocess.Popen | None``) are skipped
    because only ``ast.Call`` nodes are visited.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_spawner_ref(node.func):
            yield node.lineno, node.keywords
            continue
        if any(_is_spawner_ref(arg) for arg in node.args):
            yield node.lineno, node.keywords


def _is_compliant(keywords: list[ast.keyword]) -> bool:
    for kw in keywords:
        if kw.arg != "creationflags":
            continue
        expr = ast.unparse(kw.value)
        return any(marker in expr for marker in _COMPLIANT_MARKERS)
    return False


def _collect_bare_sites() -> dict[str, list[str]]:
    """Map relative path -> ["path:lineno", ...] for every unflagged site."""
    bare: dict[str, list[str]] = {}
    for path in _python_sources(AGENT_OS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, keywords in _spawn_sites(tree):
            if not _is_compliant(keywords):
                bare.setdefault(rel, []).append(f"{rel}:{lineno}")
    return bare


# ── 1. the sweep ───────────────────────────────────────────────────────


def test_no_spawner_is_imported_bare():
    """``from subprocess import run`` would slip past the attribute walker.

    The sweep matches ``subprocess.run`` / ``asyncio.create_subprocess_exec``
    as *attribute* references. A bare-name import would make a call site
    invisible to it, so forbid that import shape outright rather than teach
    the walker to track aliases.
    """
    violations: list[str] = []
    for path in _python_sources(AGENT_OS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "subprocess":
                banned = _SUBPROCESS_SPAWNERS
            elif node.module == "asyncio":
                banned = _ASYNCIO_SPAWNERS
            else:
                continue
            for alias in node.names:
                if alias.name in banned:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    violations.append(
                        f"{rel}:{node.lineno}: from {node.module} import {alias.name}"
                    )
    assert not violations, (
        "Import the module and call the attribute (``subprocess.run(...)``) so "
        "the bug #55 sweep can see the call site:\n  " + "\n  ".join(violations)
    )


def test_every_subprocess_site_suppresses_the_console_window():
    """No process-creation site under ``agent_os/`` may spawn unflagged.

    A violation here means a Windows user will see a black console window
    flash over the UI when that code path runs. Fix it by passing
    ``creationflags=win_no_window_flags()`` (OR it into any existing flag
    value), or — if the site genuinely cannot reach Windows — add it to
    ``_ALLOWLIST`` above with a reason.
    """
    bare = _collect_bare_sites()
    violations = [
        site
        for rel, sites in sorted(bare.items())
        if rel not in _ALLOWLIST
        for site in sites
    ]
    assert not violations, (
        "process-creation sites missing creationflags=win_no_window_flags() "
        "(bug #55 — stray Windows console window):\n  "
        + "\n  ".join(violations)
    )


def test_allowlisted_files_have_not_grown_new_bare_sites():
    """Each allowlisted file must carry exactly the bare-site count recorded.

    Keeps the allowlist honest: a new unflagged call added to, say,
    ``null.py`` fails here instead of silently inheriting the exemption.
    """
    bare = _collect_bare_sites()
    drift: list[str] = []
    for rel, (expected, reason) in sorted(_ALLOWLIST.items()):
        actual = len(bare.get(rel, []))
        if actual != expected:
            drift.append(
                f"{rel}: allowlisted for {expected} bare site(s), found {actual}. "
                f"Reason on record: {reason}"
            )
    assert not drift, (
        "Allowlist is stale. Either flag the new site with "
        "win_no_window_flags() or update the count + reason:\n  "
        + "\n  ".join(drift)
    )


# ── 2. the helper itself ───────────────────────────────────────────────


def test_helper_is_a_noop_off_windows():
    """Off Windows the helper must return 0.

    ``subprocess`` raises ``ValueError`` for a *non-zero* ``creationflags``
    on POSIX, so 0 is what keeps every call site cross-platform. It also
    means the fix is behaviour-neutral on macOS/Linux.
    """
    expected = CREATE_NO_WINDOW if sys.platform == "win32" else 0
    assert win_no_window_flags() == expected


def test_create_no_window_constant_matches_win32():
    """The literal must be the real ``CREATE_NO_WINDOW`` value.

    Spelled as a literal because ``subprocess.CREATE_NO_WINDOW`` does not
    exist off Windows — so cross-check it against the stdlib wherever the
    stdlib actually has it.
    """
    assert CREATE_NO_WINDOW == 0x08000000
    if sys.platform == "win32":
        assert CREATE_NO_WINDOW == subprocess.CREATE_NO_WINDOW


def test_zero_creationflags_is_accepted_by_subprocess_off_windows():
    """Guard the assumption the whole fix rests on.

    If ``creationflags=0`` ever became an error on POSIX, all eleven patched
    sites would break on macOS at once.
    """
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        capture_output=True,
        creationflags=win_no_window_flags(),
    )
    assert result.returncode == 0


# ── 3. the reported defect, pinned by name ─────────────────────────────


def test_run_icacls_passes_the_no_window_flag():
    """``_run_icacls`` is the site in the user's 2026-08-12 screenshot.

    Behavioural rather than source-text, and OS-independent:
    ``permissions.py`` imports fine on macOS (stdlib + platform.types only),
    so the kwarg can be asserted by intercepting ``subprocess.run``.
    """
    from agent_os.platform.windows import permissions as perms

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, "", "")

    original = perms.subprocess.run
    perms.subprocess.run = _fake_run
    try:
        perms.PermissionManager._run_icacls(["C:\\ws", "/grant", "u:(OI)(CI)F"])
    finally:
        perms.subprocess.run = original

    assert captured["cmd"][0] == "icacls"
    assert "creationflags" in captured["kwargs"], (
        "_run_icacls must pass creationflags — this is the exact call that "
        "produced the reported conhost.exe flash (bug #55)."
    )
    assert captured["kwargs"]["creationflags"] == win_no_window_flags()


# ── 4. transports keep BOTH flags ──────────────────────────────────────


def test_transport_fallbacks_keep_process_group_and_add_no_window():
    """The Windows PIPE fallbacks need ``CREATE_NEW_PROCESS_GROUP`` too.

    That flag exists so ``GenerateConsoleCtrlEvent``/``CTRL_BREAK_EVENT`` can
    reach a sub-agent CLI for graceful shutdown; it says nothing about window
    visibility. Bug #55 ORs the no-window flag in alongside it — this pins
    that a later edit cannot drop either half.
    """
    for rel in (
        "agent_os/agent/transports/pty_transport.py",
        "agent_os/agent/adapters/cli_adapter.py",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "CREATE_NEW_PROCESS_GROUP" in text, f"{rel} lost CREATE_NEW_PROCESS_GROUP"
        assert "win_no_window_flags()" in text, f"{rel} lost win_no_window_flags()"


# ── 5. the helper's home stays cross-platform-importable ───────────────


def test_helper_does_not_live_under_platform_windows():
    """``agent_os/platform/windows/`` only imports on Windows.

    Most sites that need the flag (grep_tool, workspace_scan, shell,
    setup_engine, migration, settings, both transports) are cross-platform
    modules that also run on macOS, so the helper must not be reachable only
    through a Windows-gated package. This test failing means the helper moved
    somewhere those modules cannot import from.
    """
    import agent_os.utils.subprocess_flags as mod

    assert "platform" not in Path(mod.__file__).parts

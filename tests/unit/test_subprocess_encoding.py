# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Static-analysis regression net for the Windows subprocess-decoding class.

On Windows, ``subprocess.run(text=True)`` without ``encoding=`` decodes child
output with the ANSI code page (cp936 on Chinese Windows, cp1252 on Western),
while most children here (ripgrep, node-based CLIs) emit UTF-8. And an
explicit ``encoding=`` with *strict* error handling is only half safe: any
undecodable byte still raises. The failure is nastier than a clean exception —
with ``capture_output=True`` on Windows the decode happens in subprocess's
reader thread, which dies silently and hands back ``stdout=None``, so the
crash surfaces later as ``AttributeError``/``TypeError`` on ``None``.

That is exactly how the cold-start workspace scan turned a workspace
containing 丁字路口.txt into a bogus "provider error" (fixed in
``workspace_scan.py``; regression-tested in ``test_workspace_scan_encoding.py``).

The net, in the same style as ``test_no_console_window.py``: every text-mode
process-creation site under ``agent_os/`` must pass BOTH ``encoding=`` and
``errors=`` (the canonical form is ``encoding="utf-8", errors="replace"``, as
in ``grep_tool.py``), or sit on the explicit allowlist with an exact count and
a reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_OS_DIR = REPO_ROOT / "agent_os"

_SUBPROCESS_SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}
_ASYNCIO_SPAWNERS = {"create_subprocess_exec", "create_subprocess_shell"}

#: Files permitted to keep text-mode sites without explicit encoding/errors,
#: mapped to ``(exact number of exempt sites, why)``.
_ALLOWLIST: dict[str, tuple[int, str]] = {
    "agent_os/desktop/main.py": (
        1,
        "$SHELL login-shell PATH probe — effectively macOS-only ($SHELL is "
        "unset on Windows) and wrapped in a broad `except Exception` that "
        "falls back to the existing PATH.",
    ),
    "agent_os/agent/tools/shell.py": (
        1,
        "Shell tool runs arbitrary user commands; locale decoding of their "
        "output is a deliberate policy question, and the call sits in a "
        "broad `except Exception` that degrades to an error string.",
    ),
    "agent_os/platform/null.py": (
        1,
        "NullProvider run_command executes arbitrary commands; same locale "
        "policy question as shell.py. Windows worst case is stdout=None in "
        "the CommandResult, not an escaping exception.",
    ),
    "agent_os/platform/macos/provider.py": (
        1,
        "macOS-only module (instantiated only when sys.platform == 'darwin'); "
        "macOS locales are UTF-8, so locale decoding is the right default.",
    ),
}


def _python_sources(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "vendor" not in p.parts
    )


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to the stdlib module they alias.

    Covers ``import subprocess`` and ``import subprocess as _sp`` — the
    latter is how ``desktop/main.py`` spells it.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name in ("subprocess", "asyncio"):
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _is_spawner_ref(node: ast.AST, aliases: dict[str, str]) -> bool:
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
        return False
    module, attr = aliases.get(node.value.id), node.attr
    return (
        (module == "subprocess" and attr in _SUBPROCESS_SPAWNERS)
        or (module == "asyncio" and attr in _ASYNCIO_SPAWNERS)
    )


def _spawn_sites(tree: ast.AST):
    """Yield ``(lineno, keywords)`` for every process-creation site.

    Mirrors ``test_no_console_window.py``: covers both direct calls and the
    deferred ``asyncio.to_thread(subprocess.run, ...)`` shape, where the
    outer call's kwargs are the ones forwarded.
    """
    aliases = _module_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_spawner_ref(node.func, aliases):
            yield node.lineno, node.keywords
            continue
        if any(_is_spawner_ref(arg, aliases) for arg in node.args):
            yield node.lineno, node.keywords


def _kw(keywords: list[ast.keyword], name: str) -> ast.keyword | None:
    for kw in keywords:
        if kw.arg == name:
            return kw
    return None


def _is_text_mode(keywords: list[ast.keyword]) -> bool:
    """A site is text-mode if it passes text=/universal_newlines=/encoding=."""
    for name in ("text", "universal_newlines"):
        kw = _kw(keywords, name)
        if kw is not None and not (
            isinstance(kw.value, ast.Constant) and kw.value.value in (False, None)
        ):
            return True
    return _kw(keywords, "encoding") is not None


def _is_decode_safe(keywords: list[ast.keyword]) -> bool:
    """Explicit encoding AND non-strict errors= makes decoding crash-proof."""
    enc = _kw(keywords, "encoding")
    err = _kw(keywords, "errors")
    if enc is None or err is None:
        return False
    if isinstance(err.value, ast.Constant) and err.value.value in (None, "strict"):
        return False
    return True


def _collect_unsafe_sites() -> dict[str, list[str]]:
    unsafe: dict[str, list[str]] = {}
    for path in _python_sources(AGENT_OS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, keywords in _spawn_sites(tree):
            if _is_text_mode(keywords) and not _is_decode_safe(keywords):
                unsafe.setdefault(rel, []).append(f"{rel}:{lineno}")
    return unsafe


def test_every_text_mode_subprocess_site_is_decode_safe():
    """Text-mode capture sites must pass encoding= AND a non-strict errors=.

    A violation means child output containing bytes invalid in the chosen
    (or ANSI-default) codec crashes the site on Windows — usually as a
    silent reader-thread death that resurfaces as ``AttributeError`` on
    ``stdout=None``. Fix by passing ``encoding="utf-8", errors="replace"``
    (see ``grep_tool.py``), or allowlist the file with a reason.
    """
    unsafe = _collect_unsafe_sites()
    violations = [
        site
        for rel, sites in sorted(unsafe.items())
        if rel not in _ALLOWLIST
        for site in sites
    ]
    assert not violations, (
        "text-mode subprocess sites without crash-proof decoding "
        "(encoding= + non-strict errors=):\n  " + "\n  ".join(violations)
    )


def test_allowlisted_files_have_not_grown_new_exempt_sites():
    """Each allowlisted file must carry exactly the exempt-site count recorded."""
    unsafe = _collect_unsafe_sites()
    drift: list[str] = []
    for rel, (expected, reason) in sorted(_ALLOWLIST.items()):
        actual = len(unsafe.get(rel, []))
        if actual != expected:
            drift.append(
                f"{rel}: allowlisted for {expected} exempt site(s), found "
                f"{actual}. Reason on record: {reason}"
            )
    assert not drift, (
        "Allowlist is stale. Either make the new site decode-safe or update "
        "the count + reason:\n  " + "\n  ".join(drift)
    )

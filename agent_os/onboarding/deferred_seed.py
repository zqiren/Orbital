# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Import writes nothing until first use (spec 138).

A project created by the batch import (``POST /onboarding/import``) is marked
``deferred_seed: true`` and gets neither the workspace-root ``AGENTS.md`` nor
the ``orbital/`` folder (default skills) at creation: the folder is the
user's real repo and nothing should appear in it until they work there.
:func:`ensure_project_seeded` does both on first use — the first session
opened (materialised) or the first message sent — exactly once.

Spec 153 adds a third step, the private-notes fold: the notes Claude Code
kept about this folder (``~/.claude/projects/<dir>/memory``) are copied
verbatim to ``orbital/imported/claude-code/``, listed in an ``INDEX.md``
there, and pointed at by ONE line in ``orbital/INDEX.md``, so every agent in
the project can read them. Unlike the first two steps, a failed fold leaves
``deferred_seed`` set and the next first use retries it.

Spec 160 §5.1 splits the seed in two. The ``orbital/`` part (default skills +
the fold) runs at import, keyless too, so the background scan has a folder
to write its guess into: :func:`seed_orbital_folder`, which marks
``orbital_seeded`` when it fully succeeded. The root ``AGENTS.md`` (the only
write outside ``orbital/``) still waits for first use, and ``deferred_seed``
keeps meaning "root file not yet written": :func:`ensure_project_seeded`
writes it, re-running the ``orbital/`` part only when the import's run did
not finish it (a failed fold is still retried on first use; a finished one
is never copied over again). A v0.20.0 daemon ignores ``orbital_seeded`` and
runs its whole first-use seed, which is idempotent.
"""

from __future__ import annotations

import locale
import logging
import os
import re
import shutil
import threading
import warnings

logger = logging.getLogger(__name__)

DEFERRED_SEED_FIELD = "deferred_seed"
# Durable "came from the import" marker (spec 153 §4.4): ``deferred_seed`` is
# cleared on first use and cannot stand in for it.
IMPORTED_FIELD = "imported"
# Spec 160 §5.1: the ``orbital/`` part of the seed (skills + fold) finished at
# import; first use then writes only the root AGENTS.md.
ORBITAL_SEEDED_FIELD = "orbital_seeded"

# Workspace-relative, POSIX separators (the wire form of a workspace path).
IMPORTED_NOTES_DIR = "orbital/imported/claude-code"
IMPORTED_INDEX_PATH = IMPORTED_NOTES_DIR + "/INDEX.md"
_POINTER_PREFIX = "- orbital/imported/claude-code/ — "
_LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)\)")

_lock = threading.Lock()


def is_deferred(project: dict | None) -> bool:
    return isinstance(project, dict) and bool(project.get(DEFERRED_SEED_FIELD))


def _seed_orbital_part(project_store, project_id: str, *, scanner=None,
                       lang: str | None = None) -> bool:
    """Default skills + the private-notes fold. True when both finished (a
    skills failure reconciles on the next agent start, like project
    creation); False when the fold failed. Caller holds ``_lock``."""
    from agent_os.daemon_v2.default_skills_installer import install_default_skills
    try:
        install_default_skills(project_store, project_id)
    except Exception:
        logger.error("deferred seed: default skills failed for %s", project_id,
                     exc_info=True)
    workspace = (project_store.get_project(project_id) or {}).get("workspace") or ""
    try:
        n = fold_private_notes(workspace, workspace_notes(workspace, scanner), lang=lang)
    except Exception:
        logger.error("deferred seed: private notes fold failed for %s; "
                     "retrying on next use", project_id, exc_info=True)
        return False
    logger.info("deferred seed: orbital/ seeded for %s (%d private notes folded)",
                project_id, n)
    return True


def seed_orbital_folder(project_store, project_id: str, *, scanner=None,
                        lang: str | None = None) -> bool:
    """Spec 160 §5.1: the ``orbital/`` part of a deferred project's seed, run
    at import. Returns True when this call ran it. ``deferred_seed`` stays
    set (the root AGENTS.md still waits for first use); ``orbital_seeded``
    is set only when the fold finished, so a failed fold is retried on first
    use. Idempotent; never raises; blocking — call it off the event loop."""
    try:
        project = project_store.get_project(project_id)
        if not is_deferred(project) or project.get(ORBITAL_SEEDED_FIELD):
            return False
        with _lock:
            project = project_store.get_project(project_id)
            if not is_deferred(project) or project.get(ORBITAL_SEEDED_FIELD):
                return False
            if _seed_orbital_part(project_store, project_id, scanner=scanner, lang=lang):
                project_store.update_project(project_id, {ORBITAL_SEEDED_FIELD: True})
            return True
    except Exception:
        logger.warning("deferred seed: could not seed orbital/ for %s", project_id,
                       exc_info=True)
        return False


def ensure_project_seeded(project_store, project_id: str, *, scanner=None,
                          lang: str | None = None) -> bool:
    """Seed a deferred project now. Returns True when this call seeded it,
    False when there was nothing to do. Idempotent; never raises. Blocking
    (file copies): call it off the event loop.

    ``scanner`` is the history scanner whose notes list feeds the fold (read
    directly for this one workspace when it is absent or not ready);
    ``lang`` the language of the imported index's header line.

    The ``orbital/`` part is skipped when the import already finished it
    (``orbital_seeded``, spec 160)."""
    try:
        if not is_deferred(project_store.get_project(project_id)):
            return False
        with _lock:
            project = project_store.get_project(project_id)
            if not is_deferred(project):
                return False
            from agent_os.daemon_v2.agent_md_seeder import seed_project_agent_md
            orbital_ok = (bool(project.get(ORBITAL_SEEDED_FIELD))
                          or _seed_orbital_part(project_store, project_id,
                                                scanner=scanner, lang=lang))
            try:
                seed_project_agent_md(project_store, project_id)
            except Exception:
                logger.error("deferred seed: AGENTS.md failed for %s", project_id,
                             exc_info=True)
            if not orbital_ok:
                # The field stays set: the next first use retries the fold.
                return True
            project_store.update_project(project_id, {DEFERRED_SEED_FIELD: False})
            logger.info("deferred seed: seeded %s on first use", project_id)
            return True
    except Exception:
        logger.warning("deferred seed: could not seed %s", project_id, exc_info=True)
        return False


# --------------------------------------------------------------------------
# The private-notes fold (spec 153 §4.3)
# --------------------------------------------------------------------------


def workspace_notes(workspace: str, scanner=None) -> list:
    """The private notes kept about ``workspace``: the scanner's list when its
    scan is done, else the Claude dirs whose transcripts ran here, read now
    (a project opened seconds after install). Read-only."""
    if not workspace:
        return []
    is_ready = getattr(scanner, "is_ready", None)
    if callable(is_ready) and is_ready() and hasattr(scanner, "notes_for"):
        return scanner.notes_for(workspace)
    from agent_os.onboarding.history_paths import claude_dir, history_home
    from agent_os.onboarding.import_scanner import claude_project_dirs_for
    from agent_os.onboarding.sources import claude_code

    projects = os.path.join(claude_dir(history_home()), "projects")
    dirs = claude_project_dirs_for(projects, workspace)
    return claude_code.notes_in_dirs(dirs, folder=workspace) if dirs else []


def default_lang() -> str:
    """``zh`` on a Chinese system locale, else ``en`` — the same rule the
    desktop shell uses for its native dialogs. The daemon has no UI locale."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lang = (locale.getdefaultlocale()[0] or "").lower()
    except Exception:
        lang = ""
    return "zh" if lang.startswith("zh") else "en"


def _copy_verbatim(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)


def _display_dir(path: str) -> str:
    home = os.path.expanduser("~")
    if home and (path == home or path.startswith(home.rstrip("/\\") + os.sep)):
        return "~" + path[len(home.rstrip("/\\")):]
    return path


def _dest_names(notes: list) -> list[str]:
    """One file name per note in the imported folder: the source name, except
    that a note called ``index.md`` must not overwrite the imported index and
    two source dirs may both hold the same name."""
    used: set[str] = {"index.md"}
    out = []
    for note in notes:
        name = os.path.basename(note.path)
        if name.lower() == "index.md":
            name = "index-note.md"
        base, i = name, 2
        while name.lower() in used:
            name = f"{i}-{base}"
            i += 1
        used.add(name.lower())
        out.append(name)
    return out


def _header(n: int, source_dirs: list[str], lang: str) -> str:
    dirs = ", ".join(f"`{_display_dir(d)}`" for d in source_dirs)
    if lang == "zh":
        return f"Claude Code 对这个项目的 {n} 条记忆，导入自 {dirs}"
    noun = "memory" if n == 1 else "memories"
    return f"{n} {noun} Claude Code kept about this project, imported from {dirs}"


def _pointer_line(n: int) -> str:
    noun = "memory" if n == 1 else "memories"
    return (f"{_POINTER_PREFIX}{n} {noun} Claude Code kept about this project "
            "(INDEX.md lists them; feedback entries are the user's own corrections)")


def _index_body(notes: list, names: list[str]) -> list[str]:
    """The agent's own index lines, in its order, links retargeted to the
    copies; then a ``title — hook`` line for every note it never indexed."""
    from agent_os.onboarding.sources.claude_code import MEMORY_INDEX, _index_lines

    lines: list[str] = []
    placed: set[int] = set()
    by_dir: dict[str, dict[str, int]] = {}
    for i, note in enumerate(notes):
        by_dir.setdefault(os.path.dirname(note.path), {})[os.path.basename(note.path)] = i
    for mem_dir, members in by_dir.items():
        for name in _index_lines(os.path.join(mem_dir, MEMORY_INDEX)):
            i = members.get(name)
            if i is None or i in placed or not notes[i].index_line:
                continue
            target = names[i]
            lines.append(_LINK_TARGET_RE.sub(lambda _m: f"]({target})", notes[i].index_line,
                                             count=1))
            placed.add(i)
    rest = sorted((i for i in range(len(notes)) if i not in placed), key=lambda i: names[i])
    for i in rest:
        note = notes[i]
        title = note.title.replace("[", "(").replace("]", ")")
        line = f"- [{title}]({names[i]})"
        if note.hook:
            line += f" — {note.hook}"
        lines.append(line)
    return lines


def fold_private_notes(workspace: str, notes: list, *, lang: str | None = None) -> int:
    """Copy ``notes`` into ``orbital/imported/claude-code/``, write its
    INDEX.md and the one ``orbital/INDEX.md`` pointer. Returns how many
    notes were folded; raises on any failure (the caller keeps the seed
    field set). Re-running it after a partial failure is safe: copies are
    overwritten with the same bytes and the pointer line is never doubled."""
    if not notes or not workspace:
        return 0
    from agent_os.agent.workspace_files import WorkspaceFileManager

    lang = lang or default_lang()
    dest = os.path.join(workspace, *IMPORTED_NOTES_DIR.split("/"))
    os.makedirs(dest, exist_ok=True)
    names = _dest_names(notes)
    for note, name in zip(notes, names):
        _copy_verbatim(note.path, os.path.join(dest, name))

    source_dirs = list(dict.fromkeys(os.path.dirname(n.path) for n in notes))
    text = "\n".join([_header(len(notes), source_dirs, lang), "",
                      *_index_body(notes, names)]) + "\n"
    index_path = os.path.join(dest, "INDEX.md")
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, index_path)

    wfm = WorkspaceFileManager(workspace)
    existing = wfm.read("index") or ""
    pointer = _pointer_line(len(notes))
    kept = existing.splitlines()
    if any(ln.startswith(_POINTER_PREFIX) for ln in kept):
        updated = "\n".join(pointer if ln.startswith(_POINTER_PREFIX) else ln
                            for ln in kept) + "\n"
    elif existing.strip():
        updated = existing.rstrip("\n") + "\n" + pointer + "\n"
    else:
        updated = pointer + "\n"
    if updated != existing:
        wfm.write("index", updated)
    return len(notes)

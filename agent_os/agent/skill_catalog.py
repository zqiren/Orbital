# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every skill a project can use, as one flat list (spec 132).

The composer's ``/`` picker lists it, and the inject route re-resolves a
picked skill against it: a ``skill.path`` this catalog did not return is
refused, so a forged path can never point an agent at an arbitrary file.

Sources, highest precedence first::

    {ws}/orbital/skills   Orbital's own (the 4 defaults are seeded here)
    {ws}/.claude/skills   workspace-native
    {ws}/.agents/skills
    {data}/skills         Orbital's global folder, every project's
    ~/.agents/skills      user-global; Claude Code and Codex share these
    ~/.claude/skills
    ~/.codex/skills

A skill is a directory holding a ``SKILL.md``. Dot-dirs (``~/.codex/skills/
.system``) and directories without one (``~/.claude/skills/synced``) are
skipped. Plugin skills (``~/.claude/plugins``) are not listed.

Dedupe runs in source order: first by the realpath of the skill dir (the dev
Mac's ``~/.claude/skills/x`` is a symlink into ``~/.agents/skills/x``), then
by normalized name. The first one seen wins.

Containment: a symlinked skill DIRECTORY is followed wherever it points (a
dotfiles checkout is a normal install), but the ``SKILL.md`` inside it must
resolve inside that directory's realpath. A ``SKILL.md`` that is a link out to
some other file is never parsed, listed, or handed to an agent.
"""

from __future__ import annotations

import os

from agent_os.agent import skills as _skills
from agent_os.agent.skills import parse_skill_md

_UNSET = object()


def _normalize(name: str) -> str:
    return name.strip().lower()


def _inside(child_real: str, parent_real: str) -> bool:
    return child_real.startswith(parent_real.rstrip(os.sep) + os.sep)


def _folder_entries(root: str):
    """``(entry, dir_real)`` per skill folder in ``root``, sorted by name:
    dot-dirs, non-dirs, folders without a ``SKILL.md`` and a ``SKILL.md``
    linking out of its folder are skipped; an unreadable root yields nothing."""
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        skill_md = os.path.join(entry.path, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        dir_real = os.path.realpath(entry.path)
        if not _inside(os.path.realpath(skill_md), dir_real):
            continue
        yield entry, dir_real


def _row(source: str, entry, meta: dict) -> dict:
    name = meta["name"] or entry.name
    return {
        "key": _normalize(name),
        "name": name,
        "description": meta["description"] or "",
        "source": source,
        "path": os.path.abspath(os.path.join(entry.path, "SKILL.md")),
        "dir_name": entry.name,
    }


# The sources every project shares, highest precedence first. Their skills
# can be switched off (``skills.read_disabled_skills``); project and
# workspace skills cannot.
SHARED_SOURCES = ("global", "agents", "claude", "codex")

# Whose skills a shared folder holds, for the Settings page's logo.
_AGENT_OF_SOURCE = {"global": "orbital", "agents": "shared",
                    "claude": "claude-code", "codex": "codex"}


def home_skill_folders(home: str | None = None) -> list[tuple[str, str]]:
    """The user's own CLI skill folders Orbital reads (never writes)."""
    home = home if home is not None else _skills.skills_home()
    return [
        ("agents", os.path.join(home, ".agents", "skills")),
        ("claude", os.path.join(home, ".claude", "skills")),
        ("codex", os.path.join(home, ".codex", "skills")),
    ]


def shared_skill_folders(*, home: str | None = None, global_dir=_UNSET) -> list[tuple[str, str]]:
    """``[(source, folder)]`` for the global folder (when configured) and the
    home folders, in precedence order."""
    if global_dir is _UNSET:
        global_dir = _skills.global_skills_dir()
    return ([("global", global_dir)] if global_dir else []) + home_skill_folders(home)


def _walk(sources, disabled, taken=None):
    """``(row, state, hidden_by)`` for each skill folder in source order.

    A folder already seen (by realpath: a symlink into another source) is
    yielded once more as ``"linked"``, with ``hidden_by`` = the id of the
    row it was first seen as. Otherwise ``state``: ``"ok"`` (offered),
    ``"off"`` (a shared skill switched off) or ``"hidden"`` (an earlier
    offered skill has its name). ``taken`` maps the names already offered to
    who offers them; ``hidden_by`` is that, for an off or hidden row."""
    taken = dict(taken or {})
    seen_dirs: dict[str, str] = {}
    for source, root in sources:
        for entry, dir_real in _folder_entries(root):
            if dir_real in seen_dirs:
                yield {"source": source, "dir_name": entry.name}, "linked", seen_dirs[dir_real]
                continue
            meta = parse_skill_md(os.path.join(entry.path, "SKILL.md"))
            if meta is None:
                continue
            row = _row(source, entry, meta)
            key = row["key"]
            if not key:
                continue
            seen_dirs[dir_real] = f"{source}:{key}"
            if key in taken:
                yield row, "hidden", taken[key]
            elif source in SHARED_SOURCES and f"{source}:{key}" in disabled:
                yield row, "off", None
            else:
                taken[key] = f"{source}:{key}"
                yield row, "ok", None


def scan_shared(*, shadowed_by=(), home: str | None = None, global_dir=_UNSET,
                disabled=None, require_description: bool = False) -> list[dict]:
    """The shared skills a project is offered: switched-on, not named like
    one in ``shadowed_by`` or an earlier shared one. Global rows first (by
    folder name), then home rows (by name): a stable order for the prompt."""
    if disabled is None:
        disabled = _skills.read_disabled_skills()
    taken = {_normalize(s["name"]): "project" for s in shadowed_by}
    rows = [row for row, state, _ in _walk(
                shared_skill_folders(home=home, global_dir=global_dir), disabled, taken)
            if state == "ok" and (row["description"] or not require_description)]
    return ([r for r in rows if r["source"] == "global"]
            + sorted((r for r in rows if r["source"] != "global"), key=lambda r: r["key"]))


def shared_skill_groups(*, home: str | None = None, global_dir=_UNSET, disabled=None) -> list[dict]:
    """The global Settings page: one group per shared folder, in precedence
    order, each skill folder once (in the first folder it is seen in), with
    its switch id, ``enabled`` and ``hidden_by``."""
    if disabled is None:
        disabled = _skills.read_disabled_skills()
    folders = shared_skill_folders(home=home, global_dir=global_dir)
    groups = {source: [] for source, _ in folders}
    by_id: dict[str, dict] = {}
    for row, state, hidden_by in _walk(folders, disabled):
        if state == "linked":
            # A symlink of a row listed earlier: named on that row instead.
            first = by_id.get(hidden_by)
            if first is not None and row["source"] not in first["linked_from"]:
                first["linked_from"].append(row["source"])
            continue
        sid = f"{row['source']}:{row['key']}"
        out = {
            "id": sid, "name": row["name"], "description": row["description"],
            "source": row["source"], "agent": _AGENT_OF_SOURCE[row["source"]],
            "linked_from": [],
            "path": row["path"], "dir_name": row["dir_name"],
            "editable": row["source"] == "global", "enabled": sid not in disabled,
            "hidden_by": hidden_by,
        }
        by_id.setdefault(sid, out)
        groups[row["source"]].append(out)
    return [{
        "source": source, "path": root, "editable": source == "global",
        "exists": os.path.isdir(root),
        "skills": sorted(groups[source], key=lambda r: r["id"]),
    } for source, root in folders]


def shared_read_roots() -> list[str]:
    """The shared skill folders a safe-mode manager's read tool may open:
    each folder itself (realpath), when it exists; a home folder only while
    one of its skills is offered (switched on and not hidden)."""
    out = []
    for group in shared_skill_groups():
        if not group["exists"]:
            continue
        if group["source"] != "global" and not any(
                r["enabled"] and r["hidden_by"] is None for r in group["skills"]):
            continue
        real = os.path.realpath(group["path"])
        if real not in out:
            out.append(real)
    return out


class SkillCatalog:
    """Aggregate the skill sources for one project workspace."""

    def __init__(self, workspace: str, *, home: str | None = None,
                 global_dir=_UNSET, disabled=None):
        from agent_os.agent.project_paths import ProjectPaths
        self.workspace = workspace
        self._disabled = disabled
        self._sources: list[tuple[str, str]] = []
        if workspace:
            self._sources += [
                ("project", ProjectPaths(workspace).skills_dir),
                ("workspace", os.path.join(workspace, ".claude", "skills")),
                ("workspace", os.path.join(workspace, ".agents", "skills")),
            ]
        self._sources += shared_skill_folders(home=home, global_dir=global_dir)

    def scan(self) -> list[dict]:
        """``[{key, name, description, source, path, dir_name}]``, sorted by key.

        ``path`` is the absolute SKILL.md path inside its source root (not the
        resolved symlink target), which is what an agent is told to read.
        """
        disabled = (self._disabled if self._disabled is not None
                    else _skills.read_disabled_skills())
        out = [row for row, state, _ in _walk(self._sources, disabled) if state == "ok"]
        out.sort(key=lambda e: e["key"])
        return out

    def resolve(self, path: str) -> dict | None:
        """The catalog entry whose ``path`` is exactly ``path``, else None."""
        if not path:
            return None
        for entry in self.scan():
            if entry["path"] == path:
                return entry
        return None

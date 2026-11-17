# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Skill metadata scanner for Agent OS.

Scans workspace/skills/{name}/SKILL.md files and returns metadata, the
global folder every project shares (``{data_dir}/skills``), and the install
code the project and global upload routes share.
"""

import json
import os
import re
import shutil
import tempfile
import threading
import zipfile

import yaml

from agent_os.utils.atomic_write import atomic_write_text

# Frontmatter lines read before giving up on a closing ``---``, and the H1
# window after it (the H1 window was the whole parse before spec 132).
_FRONTMATTER_MAX_LINES = 200
_H1_WINDOW_LINES = 20


def _one_line(value) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def parse_skill_md(filepath: str) -> dict | None:
    """Read a SKILL.md's name and description, metadata only.

    YAML frontmatter (``name:`` / ``description:`` between leading ``---``
    lines, the Claude Code / Codex layout) wins. Whatever it does not supply
    falls back to the first ``# `` H1 and the first non-blank line after it,
    within 20 lines of the frontmatter's end — the only format before spec
    132, so a file without frontmatter parses exactly as it did.

    Returns ``{"name": str | None, "description": str | None}``, or None when
    the file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= _FRONTMATTER_MAX_LINES + _H1_WINDOW_LINES:
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError:
        return None
    return _parse_lines(lines)


def parse_skill_text(text: str) -> dict:
    """``parse_skill_md`` for text in hand (an upload before it is written)."""
    lines = text.splitlines()[:_FRONTMATTER_MAX_LINES + _H1_WINDOW_LINES]
    return _parse_lines(lines)


def _parse_lines(lines: list[str]) -> dict:
    name = None
    description = None
    body_start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, min(len(lines), _FRONTMATTER_MAX_LINES))
                    if lines[i].strip() == "---"), None)
        if end is not None:
            body_start = end + 1
            try:
                meta = yaml.safe_load("\n".join(lines[1:end]))
            except yaml.YAMLError:
                meta = None
            if isinstance(meta, dict):
                name = _one_line(meta.get("name"))
                description = _one_line(meta.get("description"))

    h1 = None
    for line in lines[body_start:body_start + _H1_WINDOW_LINES]:
        line = line.rstrip()
        if line.startswith("# ") and h1 is None:
            h1 = line[2:].strip()
            if name is None:
                name = h1 or None
        elif h1 is not None and line.strip():
            if description is None:
                description = line.strip()
            break

    return {"name": name, "description": description}


# ---------------------------------------------------------------------------
# The global folder (every project's skills, beside projects.json)
# ---------------------------------------------------------------------------

# Set once at startup by the global skills routes (``create_app``) to
# ``project_paths.global_skills_dir(data_dir)``. None (tests, tools that never
# start a daemon) = no global skills: every prompt and catalog is exactly what
# it was before the folder existed.
_global_skills_dir: str | None = None
_UNSET = object()


def configure_global_skills_dir(path: str | None) -> None:
    global _global_skills_dir
    _global_skills_dir = os.path.abspath(path) if path else None


def global_skills_dir() -> str | None:
    return _global_skills_dir


def skills_home() -> str:
    """The home whose ~/.agents, ~/.claude, ~/.codex skill folders Orbital
    reads. ``ORBITAL_SKILLS_HOME`` overrides it (the unit suite points it at
    an empty folder so a developer's own skills never reach a test)."""
    return os.environ.get("ORBITAL_SKILLS_HOME") or os.path.expanduser("~")


def skill_id(source: str, name: str) -> str:
    """A shared skill's switch id: ``<source>:<lowercased name>``."""
    return f"{source}:{normalize_skill_name(name)}"


# Switched-off shared skills (global + home folders), beside the global
# folder: ``{data_dir}/disabled-skills.json``, a JSON list of ids. Absent =
# everything on. An older build never reads it.
_DISABLED_FILE = "disabled-skills.json"
_disabled_lock = threading.Lock()


def disabled_skills_path() -> str | None:
    if not _global_skills_dir:
        return None
    return os.path.join(os.path.dirname(_global_skills_dir), _DISABLED_FILE)


def read_disabled_skills() -> set[str]:
    path = disabled_skills_path()
    if not path:
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return set()
    if not isinstance(data, list):
        return set()
    return {x for x in data if isinstance(x, str)}


def set_skill_enabled(sid: str, enabled: bool) -> None:
    """Switch one shared skill on or off (atomic rewrite of the list)."""
    path = disabled_skills_path()
    if not path:
        raise RuntimeError("global skills folder not configured")
    with _disabled_lock:
        disabled = read_disabled_skills()
        if enabled:
            disabled.discard(sid)
        else:
            disabled.add(sid)
        atomic_write_text(path, json.dumps(sorted(disabled), ensure_ascii=False, indent=1) + "\n")


def normalize_skill_name(name: str) -> str:
    """The key two skills collide on (the ``/`` catalog's rule)."""
    return name.strip().lower()


def skill_md_inside(dir_path: str, skill_md: str) -> bool:
    """A ``SKILL.md`` that links out of its own folder is never read."""
    parent = os.path.realpath(dir_path).rstrip(os.sep) + os.sep
    return os.path.realpath(skill_md).startswith(parent)


class SkillLoader:
    """Scan for skill definitions in the workspace (and the global folder)."""

    def __init__(self, workspace: str, *, global_dir=_UNSET, home: str | None = None):
        from agent_os.agent.project_paths import ProjectPaths
        self.workspace = workspace
        self._skills_dir = ProjectPaths(workspace).skills_dir
        self._global_dir = _global_skills_dir if global_dir is _UNSET else global_dir
        self._home = home

    def scan(self) -> list[dict]:
        """Scan for SKILL.md files. Return metadata only (not full content).

        Returns [{"name": str, "description": str, "path": str}, ...]
        Returns [] if no skills directory. Sorted by directory name, so the
        system prompt's skill list is stable across scans. Project skills
        only: Settings deletes what this lists.
        """
        if not os.path.isdir(self._skills_dir):
            return []
        results = []
        for entry in sorted(os.scandir(self._skills_dir), key=lambda e: e.name):
            if entry.is_dir():
                skill_md = os.path.join(entry.path, "SKILL.md")
                if os.path.isfile(skill_md):
                    meta = self._parse_front_matter(skill_md)
                    if meta:
                        meta["path"] = skill_md
                        meta["dir_name"] = entry.name
                        results.append(meta)
        return results

    def scan_shared(self, shadowed_by: list[dict] = ()) -> list[dict]:
        """Skills every project shares, for the manager's listing: Orbital's
        global folder, then the user's ~/.agents, ~/.claude and ~/.codex skill
        folders. Switched-off ones are left out, and so is any whose name
        matches one in ``shadowed_by`` (the project's own) or an earlier
        shared one. Global rows come first (by folder), then home rows (by
        name), so the listing is the same text from session to session."""
        from agent_os.agent.skill_catalog import scan_shared
        return scan_shared(shadowed_by=shadowed_by, global_dir=self._global_dir,
                           home=self._home, require_description=True)

    def _parse_front_matter(self, filepath: str) -> dict | None:
        """Name and description, or None unless both are found.

        Expected format: YAML frontmatter with ``name`` / ``description``, or
        # Skill Name
        Description text here.
        """
        meta = parse_skill_md(filepath)
        if meta and meta["name"] and meta["description"]:
            return {"name": meta["name"], "description": meta["description"]}
        return None


# ---------------------------------------------------------------------------
# Installing a skill (shared by the project and the global upload routes)
# ---------------------------------------------------------------------------


class SkillInstallError(Exception):
    """An install refused; the route answers ``status_code`` + ``detail``."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_skill_heading(text: str) -> tuple[str, str]:
    """The project upload's rule: a ``# `` heading in the first 20 lines and
    the next non-blank line as its description."""
    name = None
    description = None
    for line in text.splitlines()[:20]:
        stripped = line.rstrip()
        if stripped.startswith("# ") and name is None:
            name = stripped[2:].strip()
        elif name is not None and stripped.strip():
            description = stripped.strip()
            break

    if not name or not description:
        raise SkillInstallError(400, "SKILL.md must have a # heading and a description line")
    return name, description


def validate_skill_text(text: str) -> tuple[str, str]:
    """What the loader lists: front matter ``name``/``description``, or the
    heading form. A file this accepts is one every prompt will list."""
    meta = parse_skill_text(text)
    if not meta["name"] or not meta["description"]:
        raise SkillInstallError(
            400, "SKILL.md needs a name and a description (front matter, "
                 "or a # heading and a line after it)")
    return meta["name"], meta["description"]


def sanitize_skill_dir_name(name: str) -> str:
    """Convert a skill name to a safe directory name (ASCII only)."""
    sanitized = re.sub(r'[^a-z0-9_-]', '-', name.lower()).strip('-')
    if not sanitized:
        raise SkillInstallError(400, "Skill name produces invalid directory name")
    return sanitized


_WINDOWS_RESERVED = {"con", "prn", "aux", "nul",
                     *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}


def sanitize_unicode_dir_name(name: str) -> str:
    """Like ``sanitize_skill_dir_name`` but letters of any script survive
    (a Chinese skill name gets a Chinese folder, not a 400)."""
    sanitized = re.sub(r"[^\w-]", "-", name.lower()).strip("-_")
    if not sanitized:
        raise SkillInstallError(400, "Skill name produces invalid directory name")
    if sanitized in _WINDOWS_RESERVED:
        sanitized += "-skill"
    return sanitized


def place_skill_dir(skills_dir: str, dir_name: str, fill, *, overwrite: bool = False) -> bool:
    """Build ``skills_dir/dir_name`` with ``fill(path)`` and move it in.

    ``fill`` builds the folder in a staging folder beside ``skills_dir``
    (never inside it, so no scan sees a half-copied skill), then one rename
    puts it in place. An existing skill is a 409 unless ``overwrite``; then
    it is swapped out by rename and removed only once the new one is in.
    Returns True when an existing skill was replaced.
    """
    dest = os.path.join(skills_dir, dir_name)
    root_real = os.path.realpath(skills_dir).rstrip(os.sep) + os.sep
    if not os.path.realpath(dest).startswith(root_real) and not os.path.islink(dest):
        raise SkillInstallError(400, "Invalid skill name")
    existed = os.path.lexists(dest)
    if existed and not overwrite:
        raise SkillInstallError(409, f"Skill already exists: {dir_name}")
    os.makedirs(skills_dir, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(skills_dir))
    staging = tempfile.mkdtemp(prefix=".skills-staging-", dir=parent)
    try:
        new_dir = os.path.join(staging, "new")
        fill(new_dir)
        if existed:
            old_dir = os.path.join(staging, "old")
            os.replace(dest, old_dir)
            try:
                os.replace(new_dir, dest)
            except OSError:
                os.replace(old_dir, dest)
                raise
        else:
            try:
                os.replace(new_dir, dest)
            except OSError:
                if os.path.lexists(dest):
                    raise SkillInstallError(409, f"Skill already exists: {dir_name}")
                raise
    finally:
        # rmtree never follows a symlinked old skill out of staging.
        shutil.rmtree(staging, ignore_errors=True)
    return existed


def _too_large(files: int, total: int, limits) -> bool:
    if limits is None:
        return False
    max_files, max_bytes = limits
    return files > max_files or total > max_bytes


def install_skill_upload(skills_dir: str, filename: str, content: bytes, *,
                         validate, sanitize, name_from_skill: bool = False,
                         overwrite: bool = False, limits=None) -> dict:
    """Install an uploaded ``.md`` (the SKILL.md) or ``.zip`` into
    ``skills_dir``. ``validate(text) -> (name, description)`` and
    ``sanitize(name) -> dir name`` are the caller's rules; a zip's own folder
    name is kept as-is unless ``name_from_skill`` (then the folder is always
    ``sanitize(name)``: macOS ``zip`` stores names without the UTF-8 flag, so
    a zip's folder name can arrive as cp437 mojibake). ``limits`` =
    ``(max_files, max_unpacked_bytes)`` for a zip (413 past either).
    Returns ``{name, description, path, dir_name, replaced}``."""
    if filename.endswith(".zip"):
        return _install_zip(skills_dir, content, validate=validate, sanitize=sanitize,
                            name_from_skill=name_from_skill, overwrite=overwrite,
                            limits=limits)
    if filename.endswith(".md"):
        return _install_md(skills_dir, content, validate=validate, sanitize=sanitize,
                           overwrite=overwrite)
    raise SkillInstallError(400, "File must be .zip or .md")


def _row(skills_dir, name, description, dir_name, replaced) -> dict:
    return {"name": name, "description": description,
            "path": os.path.join(skills_dir, dir_name, "SKILL.md"),
            "dir_name": dir_name, "replaced": replaced}


def _install_md(skills_dir, content, *, validate, sanitize, overwrite) -> dict:
    text = content.decode("utf-8", errors="replace")
    name, description = validate(text)
    dir_name = sanitize(name)

    def fill(new_dir):
        os.makedirs(new_dir)
        atomic_write_text(os.path.join(new_dir, "SKILL.md"), text)

    replaced = place_skill_dir(skills_dir, dir_name, fill, overwrite=overwrite)
    return _row(skills_dir, name, description, dir_name, replaced)


def _install_zip(skills_dir, content, *, validate, sanitize, name_from_skill,
                 overwrite, limits) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(content)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Guard against zip slip (path traversal)
                real_tmpdir = os.path.realpath(tmpdir)
                for member in zf.namelist():
                    member_path = os.path.realpath(os.path.join(tmpdir, member))
                    if not member_path.startswith(real_tmpdir + os.sep) and member_path != real_tmpdir:
                        raise SkillInstallError(400, "Zip contains path traversal entry")
                infos = zf.infolist()
                if _too_large(len(infos), sum(i.file_size for i in infos), limits):
                    raise SkillInstallError(413, "Skill is too large")
                zf.extractall(tmpdir)
        except zipfile.BadZipFile:
            raise SkillInstallError(400, "Invalid zip file")

        # Find SKILL.md in extracted content
        skill_md_path = None
        for root, _dirs, files in os.walk(tmpdir):
            if "SKILL.md" in files:
                skill_md_path = os.path.join(root, "SKILL.md")
                break

        if skill_md_path is None:
            raise SkillInstallError(400, "No SKILL.md found in zip")

        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            raise SkillInstallError(400, "SKILL.md must be UTF-8 text")

        name, description = validate(text)

        # The skill's folder: the zip's own folder holding SKILL.md, or (a
        # SKILL.md at the zip's root) one named after the skill.
        skill_src_dir = os.path.dirname(skill_md_path)
        at_root = os.path.normpath(skill_src_dir) == os.path.normpath(tmpdir)
        if at_root or name_from_skill:
            dir_name = sanitize(name)
        else:
            dir_name = os.path.basename(skill_src_dir)

        def fill(new_dir):
            # Just SKILL.md from a zip's root; the whole folder otherwise.
            if at_root:
                os.makedirs(new_dir)
                shutil.copy2(skill_md_path, os.path.join(new_dir, "SKILL.md"))
            else:
                shutil.copytree(skill_src_dir, new_dir)

        replaced = place_skill_dir(skills_dir, dir_name, fill, overwrite=overwrite)
        return _row(skills_dir, name, description, dir_name, replaced)


def install_skill_dir(src_dir: str, skills_dir: str, *, validate, sanitize,
                      overwrite: bool = False, limits=None) -> dict:
    """Copy an existing skill folder (a project's) into ``skills_dir``,
    in a folder named ``sanitize(name)``."""
    skill_md = os.path.join(src_dir, "SKILL.md")
    if not os.path.isfile(skill_md) or not skill_md_inside(src_dir, skill_md):
        raise SkillInstallError(404, "Skill not found")
    with open(skill_md, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    name, description = validate(text)
    files = total = 0
    for root, _dirs, names in os.walk(src_dir, followlinks=True):
        for n in names:
            files += 1
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
        if _too_large(files, total, limits):
            raise SkillInstallError(413, "Skill is too large")
    dir_name = sanitize(name)
    replaced = place_skill_dir(skills_dir, dir_name,
                               lambda new_dir: shutil.copytree(src_dir, new_dir),
                               overwrite=overwrite)
    return _row(skills_dir, name, description, dir_name, replaced)

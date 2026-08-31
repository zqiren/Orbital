# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""System prompt renderer for sub-agents.

Builds the inheritance template that lets sub-agents read project state
(PROJECT_STATE.md, DECISIONS.md, etc.) and their own MEMORY.md without
relying on management-agent context.

The renderer is path-aware (paths are substituted from the workspace) but
does NOT inline the contents of those files. Sub-agents read on demand.
The rendered string is freshly produced per dispatch — callers MUST NOT
cache it across dispatches; freshness on layer-1 file paths matters.
"""

import os

from agent_os.agent.project_paths import ProjectPaths


# MEMORY.md header for newly created sub-agent memory files. Used by the
# daemon when it lazily creates a sub-agent's MEMORY.md on first dispatch.
MEMORY_MD_HEADER = (
    "# MEMORY for {agent_slug}\n"
    "\n"
    "This is your long-term memory across dispatches in this project.\n"
    "Append entries below.\n"
    "\n"
    "---\n"
    "\n"
)


def _memory_md_path(workspace: str, agent_slug: str) -> str:
    """Path to a sub-agent's MEMORY.md file."""
    return os.path.join(
        ProjectPaths(workspace).sub_agent_dir(agent_slug),
        "MEMORY.md",
    )


def _pinned_retraction_titles(workspace: str) -> list[str]:
    """Titles of every user retraction, for the pinned-mode contract.

    Closes the append-resurrection window (spec 074 §3.5): a pinned worker
    writes Layer-1 directly, outside the chokepoint that reconciles
    retractions, so the prompt itself must carry the "never add entries
    about" list until the next consolidation pass. Best-effort: an unreadable
    file yields an empty list, never a render failure.
    """
    try:
        from agent_os.agent.retractions import list_retractions
        rs = list_retractions(ProjectPaths(workspace).orbital_dir)
    except Exception:
        return []
    return [r.title for r in rs if r.title]


def _list_skill_files(skills_dir: str) -> list[str]:
    """Return sorted skill .md filenames at the top level of skills_dir.

    We only emit names, not contents (per spec §5: skills index in prompt).
    Recurses one level deep into skill subdirectories to surface SKILL.md.
    """
    if not os.path.isdir(skills_dir):
        return []
    names: list[str] = []
    try:
        for entry in sorted(os.listdir(skills_dir)):
            full = os.path.join(skills_dir, entry)
            if os.path.isfile(full) and entry.endswith(".md"):
                names.append(entry)
            elif os.path.isdir(full):
                # Sub-directory style: skills/{name}/SKILL.md
                skill_md = os.path.join(full, "SKILL.md")
                if os.path.isfile(skill_md):
                    names.append(f"{entry}/SKILL.md")
    except OSError:
        return []
    return names


def render_sub_agent_prompt(
    workspace: str,
    namespace: str | None,
    agent_slug: str,
    enabled_sub_agents: list[str],
    *,
    pinned: bool = False,
) -> str:
    """Render the sub-agent inheritance system prompt.

    Args:
        workspace: Absolute path to the project workspace.
        namespace: Reserved for future per-namespace overrides; unused in v1.
        agent_slug: The sub-agent's slug (e.g. 'claude-code').
        enabled_sub_agents: Slugs of all enabled sub-agents in this project,
            including the current one. Peers whose MEMORY.md does not yet
            exist on disk are omitted from the rendered prompt to avoid
            steering the agent to issue Read calls that fail.
        pinned: Spec 074 — this spawn serves a chat session pinned directly
            to this worker (no management agent mediates). The Layer-1 ban is
            REPLACED by the narrow append/targeted-edit write contract plus
            the current retraction titles. Non-pinned output is byte-identical
            to before the flag existed.

    Returns:
        The full rendered system prompt string. Callers must not cache it.
    """
    paths = ProjectPaths(workspace)
    orbital = paths.orbital_dir
    own_memory = _memory_md_path(workspace, agent_slug)

    # Collect peers (other sub-agents) that have an existing MEMORY.md.
    # Per dispatcher disambiguation #1: enumerate ONLY peers whose
    # MEMORY.md actually exists at dispatch time.
    peers_with_memory: list[str] = []
    for peer_slug in enabled_sub_agents or []:
        if peer_slug == agent_slug:
            continue
        peer_path = _memory_md_path(workspace, peer_slug)
        if os.path.isfile(peer_path):
            peers_with_memory.append(peer_path)

    # Skills index: filenames only (not contents). Per dispatcher
    # disambiguation #3: render compact one-liner.
    skill_filenames = _list_skill_files(paths.skills_dir)
    if skill_filenames:
        skills_line = "Available skills: " + ", ".join(skill_filenames)
    else:
        skills_line = "Available skills: (none)"

    lines: list[str] = []
    lines.append(f"You are a sub-agent dispatched within an Orbital project at {workspace}.")
    lines.append("")
    lines.append("Before responding to any non-trivial request, read these files:")
    lines.append(f"- {paths.project_state}   — current task state")
    lines.append(f"- {paths.decisions}       — project decisions to respect")
    lines.append(f"- {paths.lessons}       — past learnings")
    lines.append(f"- {paths.index}         — navigation map (file tree + one line per file)")
    lines.append(f"- {paths.instructions_dir}{os.sep}      — user directives and project goals")
    lines.append(f"- {paths.skills_dir}{os.sep}            — reusable skills available in this project")
    lines.append("")
    lines.append(skills_line)
    lines.append("")
    lines.append(
        "These are authoritative. Do not ignore them. If they conflict with the user's"
    )
    lines.append("immediate request, ask for clarification rather than proceeding.")
    lines.append("")
    lines.append("You have your own long-term memory at:")
    lines.append(f"  {own_memory}")
    lines.append("")
    lines.append(
        "Append to this file when you discover something worth remembering across"
    )
    lines.append(
        "dispatches — working approaches, things to avoid, recurring patterns."
    )
    lines.append(
        "This is your CLAUDE.md-equivalent for this project. Keep entries concise"
    )
    lines.append(
        "(target: under 2KB total). It is NOT for storing work artifacts; it is for"
    )
    lines.append("your own curated memory.")

    # "Other sub-agents" section is rendered ONLY when at least one peer
    # has an existing MEMORY.md. Per dispatcher disambiguation #1:
    # "When ZERO peers have memory files, omit the entire 'Other sub-agents'
    # section, not render an empty list."
    if peers_with_memory:
        lines.append("")
        lines.append(
            "Other sub-agents in this project also have memory files. Read them when"
        )
        lines.append("relevant context may live there:")
        for peer_path in peers_with_memory:
            lines.append(f"- {peer_path}")

    if pinned:
        # Spec 074 §3.5/§3.6 — pinned-mode enrichment. Replaces the standard
        # Layer-1 ban below for pinned spawns ONLY: the user is talking to
        # this worker directly, so it is the session's only Layer-1 writer
        # and gets a narrow append/targeted-edit contract ("workers add, the
        # system tidies"). A background consolidation pass stamps/dedupes.
        lines.append("")
        lines.append(
            "PINNED MODE: you are conversing directly with the user; no "
            "manager is mediating this conversation."
        )
        lines.append(
            f"The user's standing goals and directives live under "
            f"{paths.instructions_dir}{os.sep} — read them; they apply to "
            f"you directly."
        )
        lines.append("")
        lines.append("Project memory write contract (pinned mode):")
        lines.append(
            f"- APPEND entries to {paths.decisions} and {paths.lessons} when "
            "a decision lands or a lesson is learned."
        )
        lines.append(
            f"- APPEND to or make TARGETED EDITS of individual entries in "
            f"{paths.project_state} to keep current state accurate."
        )
        lines.append(
            "- Write plain markdown bullets — NO metadata comment grammar "
            "(no ids, no <!--mem--> comments); the system stamps entries "
            "mechanically."
        )
        lines.append(
            "- NEVER reorganize, dedupe, trim, or rewrite these files "
            f"wholesale, and NEVER touch {paths.index} — cleanup and project "
            "chores stay with the system. Workers add, the system tidies."
        )
        retraction_titles = _pinned_retraction_titles(workspace)
        if retraction_titles:
            lines.append(
                "- The user retracted these topics — NEVER add entries "
                "about: " + "; ".join(retraction_titles) + ". They may only "
                "return by explicit user request."
            )
        lines.append(
            "Do not modify the other orbital-managed files (instructions/, "
            "sub_agents/ other than your own MEMORY.md)."
        )
        lines.append("")
    else:
        lines.append("")
        lines.append("Do NOT modify these orbital-managed files:")
        lines.append(
            "PROJECT_STATE.md, DECISIONS.md, LESSONS.md, INDEX.md,"
        )
        lines.append(
            "instructions/, sub_agents/ (other than your own MEMORY.md)."
        )
        lines.append("You can read them; do not write to them.")
        lines.append("")
    lines.append(
        "For your work artifacts (reports, generated files, screenshots, etc.),"
    )
    lines.append(
        "place them in appropriate paths within the workspace based on the"
    )
    lines.append("artifact's nature. Use your judgment.")

    # Suppress lint about unused arg without changing the public signature;
    # `namespace` is reserved for future per-namespace overrides.
    _ = namespace
    _ = orbital

    rendered = "\n".join(lines)

    # Windows workaround: claude.exe v2.1.138 hangs when --append-system-prompt
    # (or --system-prompt) receives an argv value containing \n. Flatten to a
    # single-line string before passing to the SDK. See:
    #   docs/investigations/TRIGGER-orbital-prompt-content.md
    #   docs/investigations/BOUNDARY-claude-exe-hang.md
    # Remove when claude-code addresses the Windows newline-in-argv bug.
    return rendered.replace("\n", "; ")


def ensure_memory_md(workspace: str, agent_slug: str) -> str:
    """Lazily create the sub-agent's MEMORY.md if it does not exist.

    Returns the absolute path. Idempotent: existing files are left alone.
    """
    path = _memory_md_path(workspace, agent_slug)
    if os.path.isfile(path):
        return path
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(MEMORY_MD_HEADER.format(agent_slug=agent_slug))
    return path


# ---------------------------------------------------------------------------
# Workspace CLAUDE.md interference detection
# ---------------------------------------------------------------------------

# Case-insensitive substring tokens that, when present in workspace
# CLAUDE.md, indicate the user has likely written instructions that may
# fight Orbital's project-state inheritance directives. See spec §4.
_CONFLICT_TOKENS = (
    "skip reading",
    "don't read",
    "do not read",
    "ignore project",
    "ignore orbital",
)


def detect_claudemd_conflict(workspace: str) -> dict | None:
    """Inspect {workspace}/CLAUDE.md for conflicting tokens.

    Returns:
        None if the file does not exist or contains no conflicting tokens.
        Otherwise: {
            "claudemd_path": <abs path>,
            "content_hash": <sha256 hex digest of full file>,
            "matched_token": <first matched token (lowercased)>,
        }
    """
    import hashlib

    path = os.path.join(workspace, "CLAUDE.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    lower = content.lower()
    matched: str | None = None
    for token in _CONFLICT_TOKENS:
        if token in lower:
            matched = token
            break
    if matched is None:
        return None

    h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return {
        "claudemd_path": path,
        "content_hash": h,
        "matched_token": matched,
    }


def hash_claudemd(workspace: str) -> str | None:
    """Return the sha256 hex digest of {workspace}/CLAUDE.md, or None."""
    import hashlib

    path = os.path.join(workspace, "CLAUDE.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()

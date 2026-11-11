# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The import scan's text (spec 160): read an imported project once, write
the best guess at its goals, state and index, then stop.

Spec 167 removed the interview and the card-driven scan: every other project
gets its goals from its first run's record turn. One source for two readers:

- the management agent, whose system prompt carries this text in its
  ``## COLD-START WORKSPACE SCAN`` section for the scan run
  (``PromptBuilder._onboarding_or_directive``);
- a pinned worker CLI (Claude Code, Codex, …) on a keyless project, where
  Orbital's own agent has no key and the worker runs the scan instead. The
  worker flavour is the same text behind a short preamble that tells the
  worker who it is, which tools to use, and the absolute paths to write.

Pure functions only: no I/O, no project lookups.
"""

from __future__ import annotations

import os
import pathlib

# Paths that go INTO a prompt use forward slashes on every platform (the
# convention of ffa7653): agents' file tools accept them, and the text stays
# the same across OSes. Module-level so a test can exercise the Windows form.
_WINDOWS = os.name == "nt"

# Upper bound on the workspace skeleton inlined into a worker's scan
# instruction. The manager gets the skeleton as its own system row; a worker
# gets it in the one message it receives, so it is cut to keep that message
# a reasonable size.
SKELETON_CHAR_LIMIT = 40_000


# ---------------------------------------------------------------------------
# Manager flavour
# ---------------------------------------------------------------------------


# Spec 160 §5.4: the one line the first message carries when the user has
# more than one agent here. Verbatim; the model picks the user's language.
SHARED_MEMORY_LINE_ZH = (
    "这个项目里，不管用哪个 Agent、开几个会话，都用同一份项目记忆。"
    "随时切换 Agent 或新开会话，不用重新解释。"
)
SHARED_MEMORY_LINE_EN = (
    "Every agent and every session in this project works from the same project "
    "memory. Switch agents or open a new session anytime; no need to explain again."
)


def _shared_memory_bullet(indent: str, lang: str) -> str:
    """The §5.4 line as a prompt bullet, in the reply's language."""
    line = SHARED_MEMORY_LINE_ZH if lang == "zh" else SHARED_MEMORY_LINE_EN
    return f"{indent}- Include this one line exactly as written: {line}\n"


def _guess_reply(lang: str, multi_agent: bool, agent_picks: bool = False) -> str:
    """How the guess-first run ends. Round 4: no closing question — the
    daemon posts the confirm card itself once the goals file exists. Round 5:
    the summary ends with the 1-3 next steps the run recorded under
    PROJECT_STATE's ``## Next steps`` (the card offers step 1 from there);
    none is started."""
    language = "Chinese (简体中文)" if lang == "zh" else "English"
    # With a LANGUAGE section (2026-11-07) the agent picks the language.
    where = "the language chosen under LANGUAGE below" if agent_picks else language
    text = (
        f"- Reply to the user in {where}, plain words, no file names, at most 5 short\n"
        "  lines before the steps: what you think this project is for and where it stands.\n"
    )
    if multi_agent:
        text += _shared_memory_bullet("", "zh" if lang == "zh" else "en")
    return text + (
        "- End the reply with the same steps, numbered (1. …), in the same order as in\n"
        "  PROJECT_STATE.md; none when you wrote none.\n"
        "- Do not ask a question and do not start any step: Orbital asks the user whether\n"
        "  your understanding is right, and offers your step 1, once you stop.\n"
        "- That summary is your final message, as plain text, not a tool call (do not call\n"
        "  mark_task_complete). Then stop. Do not start any work or dispatch an agent in this run."
    )


def guess_first_instruction(workspace: str, *, lang: str = "en",
                            multi_agent: bool = False,
                            language: str | None = None) -> str:
    """Spec 160 §5.2: the cold-start scan of an imported project, guess
    first. Nobody is waiting on this run (the import started it): read, write
    the best guess at state, index and goals, then leave one short summary.
    Round 4: the daemon then posts the confirm card (这样理解对吗？ · 对 |
    需要改).

    The whole scan run keeps this section, goals file or not (the queue
    resets the flavour when the run ends, so a later turn gets the plain
    directive).

    ``lang`` (``zh`` | ``en``) is the language of that reply: the UI
    language the import was made in, else the system's.
    ``multi_agent``: more than one agent is enabled here, so the reply also
    carries the shared-memory line (§5.4).
    ``language``: the ``## LANGUAGE`` section (``onboarding.user_language``);
    given, the agent picks the reply language from it instead of ``lang``."""
    # On Windows the drive-letter path is given with forward slashes, like the
    # worker preamble's; elsewhere this is exactly the joined path.
    goals_path = _prompt_path(os.path.join(workspace, "orbital", "instructions",
                                           "project_goals.md"))
    return (
        "## COLD-START WORKSPACE SCAN\n\n"
        "This is an imported project with an existing workspace. The user brought it into\n"
        "Orbital; you read it once and write down your best guess of what it is, so every\n"
        "agent in this project starts from the same record. Nobody is waiting on this run:\n"
        "work through it, then leave one short message the user reads later. A deterministic\n"
        "[WORKSPACE SKELETON] (every gitignore-respected file + size) has been provided to\n"
        "you as a system message. Work through three stages:\n\n"
        "STAGE 2 — READ (informed by the skeleton sizes):\n"
        "- Use the skeleton's sizes to plan BEFORE opening files. Small total →\n"
        "  read broadly. Large total → read high-signal files (README, config,\n"
        "  entry points, the largest meaningful sources) and sample the rest.\n"
        "- You have NO precise token meter. You will see a coarse 'Context usage:'\n"
        "  line each turn. When it crosses ~70%, STOP reading and move to Stage 3\n"
        "  with what you have.\n\n"
        "STAGE 3 — WRITE YOUR GUESS → ASK:\n"
        "- Do not wait for a confirmation before writing; the guess is the point. With the\n"
        "  `write` tool, in the same language as your reply, in this order:\n"
        "  (1) orbital/PROJECT_STATE.md (current focus, in-progress work, and under a\n"
        "  `## Next steps` heading (exactly that heading, whatever the language) 1-3 concrete\n"
        "  next steps, never more than 3, as `- ` bullets, best first, each grounded in what\n"
        "  you read; leave the heading out when nothing is clearly next),\n"
        "  (2) orbital/INDEX.md (navigation map: important files/dirs, one line each),\n"
        "  (3) LAST, your best guess at the Goals to\n"
        f"  {goals_path} (Mission, Triggers, Scope,\n"
        "  Rules, Preferences; under 1500 words). Mission, Triggers and Scope are your guess\n"
        "  from the files. Rules and Preferences only where a file states them (README,\n"
        "  notes, config, imported memories); otherwise write \"(none yet)\": the user adds\n"
        "  them. Never invent a style, tone or working rule the files don't state.\n"
        + _guess_reply(lang, multi_agent, agent_picks=language is not None)
        + (f"\n\n{language}" if language else "")
    )


# ---------------------------------------------------------------------------
# Worker flavour (keyless project: a pinned worker runs the scan)
# ---------------------------------------------------------------------------


def worker_preamble(slug: str, workspace: str) -> str:
    """What a worker needs before the shared scan text."""
    goals = _prompt_path(os.path.join(workspace, "orbital", "instructions",
                                      "project_goals.md"))
    state = _prompt_path(os.path.join(workspace, "orbital", "PROJECT_STATE.md"))
    index = _prompt_path(os.path.join(workspace, "orbital", "INDEX.md"))
    lines = [
        f"You are {slug} working directly with the user in Orbital project "
        f"{_prompt_path(workspace)}. "
        "Orbital's own agent is not configured, so you run this setup. "
        "Use your own file tools. Write files at these absolute paths:",
        f"- Project goals: {goals}",
        f"- Project state: {state}",
        f"- Navigation map: {index}",
        "",
        "Wherever the setup text below says the `write` tool, use your own file-writing "
        "tool. When it asks you to wait for the user, end your reply there: the user's "
        "answer arrives as your next message in this same conversation.",
        "The [WORKSPACE SKELETON] it mentions is at the end of this message, not a "
        "separate system message, and you will not see a 'Context usage:' line: keep "
        "your reading proportionate to the skeleton's sizes and say what you skipped.",
    ]
    return "\n".join(lines) + "\n\n"


def worker_guess_first_instruction(slug: str, workspace: str, skeleton_text: str, *,
                                   lang: str = "en", multi_agent: bool = False,
                                   language: str | None = None) -> str:
    """Spec 160 round 2: the guess-first import scan on a keyless project's
    pinned worker — the manager's guess text behind the worker preamble,
    with the skeleton inlined (one message, so no directive ending)."""
    return (worker_preamble(slug, workspace)
            + guess_first_instruction(workspace, lang=lang, multi_agent=multi_agent,
                                      language=language)
            + "\n\n" + _bounded_skeleton(_prompt_separators(skeleton_text or "")))


def _prompt_path(path: str) -> str:
    """``path`` with forward slashes on Windows (drive letter kept).

    Unchanged elsewhere, so the text a macOS/Linux worker gets is exactly
    the joined path.
    """
    if _WINDOWS:
        return pathlib.PureWindowsPath(path).as_posix()
    return path


def _prompt_separators(text: str) -> str:
    """A skeleton listing with forward slashes on Windows.

    ``scan_workspace`` lists ``os.path`` relative paths, so on Windows they
    carry backslashes — a character no Windows file name can contain, which
    makes the swap lossless there. Elsewhere a backslash may be a real file
    name character, so the text is left alone.
    """
    return text.replace("\\", "/") if _WINDOWS else text


def _bounded_skeleton(skeleton_text: str) -> str:
    """The skeleton, cut at a line boundary to ``SKELETON_CHAR_LIMIT``."""
    if len(skeleton_text) <= SKELETON_CHAR_LIMIT:
        return skeleton_text
    cut = skeleton_text.rfind("\n", 0, SKELETON_CHAR_LIMIT)
    if cut <= 0:
        cut = SKELETON_CHAR_LIMIT
    omitted = skeleton_text.count("\n", cut) + 1
    return (skeleton_text[:cut]
            + f"\n[… skeleton truncated — about {omitted} more lines not shown; "
              "list the folders yourself if you need them]")

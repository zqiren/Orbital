# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""First-run onboarding text: the interview and the cold-start scan.

One source for two readers:

- the management agent, whose system prompt carries this text in its
  ``## ONBOARDING MODE`` / ``## COLD-START WORKSPACE SCAN`` section
  (``PromptBuilder._onboarding_or_directive``) — the manager flavour MUST stay
  byte-identical to what that section rendered before the text moved here
  (prompt-cache safety; pinned by ``tests/unit/test_worker_onboarding.py``);
- a pinned worker CLI (Claude Code, Codex, …) on a keyless project, where
  Orbital's own agent has no key and the worker runs the setup instead. The
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
# Manager flavour (verbatim section text)
# ---------------------------------------------------------------------------


def cold_start_instruction(workspace: str, skeleton_text: str | None = None) -> str:
    """The cold-start scan section the management agent's prompt carries.

    The manager receives the ``[WORKSPACE SKELETON]`` as a separate system
    row, so it never passes ``skeleton_text``; when given, the bounded
    skeleton is appended after the section text.
    """
    text = (
        "## COLD-START WORKSPACE SCAN\n\n"
        "This is an imported project with an existing workspace. The user has\n"
        "consented to a one-time scan. A deterministic [WORKSPACE SKELETON] (every\n"
        "gitignore-respected file + size) has been provided to you as a system\n"
        "message. Work through three stages:\n\n"
        "STAGE 2 — READ (informed by the skeleton sizes):\n"
        "- Use the skeleton's sizes to plan BEFORE opening files. Small total →\n"
        "  read broadly. Large total → read high-signal files (README, config,\n"
        "  entry points, the largest meaningful sources) and sample the rest.\n"
        "- You have NO precise token meter. You will see a coarse 'Context usage:'\n"
        "  line each turn. When it crosses ~70%, STOP reading and move to Stage 3\n"
        "  with what you have. State what you skipped.\n\n"
        "STAGE 3 — PROPOSE → CONFIRM → WRITE:\n"
        "- Propose, in chat, your read of the project (descriptive State) and a\n"
        "  DRAFT of suggested Goals. Report which files you read and which you skipped.\n"
        "- State is yours to assert. Goals are a SUGGESTION the user owns — invite edits.\n"
        "- Do NOT propose or write prescriptive Instructions; a scan cannot infer intent.\n"
        "- Write NOTHING until the user confirms (ok / yes / looks good / any affirmative).\n"
        "- On confirmation: (1) write the agreed Goals to\n"
        f"  {workspace}/orbital/instructions/project_goals.md using the `write`\n"
        "  tool (Mission, Triggers, Scope, Rules, Preferences; under 1500 words), then\n"
        "  (2) write PROJECT_STATE.md (current focus, in-progress work, next steps) and\n"
        "     INDEX.md (navigation map) in orbital/ yourself with the `write` tool.\n"
        "- After writing, say in two or three lines what you would do first and which agent\n"
        "  would do it, then stop and wait for the user's go. If the user's reply picks a\n"
        "  different agent, record it under Preferences in project_goals.md."
    )
    if skeleton_text:
        text += "\n\n" + _bounded_skeleton(skeleton_text)
    return text


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


def _shared_memory_bullet(indent: str, lang: str | None = None) -> str:
    """The §5.4 line as a prompt bullet: both languages when the user's is
    not known yet (the interview), else just the one the reply is in."""
    if lang == "zh":
        return f"{indent}- Include this one line exactly as written: {SHARED_MEMORY_LINE_ZH}\n"
    if lang == "en":
        return f"{indent}- Include this one line exactly as written: {SHARED_MEMORY_LINE_EN}\n"
    return (
        f"{indent}- Then add this one line exactly as written, in the user's language:\n"
        f"{indent}  中文: {SHARED_MEMORY_LINE_ZH}\n"
        f"{indent}  English: {SHARED_MEMORY_LINE_EN}\n"
    )


def _guess_reply(lang: str, multi_agent: bool) -> str:
    """How the guess-first run ends. Round 4: no closing question — the
    daemon posts the confirm card itself once the goals file exists. Round 5:
    the summary ends with the 1-3 next steps the run recorded under
    PROJECT_STATE's ``## Next steps`` (the card offers step 1 from there);
    none is started."""
    language = "Chinese (简体中文)" if lang == "zh" else "English"
    text = (
        f"- Reply to the user in {language}, plain words, no file names, at most 5 short\n"
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
                            multi_agent: bool = False) -> str:
    """Spec 160 §5.2: the cold-start scan of an imported project, guess
    first. Nobody is waiting on this run (the import started it): read, write
    the best guess at state, index and goals, then leave one short summary.
    Round 4: the daemon then posts the confirm card (这样理解对吗？ · 对 |
    需要改); the onboarding gate arms on the goals write.

    The whole scan run keeps this section, goals file or not (the queue
    resets the flavour when the run ends, so a later turn gets the plain
    directive).

    ``lang`` (``zh`` | ``en``) is the language of that reply: the UI
    language the import was made in, else the system's.
    ``multi_agent``: more than one agent is enabled here, so the reply also
    carries the shared-memory line (§5.4)."""
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
        "  `write` tool, in this order:\n"
        "  (1) orbital/PROJECT_STATE.md (current focus, in-progress work, and under a\n"
        "  `## Next steps` heading (exactly that heading, whatever the language) 1-3 concrete\n"
        "  next steps, never more than 3, as `- ` bullets, best first, each grounded in what\n"
        "  you read; leave the heading out when nothing is clearly next),\n"
        "  (2) orbital/INDEX.md (navigation map: important files/dirs, one line each),\n"
        "  (3) LAST, your best guess at the Goals to\n"
        f"  {workspace}/orbital/instructions/project_goals.md (Mission, Triggers, Scope,\n"
        "  Rules, Preferences; under 1500 words). Mission, Triggers and Scope are your guess\n"
        "  from the files. Rules and Preferences only where a file states them (README,\n"
        "  notes, config, imported memories); otherwise write \"(none yet)\": the user adds\n"
        "  them. Never invent a style, tone or working rule the files don't state.\n"
        + _guess_reply(lang, multi_agent)
    )


def interview_instruction(workspace: str, *, is_scratch: bool,
                          project_name: str = "",
                          project_instructions: str = "",
                          multi_agent: bool = False) -> str:
    """The onboarding-interview section the management agent's prompt carries.

    ``project_name`` / ``project_instructions`` are what the user typed when
    creating the project (the section quotes them back). ``multi_agent``
    (spec 160 §5.4): more than one agent is enabled here, so the greeting
    also carries the shared-memory line. Never on scratch, which keeps no
    project memory.
    """
    # Scratch (Quick Tasks) keeps no project memory, so the "this
    # project becomes an asset" introduction would be a false promise
    # there — it gets the plain greeting.
    if is_scratch:
        greeting = (
            "1. Greet the user briefly. Introduce yourself as their agent for this project.\n"
        )
    else:
        greeting = (
            "1. Greet the user briefly and introduce yourself as their agent for this project.\n"
            "   In that same first message, tell them what they are getting. Plain words, no file\n"
            "   names, no jargon, about 4-6 short lines in total:\n"
            "   - What you can do: name the abilities you actually have, read from your tool list\n"
            "     (for example working with the files in this folder, running commands, using the\n"
            "     web, running things on a schedule or when something changes, handing work to\n"
            "     coding agents). Mention only what is really available to you.\n"
            "   - How Orbital turns this project into an asset: as you work together you keep the\n"
            "     project's goals, where things stand, the decisions made and why, the lessons\n"
            "     learned, and a map of where everything lives — as plain files inside this folder.\n"
            "   - Why that is valuable: every new session starts from that record instead of from\n"
            "     zero, so they never have to re-explain the project. The files are theirs — readable,\n"
            "     editable, and usable by any other AI tool they open in this folder.\n"
            "   - That the work accumulates: every task leaves the project better understood, so you\n"
            "     get faster and more accurate the longer you work together.\n"
            + (_shared_memory_bullet("   ") if multi_agent else "") +
            "   Write it in the user's language and your own words. Do not recite this list, do not\n"
            "   oversell, and do not let it crowd out the question that follows.\n"
        )
    base = (
        "## ONBOARDING MODE\n\n"
        "This is a new project. No project_goals.md exists yet. Your priority is to understand\n"
        "what the user wants before doing any work.\n\n"
        "The user created this project with:\n"
        f"- Name: {project_name}\n"
        f"- Instructions: {project_instructions}\n\n"
        "YOUR TASK:\n"
        + greeting +
        "2. If the user's instructions are clear and detailed enough, confirm your understanding\n"
        "   and present a summary of how you'll operate. When sub-agents are available, the\n"
        "   summary names which agent will handle which kind of work, as a proposal the user can\n"
        "   change in one word. Ask if they want to adjust anything.\n"
        "3. If the instructions are vague or missing, ask clarifying questions about:\n"
        "   - What they want you to do (objective)\n"
        "   - How they want you to do it (preferences, constraints)\n"
        "   - What you should NOT do (boundaries)\n"
        "4. Keep it to at most 5 exchanges. Do not over-ask. If the user gives short answers,\n"
        "   work with what you have.\n"
        "   Do not promise to start the work as soon as they answer: say that after their answer\n"
        "   you will come back with a suggested first step for them to approve.\n"
        "5. Once confirmed (user says ok/yes/looks good/any affirmative, OR you've hit 5 exchanges),\n"
        f"   write project_goals.md to {workspace}/orbital/instructions/project_goals.md using the structure:\n"
        "   Mission, Triggers, Scope, Rules, Preferences.\n"
        "6. Keep project_goals.md under 1500 words. Distill, don't dump.\n"
    )
    # Scratch (quick-action) projects skip workspace mapping — they are
    # ephemeral and excluded from INDEX.md/persistence machinery
    # (see _memory and _workspace_persistence).
    if is_scratch:
        return base + (
            "\nDO NOT use any tools (read, shell, write, edit, browser, etc.) until onboarding "
            "is complete.\n"
            "The only tool call you make during onboarding is the final `write` to create "
            "project_goals.md.\n"
            "After writing project_goals.md, say in two or three lines what you would do first and\n"
            "which agent would do it, then stop and wait for the user's go. Do not start on the\n"
            "project instructions as a task: they describe the project, not an order. If the user's\n"
            "reply picks a different agent, record it under Preferences in project_goals.md."
        )
    return base + (
        "7. After writing project_goals.md, explore the workspace and build INDEX.md:\n"
        "   - List the top-level directory structure and read the key files\n"
        "     (README, config files, entry points) to understand the project\n"
        "   - Create INDEX.md (in orbital/) as a NAVIGATION MAP: the important\n"
        "     files/dirs, ONE sentence each ('path — what it is'). Not prose, a map —\n"
        "     decisions/status/lessons live in their own files, not here.\n"
        "   - If the workspace is empty, still create INDEX.md with a brief note\n"
        "     (\"No files yet — will populate as the project develops.\")\n"
        "8. Once INDEX.md is written, say in two or three lines what you would do first and which\n"
        "   agent would do it, then stop and wait for the user's go.\n\n"
        "DO NOT use any tools until goal-setting is complete. After writing project_goals.md,\n"
        "use read/list tools to explore the workspace, then write INDEX.md. After INDEX.md\n"
        "is written, propose your first step and wait for the user's go. Do not start on the\n"
        "project instructions as a task: they describe the project, not an order. If the user's\n"
        "reply picks a different agent, record it under Preferences in project_goals.md."
    )


# ---------------------------------------------------------------------------
# Worker flavour (keyless project: a pinned worker runs the setup)
# ---------------------------------------------------------------------------


def worker_preamble(slug: str, workspace: str, *, scan: bool = False) -> str:
    """What a worker needs before the shared onboarding text."""
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
    ]
    if scan:
        lines.append(
            "The [WORKSPACE SKELETON] it mentions is at the end of this message, not a "
            "separate system message, and you will not see a 'Context usage:' line: keep "
            "your reading proportionate to the skeleton's sizes and say what you skipped."
        )
    return "\n".join(lines) + "\n\n"


def worker_interview_instruction(slug: str, workspace: str, *,
                                 project_name: str = "",
                                 project_instructions: str = "",
                                 is_scratch: bool = False) -> str:
    """The interview, for a worker CLI running a keyless project's setup."""
    return worker_preamble(slug, workspace) + interview_instruction(
        workspace, is_scratch=is_scratch, project_name=project_name,
        project_instructions=project_instructions,
    )


def worker_cold_start_instruction(slug: str, workspace: str,
                                  skeleton_text: str) -> str:
    """The cold-start scan, for a worker CLI, with the skeleton inlined."""
    return (worker_preamble(slug, workspace, scan=True)
            + cold_start_instruction(workspace, None)
            + "\n\n" + _bounded_skeleton(_prompt_separators(skeleton_text or "")))


def worker_guess_first_instruction(slug: str, workspace: str, skeleton_text: str, *,
                                   lang: str = "en", multi_agent: bool = False) -> str:
    """Spec 160 round 2: the guess-first import scan on a keyless project's
    pinned worker — the manager's guess text behind the worker preamble,
    with the skeleton inlined (one message, so no directive ending)."""
    return (worker_preamble(slug, workspace, scan=True)
            + guess_first_instruction(workspace, lang=lang, multi_agent=multi_agent)
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

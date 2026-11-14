# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The goals confirm card, and the fields the retired import read left behind.

Spec 160 read every imported project once in the background (a model run per
project; keyless, one Claude Code process per project that nothing stopped:
spec 178). Batch 1114 removed that read: an import is a list write that
starts no process, spends no tokens and writes nothing into the folder until
the user works there, and the project gets its goals at the end of its first
real run (spec 167's record turn), like a new project.

What stays here is the card that asks whether the drafted goals are right
(这样理解对吗？ · 对 | 需要改, spec 160 round 4/5, posted by
``agents_v2.post_goals_confirm`` when a first run's record turn creates the
goals file) and the helpers that answer it.

**Fields an older build wrote** (``import_scan``, ``import_scan_at``,
``import_scan_activity``, ``import_scan_lang``, ``import_scan_session``,
``import_scan_attempts``) are left on the project records as they are:
nothing reads the state any more, so a project stamped ``pending``,
``deferred``, ``running`` or ``failed`` is simply a normal project, and a
downgrade to v0.22.0 finds them unchanged. ``import_scan_lang`` still picks
the confirm card's language, and ``import_scan_ask`` (``ASK_FIELD``) keys the
open card.
"""

from __future__ import annotations

import re

STATE_FIELD = "import_scan"
AT_FIELD = "import_scan_at"
ACTIVITY_FIELD = "import_scan_activity"
# The UI language at import (``zh`` | ``en``), written by v0.20-v0.22: the
# confirm card's language when the goals file does not decide it.
LANG_FIELD = "import_scan_lang"
# The session a v0.20-v0.22 read ran in.
SESSION_FIELD = "import_scan_session"
LANGS = ("zh", "en")

# The states those builds wrote. Read by nothing; kept so tests and a
# reader of an old record can name them.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
OFF = "off"
DEFERRED = "deferred"


# Round 4: the goals confirmation is a daemon-posted ask card, never the
# model's own question, so "对" can never be read as a go-ahead.
# ``(question, yes, needs-changes)``; the yes option is answered by the daemon
# alone (no run), anything else runs one guarded turn.
CONFIRM_ASK = {
    "zh": ("这样理解对吗？", "对", "需要改"),
    "en": ("Did I get this right?", "Yes", "Needs changes"),
}
# Round 5: the scan recorded next steps (PROJECT_STATE's ``## Next steps``):
# the card lists them as its detail and offers step 1.
# ``(question, go, not-now, needs-changes)``: go runs step 1 (one run, the
# user's go), not-now is the zero-call close, needs-changes the guarded turn.
CONFIRM_ASK_STEPS = {
    "zh": ("这样理解对吗？", "对，做第 1 步", "对，先不做", "需要改"),
    "en": ("Did I get this right?", "Yes, do step 1", "Yes, not now", "Needs changes"),
}
# The words step 1's run carries after the user's tap (hidden from the
# bubble, kept in context): an explicit approval naming the step.
PICKED_STEP_OPEN = "<orbital_picked_step>"
PICKED_STEP_CLOSE = "</orbital_picked_step>"
PICKED_STEP = {
    "zh": "对，按第 1 步开始：{step}",
    "en": "Yes, go ahead with step 1: {step}",
}
# A bare yes typed (or tapped) on either card is always the zero-call close
# (the round-3 bug: a bare 对 became the go-ahead). Compared after
# ``_norm_answer``.
BARE_YES = frozenset({"对", "对的", "是的", "没错", "yes", "correct"})
MAX_STEPS = 3
# On the project: {"id": <ask id>, "session_id": <scan session>,
# "yes": <zero-call option>, "change": <needs-changes option>} plus, with
# steps, "go": <do-step-1 option> and "steps": [<text>, ...].
ASK_FIELD = "import_scan_ask"

# The one-turn instruction a non-yes answer carries (folded into that user
# message; ``context.strip_turn_notes`` hides it from every later turn and the
# chat never shows it).
TURN_NOTE_OPEN = "<orbital_turn_note>"
TURN_NOTE_CLOSE = "</orbital_turn_note>"
TURN_NOTES = {
    "zh": ("用户在回答你对这个项目的理解。如果是在纠正，只改 orbital/ 下的文件"
           "（目标、PROJECT_STATE.md、INDEX.md）让它们一致；如果只点了「需要改」"
           "却没说哪里不对，就问哪里不对。这一轮不要开始任何任务，也不要动 orbital/ 以外的文件。"),
    "en": ("The user is answering your understanding of this project. If they're "
           "correcting it, edit only the files under orbital/ (goals, PROJECT_STATE.md, "
           "INDEX.md) to match; if they only tapped Needs changes without saying what, "
           "ask what's wrong. Do not start any task or touch files outside orbital/ in "
           "this turn."),
}


def confirm_ask(lang: str) -> tuple[str, str, str]:
    return CONFIRM_ASK.get(lang, CONFIRM_ASK["en"])


def confirm_ask_steps(lang: str) -> tuple[str, str, str, str]:
    return CONFIRM_ASK_STEPS.get(lang, CONFIRM_ASK_STEPS["en"])


def picked_step_block(lang: str, step: str) -> str:
    text = PICKED_STEP.get(lang, PICKED_STEP["en"]).format(step=" ".join(step.split()))
    return f"{PICKED_STEP_OPEN}{text}{PICKED_STEP_CLOSE}"


_STEPS_HEADING_RE = re.compile(r"^#{2,6}\s+.*(next\s+steps?|下一步)", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s")
_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*\S)")
# A bracketed note, or an item that opens with "none" / "nothing" / "n/a" /
# "无" / "暂无" / "没有" ("None pending; awaiting the user's next instruction." is
# the record turn's way of saying nothing is next; it must not become step 1).
_PLACEHOLDER_RE = re.compile(
    r"^[(（].*[)）]$|^(none|nothing|n/a)(?![a-z])|^(无|暂无|没有)(?:[\s.。,，;；:：]|$)"
    # "Wait for the user's actual request…" (dsh, batch 1110) is not a step
    # the card can start: nothing to do until the user speaks.
    r"|^(wait|waiting)\s+(for\s+)?(the\s+)?user\b|^等待?用户",
    re.IGNORECASE)


def next_steps(workspace: str) -> list[str]:
    """Round 5: the next steps the scan recorded — the items under the first
    ``## Next steps`` (or ``下一步``) heading of ``orbital/PROJECT_STATE.md``,
    top-level bullets or numbered lines only (indented detail and the
    daemon's mem-comments are not steps), placeholders like "(none yet)"
    dropped, at most 3. ``[]`` when the file or the section is missing.
    A file the run wrote is the record: no chat text is parsed, and a
    worker runner writes it the same way."""
    if not workspace:
        return []
    from agent_os.agent.project_paths import ProjectPaths
    try:
        with open(ProjectPaths(workspace).project_state, encoding="utf-8",
                  errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    steps: list[str] = []
    inside = False
    for line in lines:
        if _HEADING_RE.match(line):
            if inside:
                break
            inside = bool(_STEPS_HEADING_RE.match(line))
            continue
        if not inside:
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        text = " ".join(m.group(1).split())
        if _PLACEHOLDER_RE.match(text):
            continue
        steps.append(text)
        if len(steps) >= MAX_STEPS:
            break
    return steps


def _norm_answer(text: str | None) -> str:
    return " ".join((text or "").split()).strip(" 。.!！~～,，、").casefold()


def classify_answer(rec: dict, words: str | None, lang: str) -> str:
    """What the answer to the scan's confirm card asks for:

    * ``"yes"`` — the zero-call close: the card's not-now / 对 option, or a
      bare yes typed on either card;
    * ``"go"`` — the do-step-1 option: one run, the user's go;
    * ``"change"`` — the 需要改 option: one guarded turn;
    * ``"message"`` — anything else typed: the user's own message, sent as
      is (the daemon can't tell a correction from "做第 2 步").

    Options are matched as exact text (normalised for spaces, closing
    punctuation and case), so a tap and the same words typed are one answer.
    A round-4 record (no ``change``) falls back to the language's label."""
    said = _norm_answer(words)
    if not said:
        return "message"
    if said == _norm_answer(rec.get("yes")) or said in BARE_YES:
        return "yes"
    if rec.get("go") and rec.get("steps") and said == _norm_answer(rec.get("go")):
        return "go"
    change = rec.get("change") or confirm_ask(lang)[2]
    if said == _norm_answer(change):
        return "change"
    # Batch 1110 item 2: the card prefills the reply with the change option
    # ("需要改：…" / "Needs changes: …"); the words that follow are the
    # correction, and the turn is still the guarded orbital-only one.
    for sep in ("：", ":"):
        if said.startswith(_norm_answer(change) + sep):
            return "change"
    return "message"


def turn_note_block(lang: str) -> str:
    return f"{TURN_NOTE_OPEN}{TURN_NOTES.get(lang, TURN_NOTES['en'])}{TURN_NOTE_CLOSE}"


def scan_ask_of(project: dict | None, ask_id: str | None) -> dict | None:
    """The project's open goals-confirm record when ``ask_id`` is it."""
    rec = (project or {}).get(ASK_FIELD)
    if isinstance(rec, dict) and ask_id and rec.get("id") == (ask_id or "").lower():
        return rec
    return None


def guess_lang(project: dict | None) -> str:
    """The confirm card's fallback language: the UI language an older build
    stamped at import, else the system's."""
    lang = (project or {}).get(LANG_FIELD)
    if lang in LANGS:
        return lang
    from agent_os.onboarding.deferred_seed import default_lang
    return default_lang()


# Spec 169 D (R8): a draft is Chinese when CJK is a real share of its letters,
# so an English draft quoting a Chinese file name stays English.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_SHARE = 0.3


def text_lang(text: str | None) -> str | None:
    """``"zh"`` when CJK characters are at least ``CJK_SHARE`` of the
    letters in ``text``; ``None`` otherwise (the caller's rule decides)."""
    letters = sum(1 for ch in text or "" if ch.isalpha())
    if not letters:
        return None
    cjk = len(_CJK_RE.findall(text or ""))
    return "zh" if cjk / letters >= CJK_SHARE else None


def goals_lang(workspace: str) -> str | None:
    """:func:`text_lang` of the project's goals file (``None`` when unreadable)."""
    if not workspace:
        return None
    from agent_os.agent.project_paths import ProjectPaths
    try:
        with open(ProjectPaths(workspace).project_goals, encoding="utf-8",
                  errors="replace") as fh:
            return text_lang(fh.read())
    except OSError:
        return None

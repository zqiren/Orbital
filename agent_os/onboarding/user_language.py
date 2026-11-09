# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which language onboarding speaks: the agent decides from signals.

The import scan's prompt (the manager's and its keyless-worker version)
carries one ``## LANGUAGE`` section listing what Orbital knows about the user's language. The agent
picks the language from those signals, so a language the interface does not
offer (Japanese, say) still works. Owner ask 2026-11-07.

The signals:

* ``ui``: the interface language the app last reported
  (``PUT /settings/ui-language``; per device, so the last one wins).
* ``device``: the browser's ``navigator.languages`` (the OS UI languages
  inside the desktop app).
* ``system``: the daemon host's locale.

The import's own interface language (``import_scan_lang``) is passed in
separately when known. Values come from the client, so each one is checked
against a tight tag pattern before it reaches a prompt.
"""

from __future__ import annotations

import locale
import re
import warnings

_TAG = re.compile(r"^[A-Za-z]{2,8}(?:[-_][A-Za-z0-9]{1,8}){0,3}(?:\.[A-Za-z0-9-]{1,16})?$")
MAX_DEVICE = 8

NAMES = {"zh": "Chinese (简体中文)", "en": "English"}


def clean_tag(value) -> str | None:
    """A language tag (``zh-CN``, ``en_US.UTF-8``) or None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if len(v) > 40 or not _TAG.match(v):
        return None
    return v


def clean_tags(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        t = clean_tag(v)
        if t and t not in out:
            out.append(t)
        if len(out) >= MAX_DEVICE:
            break
    return out


def system_locale() -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return clean_tag(locale.getdefaultlocale()[0] or "") or ""
    except Exception:
        return ""


def signals_from_settings(settings) -> dict:
    """The signals a prompt gets, from ``GlobalSettings`` (or None)."""
    return {
        "ui": clean_tag(getattr(settings, "ui_language", None)) if settings else None,
        "device": clean_tags(getattr(settings, "ui_languages", None)) if settings else [],
        "system": system_locale(),
    }


def _name(tag: str | None) -> str | None:
    if not tag:
        return None
    return NAMES.get(tag.split("-")[0].split("_")[0].lower(), tag)


def language_section(signals: dict, *, import_lang: str | None = None) -> str:
    """The ``## LANGUAGE`` section of an onboarding prompt."""
    ui = _name(clean_tag(signals.get("ui")))
    imported = _name(clean_tag(import_lang))
    device = clean_tags(signals.get("device"))
    system = clean_tag(signals.get("system"))

    lines = []
    if ui:
        lines.append(f"- Orbital's interface language: {ui}")
    if imported and imported != ui:
        lines.append(f"- The interface language when this project was imported: {imported}")
    elif imported:
        lines[-1] += " (also when this project was imported)"
    if device:
        lines.append(f"- The device's preferred languages, in order: {', '.join(device)}")
    if system:
        lines.append(f"- The system locale: {system}")
    known = "\n".join(lines) if lines else "- None known."
    fallback = "the interface language" if (ui or imported) else (
        "the language these signals point to" if (device or system) else "English")
    return (
        "## LANGUAGE\n\n"
        "Write every message the user reads in the language they are most comfortable with.\n"
        "You decide, from these signals:\n"
        f"{known}\n"
        "Once the user has written to you, use the language they write in. Before that, use\n"
        f"{fallback}, unless the device or system languages clearly point to a language the\n"
        "interface does not offer (it has only English and Chinese): then use that one.\n"
        "Fixed headings in the files you write (such as `## Next steps`) stay exactly as given."
    )

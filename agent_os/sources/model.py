# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The values adapters hand to the core, and the errors they raise.

An adapter never talks to routing, storage or the UI. It turns its service
into :class:`Item` (something new) and :class:`Resolve` (a condition that
opened an item is over, e.g. a website back up), and reports trouble with
:class:`VerifyError` (at connect time) or :class:`SourceFailure` (later).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    """One thing that arrived.

    ``id``: stable per item. The core records each id once per source, so an
    adapter may return the same items on every poll (re-read an overlapping
    window, keep no cursor).
    ``kind``: the arrival kind key it matched (``"assigned"``, ``"dm"``…);
    must be one of the source's switched-on kinds or it is dropped.
    ``subject``: what the item is about (an issue, a chat thread). Items
    with the same subject while that subject's task is still open merge into
    one history entry. Defaults to ``id``.
    ``title``: one line, shown in history and as the task's first line.
    ``body``: quoted to the agent as outside material (capped by the core).
    ``place``: bot types: ``{"kind": "dm"}`` / ``{"kind": "group", "name"}``
    / ``{"kind": "channel", "name"}``.
    """

    id: str
    kind: str
    title: str
    subject: str | None = None
    body: str = ""
    url: str | None = None
    sender: str | None = None
    place: dict | None = None
    at: str | None = None  # ISO time the service gives the item; core uses arrival time if None


@dataclass
class Resolve:
    """The condition behind the open entry for ``subject`` is over (a
    website is back up). Marks that same entry recovered; never a new task."""

    subject: str
    at: str | None = None


@dataclass
class Verified:
    """A passed connection check. ``label`` is the service's own name for
    the connection (account login, bot handle, mailbox, host).
    ``config`` holds non-secret values the check learned and the core should
    keep (merged into the source's config)."""

    label: str | None = None
    config: dict = field(default_factory=dict)


@dataclass
class Choice:
    value: str
    label: str | None = None

    def to_json(self) -> dict:
        return {"value": self.value, "label": self.label if self.label is not None else self.value}


@dataclass
class Accounts:
    """CLI types: what the local CLI holds. ``state``: ``signed_in`` |
    ``signed_out`` | ``not_installed``. ``accounts``: ``[{"login", "host",
    "active"}]``."""

    state: str
    accounts: list[dict] = field(default_factory=list)


VERIFY_REASONS = ("rejected", "not_found", "signed_out", "cli_missing", "permission",
                  "unreachable", "invalid", "error")


class VerifyError(Exception):
    """The connection check failed. ``field``: the form field / kind param to
    mark (or None). ``reason``: one of :data:`VERIFY_REASONS`. ``message``
    must never contain a secret."""

    def __init__(self, reason: str, field: str | None = None, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason if reason in VERIFY_REASONS else "error"
        self.field = field
        self.message = message or reason


# Every failure code a source can be in. The panel has a zh and en line for
# each (tests/unit/test_sources_codes.py holds both sides to this list); a
# code outside it is reported as ``error``.
FAILURE_CODES = (
    "cli_signed_out", "cli_missing", "auth_expired", "token_rejected", "repo_not_found",
    "label_not_found", "folder_not_found", "calendar_permission", "command_exit",
    "command_output", "command_timeout", "unreachable", "rate_limited", "webhook_set",
    "bot_in_use", "error",
)


class SourceFailure(Exception):
    """A source that worked has stopped working. ``code`` is one of
    :data:`FAILURE_CODES`; ``params`` are shown with it. ``retryable``:
    False when checking again can only fail again (a revoked token): the
    runner then stops until the user reconnects, and the panel offers no
    「再检查」."""

    def __init__(self, code: str, params: dict | None = None, *, retryable: bool = True,
                 message: str = ""):
        super().__init__(message or code)
        self.code = code if code in FAILURE_CODES else "error"
        self.params = dict(params or {})
        self.retryable = retryable


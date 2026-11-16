# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Source adapters: one module per type, named after the type
(``github.py`` → type ``github``), each exporting ``ADAPTER`` (an
:class:`Adapter` subclass). The registry imports them by name; adding one
touches no shared file. See ``agent_os/sources/ADAPTERS.md``.
"""

from __future__ import annotations

from agent_os.sources.model import Accounts, Choice, Item, Resolve, Verified  # noqa: F401


class Adapter:
    """Base class. Override what your type needs; the defaults refuse."""

    #: ``"poll"`` (the core calls :meth:`poll` on a timer) or ``"stream"``
    #: (the core runs :meth:`stream` and restarts it with backoff).
    mode: str = "poll"
    #: Seconds between polls; ``None`` = the catalogue's ``interval_s``.
    interval_s: int | None = None
    #: Poll types: the first successful poll of a kind records its items as
    #: seen without sending them (no flood of what was already there).
    baseline: bool = True
    #: Stream types: True when the service replays what arrived while the
    #: connection was down (Telegram), so the core records no gaps.
    lossless: bool = False
    #: Platforms this adapter runs on (``None`` = the catalogue's list).
    platforms: list[str] | None = None

    async def verify(self, ctx) -> Verified:
        """The deterministic connection check, with what the user typed.
        Return :class:`Verified` (``label`` = the service's own name for the
        connection) or raise :class:`~agent_os.sources.model.VerifyError`."""
        return Verified()

    async def poll(self, ctx) -> list:
        """Return ``[Item | Resolve]``. Raise
        :class:`~agent_os.sources.model.SourceFailure` when the source stopped
        working."""
        return []

    async def stream(self, ctx) -> None:
        """Hold the long connection: call ``ctx.connected()`` once it is up,
        ``await ctx.emit([...])`` for each arrival. Return or raise when the
        connection ends; the core restarts it with backoff. Raise
        ``SourceFailure(..., retryable=False)`` to stop until the user acts."""
        raise NotImplementedError

    async def choices(self, ctx, list_name: str, params: dict) -> list[Choice]:
        return []

    async def accounts(self) -> Accounts:
        """CLI types only: the accounts the local CLI holds."""
        return Accounts(state="not_installed")

    async def qr_start(self, ctx) -> dict:
        """QR connect: ``{"handle": <opaque, kept in the daemon>, "qr_text",
        "expires_at"}``."""
        raise NotImplementedError

    async def qr_poll(self, ctx, handle) -> dict:
        """``{"state": "pending"|"done"|"expired"|"failed",
        "label", "config": {...}, "secrets": {...}}`` (config/secrets only on
        ``done``)."""
        raise NotImplementedError

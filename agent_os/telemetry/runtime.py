# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Process-wide telemetry runtime (spec 046 §4, §10 step 4).

The emit seams (ledger append, api-key route, project store, session route,
error classifier) call the module-level ``emit``/``latch`` here instead of
threading telemetry objects through constructors. Until ``configure()`` runs
— e.g. in unit tests — every call is a silent no-op.

Toggle semantics (§6 + Q2 decision): when ``telemetry_enabled`` is off,
counters and milestones stop recording and nothing sends; ``llm_error`` rows
keep spooling locally (``always_spool=True``) because their debugging value is
local-only — they never leave the machine while the toggle is off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .identity import InstallIdentity
from .sender import DEFAULT_ENDPOINT, TelemetrySender
from .spool import Spool

_identity: InstallIdentity | None = None
_spool: Spool | None = None
_sender: TelemetrySender | None = None
_is_enabled: Callable[[], bool] = lambda: False
_fresh_install: bool = False


def configure(
    data_dir: str | Path,
    is_enabled: Callable[[], bool],
    endpoint: str = DEFAULT_ENDPOINT,
) -> TelemetrySender:
    """Wire the process singletons. Returns the sender for lifecycle control."""
    global _identity, _spool, _sender, _is_enabled, _fresh_install
    data_dir = Path(data_dir)
    _fresh_install = not (data_dir / "telemetry" / "install.json").exists()
    _identity = InstallIdentity(data_dir)
    _spool = Spool(data_dir)
    _is_enabled = is_enabled
    _sender = TelemetrySender(
        data_dir, _identity, _spool, is_enabled=is_enabled, endpoint=endpoint
    )
    return _sender


def emit(event: str, fields: dict | None = None, *, always_spool: bool = False) -> None:
    if _spool is None:
        return
    try:
        if always_spool or _is_enabled():
            _spool.append(event, fields)
    except Exception:
        pass


def latch(milestone: str) -> None:
    if _identity is None:
        return
    try:
        if _is_enabled():
            _identity.latch_milestone(milestone)
    except Exception:
        pass


def was_fresh_install() -> bool:
    return _fresh_install


def get_sender() -> TelemetrySender | None:
    return _sender


def reset_for_tests() -> None:
    global _identity, _spool, _sender, _is_enabled, _fresh_install
    if _sender is not None:
        _sender.stop()
    _identity = None
    _spool = None
    _sender = None
    _is_enabled = lambda: False
    _fresh_install = False

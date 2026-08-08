# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal aggregate-only telemetry (spec 046).

Raw events never leave the machine: the spool and identity live under
``{data_dir}/telemetry/``, and only a per-day aggregate of counters, enums and
booleans is transmitted (see ``rollup.py``). No content, paths, or IDs beyond
the random resettable ``install_id``.
"""

from .runtime import (  # noqa: F401
    configure,
    emit,
    get_sender,
    latch,
    reset_for_tests,
    was_fresh_install,
)

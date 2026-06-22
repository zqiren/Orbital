# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Removed: SESSION_LOG.md no longer exists (Layer-1 memory redesign).

This module previously locked in the write-time bounding contract for
SESSION_LOG.md — capping the on-disk file at the 10 newest `## Session`
entries via `_truncate_session_log`, surfacing the last 3 entries in the
session-end prompt, and preserving malformed files.

The Layer-1 memory redesign GENUINELY REMOVED the SESSION_LOG feature in its
entirety. There is no longer a "session_log" roster key, no SESSION_LOG.md
file, no `_truncate_session_log` / `_get_session_log_lock` helpers, no
`_SESSION_LOG_WRITE_CAP` / `_SESSION_LOG_MAX_SESSIONS` constants, and no
`session_log_entry` key in the session-end LLM contract. `build_cold_resume_context`
and `sub_agent_prompt` no longer reference it. The Layer-1 roster is now
state / decisions / lessons / index (CONTEXT.md → INDEX.md).

Every test and helper in this file exercised that removed feature, so there
is nothing left to re-express against the new behavior — the file is
intentionally empty of tests.

Coverage of the new contract lives elsewhere:
  - `tests/regression/test_layer1_memory_redesign.py` asserts the removal
    directly (`test_session_log_removed_from_roster`,
    `test_cold_resume_has_no_session_log`,
    `test_sub_agent_prompt_omits_session_log`).
  - The bounding intent (keep durable memory within budget instead of letting
    a log grow monotonically) is now served by the deterministic hard cap
    `workspace_files._apply_hard_caps` / `memory_entries.split_for_demotion`,
    which demotes coldest entries to *_ARCHIVE.md rather than truncating a log.
"""

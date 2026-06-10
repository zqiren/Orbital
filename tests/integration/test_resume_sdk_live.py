# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live SDK resume round-trip (TASK-resume-persistence, TEST RULE 2+5).

Closes the gap flagged by INVESTIGATION-session-binding-resume-and-reap-
trigger R3: the investigation validated resume fidelity via the CLI
``--resume`` flag; this exercises the SDK ``ClaudeAgentOptions(resume=…)``
path specifically, through the real transport: seed context in transport A,
reap it (stop()), resume in a fresh transport B with the captured id, and
assert prior conversation context is recalled and the session id is stable.

Requires the real claude binary + claude-agent SDK; skipped otherwise.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK

pytestmark = pytest.mark.live_daemon  # heavy: spawns real claude — opt-in

CODEWORD = "SIENNA-OTTER-3"


def _find_claude() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    for c in (Path.home() / ".claude/local/node_modules/.bin/claude",
              Path.home() / ".claude/local/claude"):
        if c.exists():
            return str(c)
    return None


CLAUDE = _find_claude()

requires_claude = pytest.mark.skipif(
    not HAS_SDK or CLAUDE is None, reason="claude SDK/binary not available")


async def _run_turn(transport: SDKTransport, message: str,
                    timeout: float = 150.0) -> tuple[str, dict]:
    """Dispatch one turn; return (collected message text, turn_complete data)."""
    await transport.dispatch(message)
    texts: list[str] = []

    async def _consume():
        async for ev in transport.read_stream():
            if ev.event_type == "message":
                texts.append(ev.raw_text)
            if ev.event_type == "turn_complete":
                return ev.data
        return {}

    data = await asyncio.wait_for(_consume(), timeout=timeout)
    return "\n".join(texts), data


@requires_claude
async def test_sdk_resume_round_trip_recalls_context(tmp_path):
    # --- Transport A: seed context, capture the resume identity ---
    a = SDKTransport(autonomy=Autonomy.HANDS_OFF,
                     system_prompt="disposable resume-persistence test agent")
    await a.start(command=CLAUDE, args=[], workspace=str(tmp_path), env={})
    try:
        _, data = await _run_turn(
            a, f"Remember this codeword for later: {CODEWORD}. "
               f"Reply with exactly: OK")
        assert data.get("cause") == "success"
        sid = data.get("session_id")
        assert sid, "turn_complete must carry the resume session id"
        assert sid == a.session_id

        # The pre-check must see the freshly written session store file.
        assert SDKTransport.resume_source_exists(str(tmp_path), sid), (
            "resume_source_exists must find the live session store — "
            "either the cwd-slug derivation or the glob fallback is broken"
        )
    finally:
        await a.stop()  # the reap

    # --- Transport B: resume by id in a fresh process ---
    b = SDKTransport(autonomy=Autonomy.HANDS_OFF,
                     system_prompt="disposable resume-persistence test agent",
                     resume_session_id=sid)
    await b.start(command=CLAUDE, args=[], workspace=str(tmp_path), env={})
    try:
        text, data = await _run_turn(
            b, "Without using any tools: what is the codeword I gave you "
               "earlier? Answer with just the codeword.")
        assert data.get("cause") == "success"
        assert CODEWORD in text, (
            f"resumed session lost prior context — reply was: {text[:200]!r}"
        )
        # Session id stays stable across the resume (no fork), so the
        # persisted record remains valid for the NEXT resume too.
        assert data.get("session_id") == sid
    finally:
        await b.stop()

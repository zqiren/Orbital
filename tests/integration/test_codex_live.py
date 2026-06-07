# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Codex live journeys (TEST RULE 2 + 5): real codex 0.125.0 app-server.

Covers: dispatch -> command + fileChange items -> final answer -> ONE honest
turn boundary; on-request approval round-trip (decline continues / cancel
interrupts); kill -9 -> thread/resume -> context recalled; Stop leaves zero
in-tree survivors.

Heavy + costs live turns — opt-in: not in the default unit/platform run.
Pinned to codex-cli 0.125.0; on a version bump re-run schema-gen + this file.
"""

from __future__ import annotations

import asyncio
import os
import shutil

import psutil
import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import (
    CodexTransport, SUPPORTED_CODEX_VERSION)

pytestmark = pytest.mark.live_daemon  # heavy: spawns real codex — opt-in

CODEX = shutil.which("codex")
requires_codex = pytest.mark.skipif(
    CODEX is None, reason="codex binary not on PATH")


async def _events_until_turn_complete(transport, timeout=240.0):
    events = []
    async def consume():
        async for event in transport.read_stream():
            events.append(event)
            if event.event_type == "turn_complete":
                return
    await asyncio.wait_for(consume(), timeout=timeout)
    return events


async def _cheapest_model(transport) -> str | None:
    result = await transport._request("model/list", {"includeHidden": False},
                                      timeout=30.0)
    for m in result.get("data", []):
        if "mini" in (m.get("id") or ""):
            return m["id"]
    return None


async def _start(tmp_path, autonomy=Autonomy.HANDS_OFF, resume_record=None,
                 model: str | None = None):
    # Fresh starts self-resolve a startup model from model/list
    # (TASK-codex-startup-model) — no probe dance needed. Pass `model` to
    # pin one explicitly; resume omits unless pinned (amendment semantics).
    t = CodexTransport(autonomy=autonomy, resume_record=resume_record)
    await t.start(CODEX, ["-m", model] if model else [], str(tmp_path))
    return t


@requires_codex
@pytest.mark.asyncio
async def test_round_trip_command_filechange_final_answer(tmp_path):
    # Fresh start self-resolves a startup model from model/list
    # (TASK-codex-startup-model) — no probe dance needed.
    t = await _start(tmp_path)
    try:
        await t.dispatch(
            "Do exactly these two things using your tools, then stop. "
            "1) Run this shell command: python3 -c 'print(6*7)' "
            "2) Create a file named hello.txt in the current directory "
            "containing exactly: hello from orbital codex")
        events = await _events_until_turn_complete(t)
        kinds = [e.event_type for e in events]
        assert kinds.count("turn_complete") == 1, "exactly one turn boundary"
        assert kinds[-1] == "turn_complete"
        tc = events[-1]
        assert tc.data["cause"] == "success"
        assert tc.data["session_id"] == t._thread_id
        assert tc.data["rollout_path"] and os.path.isfile(tc.data["rollout_path"])
        tools = [e for e in events if e.event_type == "tool_use"]
        assert any(e.data["tool_name"] == "commandExecution" for e in tools)
        assert any(e.data["tool_name"] == "fileChange" for e in tools)
        assert any(e.event_type == "message" for e in events)
        assert (tmp_path / "hello.txt").read_text().startswith("hello from orbital codex")
    finally:
        await t.stop()


_APPROVAL_PROMPT = (
    "Create an empty file named marker.txt by running exactly: "
    "touch marker.txt . Your sandbox may deny it - if so, "
    "immediately request escalated permissions (approval). "
    "If the approval is denied, say DENIED-OK and stop.")


async def _run_approval_turn(tmp_path, decision):
    """One read-only on-request approval turn on a FRESH thread, answered
    with `decision`. Returns the turn_complete event.

    Each decision gets its own transport/thread deliberately: replaying the
    same prompt on a shared thread poisons the second turn — the agent
    *remembers* the prior denial from conversation history and short-circuits
    to the canned 'DENIED-OK' reply WITHOUT re-attempting the command, so no
    approval ever fires and the cancel decision is never exercised (observed
    live). A clean thread per decision is the only way to deterministically
    force the escalation each time.
    """
    # CHECK_IN maps to on-request; we override the thread sandbox to
    # read-only so `touch` is denied and an escalation approval fires.
    tmp_path.mkdir(parents=True, exist_ok=True)
    t = CodexTransport(autonomy=Autonomy.CHECK_IN)
    await t.start(CODEX, [], str(tmp_path))
    try:
        # Use the cheapest MINI model — NOT `_effective_model`: the
        # server-default `gpt-5.x-codex` is rejected with HTTP 400 ("not
        # supported when using Codex with a ChatGPT account"), which would
        # error the turn before any approval ever fires.
        model = await _cheapest_model(t)
        params = {"cwd": str(t._workspace), "approvalPolicy": "on-request",
                  "sandbox": "read-only"}
        if model:
            params["model"] = model
        result = await t._request("thread/start", params)
        t._thread_id = result["thread"]["id"]
        t._rollout_path = result["thread"].get("path")

        await t.dispatch(_APPROVAL_PROMPT)
        saw_request = False

        async def run():
            nonlocal saw_request
            async for event in t.read_stream():
                if event.event_type == "permission_request":
                    saw_request = True
                    await t.respond_to_permission_decision(
                        event.data["request_id"], decision)
                if event.event_type == "turn_complete":
                    return event
        tc = await asyncio.wait_for(run(), timeout=240.0)
        assert saw_request, (
            f"the read-only sandbox must force an escalation approval for "
            f"the {decision} journey — none fired")
        return tc
    finally:
        await t.stop()


@requires_codex
@pytest.mark.asyncio
async def test_on_request_approval_decline_continues_cancel_interrupts(tmp_path):
    # read-only sandbox + on-request forces an escalation approval
    # (probe step4/4b shape). Two fresh threads so the journeys don't poison
    # each other through shared conversation memory (see _run_approval_turn).

    # --- decline: the turn must CONTINUE to a final answer
    tc = await _run_approval_turn(tmp_path / "decline", "decline")
    assert tc.data["cause"] == "success", \
        "decline must let the turn continue to completion (FINDINGS A4d)"

    # --- cancel: the turn must END interrupted, no final answer
    tc = await _run_approval_turn(tmp_path / "cancel", "cancel")
    assert tc.data["cause"] == "interrupted", \
        ("cancel ends the turn `interrupted` while the agent lives on "
         "(FINDINGS A4d) — routed to the wake, NOT to silent `stopped`")


@requires_codex
@pytest.mark.asyncio
async def test_kill9_then_resume_recalls_context(tmp_path):
    t = await _start(tmp_path)
    thread_id = model = rollout = None
    try:
        await t.dispatch("Remember this codeword: PELICAN-77. Reply only "
                         "with a confirmation that you stored it. Do not "
                         "use any tools.")
        events = await _events_until_turn_complete(t)
        tc = events[-1]
        thread_id, model, rollout = (tc.data["session_id"], tc.data["model"],
                                     tc.data["rollout_path"])
        # kill -9 the whole tree (simulated daemon death — no clean stop)
        root = psutil.Process(t._popen.pid)
        for child in root.children(recursive=True):
            child.kill()
        root.kill()
        await asyncio.sleep(0.5)
    finally:
        await t.stop()  # idempotent cleanup

    record = {"session_id": thread_id, "model": model, "rollout_path": rollout}
    assert CodexTransport.resume_source_exists(record), "rollout must exist"
    t2 = CodexTransport(autonomy=Autonomy.HANDS_OFF, resume_record=record)
    await t2.start(CODEX, [], str(tmp_path))
    try:
        assert t2._thread_id == thread_id
        await t2.dispatch("What is the codeword? Reply with the codeword "
                          "only. Do not use any tools.")
        events = await _events_until_turn_complete(t2)
        text = " ".join(e.raw_text for e in events if e.event_type == "message")
        assert "PELICAN-77" in text, "resume must be conversation-lossless"
    finally:
        await t2.stop()


@requires_codex
@pytest.mark.asyncio
async def test_stop_mid_exec_leaves_zero_survivors(tmp_path):
    # THE teardown failure mode (FINDINGS A5b): an in-flight exec survives
    # SIGTERM-of-root, reparented to pid 1. The tree-walk must reap it.
    t = await _start(tmp_path)
    try:
        await t.dispatch("Run this shell command and wait for it to finish: "
                         "sleep 60")
        # wait for the exec to actually start (commandExecution item/started)
        async def until_exec():
            async for event in t.read_stream():
                if (event.event_type == "tool_use"
                        and event.data["tool_name"] == "commandExecution"
                        and "sleep 60" in (event.data["tool_input"]["command"] or "")):
                    return
        await asyncio.wait_for(until_exec(), timeout=120.0)
        victims = [t._proc] + t._proc.children(recursive=True)
        assert len(victims) >= 2, "expected the in-flight exec in the tree"
        await t.stop()
        await asyncio.sleep(1.0)
        survivors = [p for p in victims if p.is_running()
                     and p.status() != psutil.STATUS_ZOMBIE]
        assert survivors == [], \
            f"tree-walk teardown left survivors: {[p.pid for p in survivors]}"
    finally:
        await t.stop()


@requires_codex
@pytest.mark.asyncio
async def test_version_is_pinned(tmp_path):
    import subprocess
    out = subprocess.run([CODEX, "--version"], capture_output=True,
                         text=True).stdout.strip()
    assert SUPPORTED_CODEX_VERSION in out, (
        f"codex on PATH is {out!r}, suite is pinned to "
        f"{SUPPORTED_CODEX_VERSION} — re-run "
        "`codex app-server generate-json-schema` + this suite, update the "
        "pin, and re-verify the FINDINGS constraints before shipping")


@requires_codex
@pytest.mark.asyncio
async def test_rejected_model_heals_on_resume_with_override(tmp_path):
    """THE trap (model-is-config amendment, DONE-WHEN): a thread whose first
    turn 400'd on a ChatGPT-rejected model must heal on resume once a working
    override exists — the SAME thread, not a fresh one.

    The resolver (TASK-codex-startup-model) prevents accidental bad starts,
    so the trap is reproduced deliberately via an explicit bad override
    ("-m gpt-5.3-codex"). The heal invariant (override beats record on the
    same thread) is unchanged."""
    # 1. Fresh thread, EXPLICIT bad model -> 400.
    t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
    await t.start(CODEX, ["-m", "gpt-5.3-codex"], str(tmp_path))
    try:
        await t.dispatch("Say OK. Do not use any tools.")
        events = await _events_until_turn_complete(t)
        tc = events[-1]
        assert tc.data["cause"] == "error", \
            "expected the ChatGPT-account model rejection on gpt-5.3-codex"
        record = {"session_id": tc.data["session_id"],
                  "model": tc.data["model"],          # the rejected model
                  "rollout_path": tc.data["rollout_path"]}
        thread_id = tc.data["session_id"]
        assert thread_id and record["rollout_path"]
    finally:
        await t.stop()

    # 2. The user sets a working override (config-store -> "-m" argv) and the
    #    EXISTING thread resumes on it — record's rejected model ignored.
    probe = await _start(tmp_path)
    try:
        mini = await _cheapest_model(probe)
    finally:
        await probe.stop()
    assert mini, "account must expose a mini model for the heal leg"
    t2 = CodexTransport(autonomy=Autonomy.HANDS_OFF, resume_record=record)
    await t2.start(CODEX, ["-m", mini], str(tmp_path))
    try:
        assert t2._thread_id == thread_id, "must heal the SAME thread"
        await t2.dispatch("Say OK. Do not use any tools.")
        events = await _events_until_turn_complete(t2)
        tc = events[-1]
        assert tc.data["cause"] == "success", \
            "override must heal the previously rejected thread"
        assert tc.data["model"] == mini
    finally:
        await t2.stop()


@requires_codex
@pytest.mark.asyncio
async def test_cold_start_resolves_model_no_400(tmp_path):
    """DONE-WHEN (TASK-codex-startup-model): fresh Codex, NO override, on
    ChatGPT-account auth — Orbital resolves from model/list and the first
    turn completes. The path the probe never ran; proves the resolution
    PREVENTS the failure rather than assuming it."""
    t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
    await t.start(CODEX, [], str(tmp_path))  # no override anywhere
    try:
        assert t._model in ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5"), \
            f"resolved {t._model!r} — expected a preference-order pick"
        assert t._effective_model == t._model, \
            "server must confirm the resolved model"
        await t.dispatch("Say OK. Do not use any tools.")
        events = await _events_until_turn_complete(t)
        tc = events[-1]
        assert tc.data["cause"] == "success", \
            "cold start must complete — the 400 class is dead"
        assert not any(e.event_type == "error" for e in events)
    finally:
        await t.stop()

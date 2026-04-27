"""Live-daemon smoke test for state-refresh-triggers.

Runs a real interactive session against a live AgentOS daemon using DeepSeek v4,
asking the agent to gather Orbital competitor information. Captures the
`state_refresh.lifecycle` WS events fired by the new turn-count trigger,
verifies PROJECT_STATE mtime changes, and saves the competitor research
output to the user's Desktop as a markdown file.

Usage:
    python scripts/smoke_state_refresh.py

Env required:
    DEEPSEEK_API_KEY (the test will set it from the hardcoded value if unset)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import websockets

DAEMON = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
TARGET_TURNS = 17  # COOLDOWN_TURNS=15; need at least 16 for first fire
DESKTOP = Path.home() / "Desktop"

PROMPTS = [
    "Hi! I'm researching the AI agent orchestration market. Orbital (also called AgentOS) is a desktop app for running and orchestrating AI coding agents. What are the most direct competitors? List 3 with one-line summaries.",
    "Tell me more about Cursor. How does its agent mode compare to a desktop orchestration platform like Orbital?",
    "What about Cline (formerly Claude Dev)? Where does it sit in the stack relative to Orbital?",
    "How does Aider compare? It runs in the terminal, right?",
    "What's the deal with Claude Code (Anthropic's CLI)? Is it a competitor or complementary to a multi-agent orchestrator?",
    "What about Roo Code (formerly Roo Cline)? How does it differ from Cline?",
    "Continue.dev — what's their angle? Where do they compete with Orbital?",
    "What about Devin from Cognition? Different category or competitor?",
    "OpenDevin / OpenHands — open source competitor?",
    "Smol Developer / smolagents — relevant or too small?",
    "Where do agentic IDEs like Windsurf fit in?",
    "What about MCP-based orchestrators? Any worth naming?",
    "Pulumi AI / Agentic infrastructure tools — relevant tangent or out of scope?",
    "Auto-GPT and BabyAGI — historical or still relevant in 2026?",
    "GitHub Copilot Workspace — Microsoft's answer to all this. Where does it fit?",
    "Summarize the competitive landscape so far in a paragraph for our research notes.",
    "Now write a short table comparing Orbital, Cursor, Cline, and Claude Code on these dimensions: deployment model, agent autonomy, multi-agent support, OSS-vs-proprietary.",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def wait_until_idle(client: httpx.AsyncClient, project_id: str, timeout: float = 180.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        r = await client.get(f"{DAEMON}/api/v2/agents/{project_id}/run-status")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "idle":
                return
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Agent did not become idle within {timeout}s")


async def ws_listener(project_id: str, events: list, stop_event: asyncio.Event) -> None:
    """Subscribe and capture state_refresh.lifecycle + agent.status events."""
    try:
        async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "subscribe", "project_ids": [project_id]}))
            ack = await ws.recv()
            log(f"WS subscribed: {ack}")
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    t = data.get("type", "")
                    if t == "state_refresh.lifecycle":
                        log(f"  >> STATE_REFRESH: {data}")
                        events.append(data)
                    elif t == "agent.status":
                        if data.get("status") in ("idle", "running", "new_session"):
                            pass  # quiet
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    break
    except Exception as e:
        log(f"WS listener error: {e}")


async def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="orbital-smoke-"))
    log(f"Workspace: {workspace}")
    project_id = f"smoke-{uuid.uuid4().hex[:8]}"
    log(f"Project ID: {project_id}")

    state_file = workspace / "PROJECT_STATE.md"

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        # 1. Create project pointing at temp workspace, with DeepSeek config
        create_payload = {
            "name": f"smoke-{project_id}",
            "workspace": str(workspace),
            "model": "deepseek-v4-flash",
            "api_key": DEEPSEEK_KEY,
            "base_url": "https://api.deepseek.com/v1",
            "provider": "deepseek",
            "sdk": "openai",
            "autonomy": "hands_off",
            "is_scratch": True,
        }
        r = await client.post(f"{DAEMON}/api/v2/projects", json=create_payload)
        if r.status_code not in (200, 201):
            log(f"Create project failed: {r.status_code} {r.text}")
            return 1
        proj = r.json()
        project_id = proj.get("id") or proj.get("project_id") or project_id
        log(f"Project created: {project_id}")

        # 2. Subscribe to WS in background
        events: list = []
        stop_event = asyncio.Event()
        ws_task = asyncio.create_task(ws_listener(project_id, events, stop_event))
        await asyncio.sleep(1.0)

        # 3. Start agent
        start_payload = {"project_id": project_id, "initial_message": None}
        r = await client.post(f"{DAEMON}/api/v2/agents/start", json=start_payload)
        if r.status_code != 200:
            log(f"Start agent failed: {r.status_code} {r.text}")
            stop_event.set()
            return 1
        log(f"Agent started: {r.json()}")
        await asyncio.sleep(1.5)

        # 4. Snapshot mtime before any turns (file may not exist yet)
        mtime_before = state_file.stat().st_mtime if state_file.exists() else None
        log(f"PROJECT_STATE mtime before: {mtime_before}")

        # 5. Inject prompts one by one, waiting for idle between each
        successes = 0
        for idx, prompt in enumerate(PROMPTS[:TARGET_TURNS], start=1):
            log(f"--- Turn {idx}/{TARGET_TURNS} --- {prompt[:60]}...")
            try:
                await wait_until_idle(client, project_id, timeout=120)
            except RuntimeError as e:
                log(f"  pre-inject idle wait failed: {e}; continuing anyway")
            inject_payload = {"content": prompt, "nonce": uuid.uuid4().hex}
            r = await client.post(f"{DAEMON}/api/v2/agents/{project_id}/inject", json=inject_payload)
            if r.status_code not in (200, 201):
                log(f"  inject failed: {r.status_code} {r.text[:200]}")
                continue
            successes += 1
            try:
                await wait_until_idle(client, project_id, timeout=120)
            except RuntimeError as e:
                log(f"  post-inject idle wait failed: {e}; continuing")

        log(f"Injected {successes}/{TARGET_TURNS} turns successfully")

        # Give time for any tail-end refresh to complete
        await asyncio.sleep(3.0)

        # 6. Verify PROJECT_STATE mtime changed
        mtime_after = state_file.stat().st_mtime if state_file.exists() else None
        log(f"PROJECT_STATE mtime after:  {mtime_after}")
        mtime_changed = (
            mtime_before is None and mtime_after is not None
        ) or (
            mtime_before is not None
            and mtime_after is not None
            and mtime_after > mtime_before
        )

        # 7. Stop the WS listener and the agent
        stop_event.set()
        try:
            await asyncio.wait_for(ws_task, timeout=5.0)
        except asyncio.TimeoutError:
            ws_task.cancel()

        # Pull the chat transcript so we can save competitor research output
        chat_text = ""
        try:
            r = await client.get(
                f"{DAEMON}/api/v2/agents/{project_id}/chat",
                params={"limit": 500, "offset": 0},
            )
            if r.status_code == 200:
                data = r.json()
                msgs = data if isinstance(data, list) else data.get("messages", [])
                lines = []
                for m in msgs:
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            str(c.get("text") or c) if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    lines.append(f"\n## {role.upper()}\n\n{content}\n")
                chat_text = "".join(lines)
        except Exception as e:
            log(f"chat fetch failed: {e}")

        # 8. Stop agent
        try:
            await client.post(f"{DAEMON}/api/v2/agents/{project_id}/stop")
        except Exception:
            pass

    # 9. Save competitor research to Desktop
    DESKTOP.mkdir(parents=True, exist_ok=True)
    out_path = DESKTOP / "orbital-competitor-research.md"
    state_content = state_file.read_text(encoding="utf-8") if state_file.exists() else "(PROJECT_STATE.md not written)"
    refresh_events_md = "\n".join(
        f"- {e.get('timestamp', '?')}: trigger=`{e.get('trigger')}` status=`{e.get('status')}`"
        for e in events
    ) or "(none)"

    out_path.write_text(
        f"# Orbital Competitor Research — Smoke Test Output\n\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Project ID: `{project_id}`\n"
        f"Model: deepseek-v4-flash\n\n"
        f"## State-refresh lifecycle events captured\n\n"
        f"{refresh_events_md}\n\n"
        f"## PROJECT_STATE.md (final)\n\n"
        f"{state_content}\n\n"
        f"## Full conversation transcript\n"
        f"{chat_text}\n",
        encoding="utf-8",
    )

    # 10. Summary
    fire_count = sum(
        1 for e in events
        if e.get("status") == "done" and e.get("trigger") == "turn_count"
    )
    log("=" * 60)
    log("SMOKE TEST SUMMARY")
    log("=" * 60)
    log(f"Turns injected:                {successes}")
    log(f"state_refresh.lifecycle events: {len(events)}")
    log(f"  done/turn_count:             {fire_count}")
    log(f"PROJECT_STATE mtime changed:   {mtime_changed}")
    log(f"Output written to:             {out_path}")
    log("=" * 60)

    if successes < 16:
        log("FAIL: did not inject enough turns to test the trigger")
        return 2
    if fire_count < 1:
        log("FAIL: no turn_count refresh fired during 17 turns")
        return 3
    if not mtime_changed:
        log("FAIL: PROJECT_STATE mtime did not change")
        return 4
    log("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

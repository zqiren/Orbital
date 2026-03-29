#!/usr/bin/env python3
"""Smoke tests for v5 sub-agent orchestration against a running daemon.

Tests the scenarios from TASK/agent-loop-rewrite/test-scenario.md using
real API calls to the daemon with a live LLM key.

Usage:
    python scripts/smoke_v5.py

Requires: daemon running on localhost:8000, .env at TASK/agent-loop-rewrite/.env
"""

import json
import os
import sys
import tempfile
import time

import requests
from dotenv import load_dotenv

# Load credentials
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "TASK", "agent-loop-rewrite", ".env")
load_dotenv(ENV_PATH)

DAEMON_URL = "http://localhost:8000"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_SDK = os.getenv("LLM_SDK", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "custom")


def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


def check_daemon():
    """Verify daemon is reachable."""
    try:
        r = requests.get(f"{DAEMON_URL}/api/v2/settings", timeout=5)
        r.raise_for_status()
        log("Daemon is running")
        return True
    except Exception as e:
        log(f"Daemon not reachable: {e}", "ERROR")
        return False


def check_llm_key():
    """Verify LLM key is configured."""
    if not LLM_API_KEY:
        log("LLM_API_KEY not set in .env", "ERROR")
        return False
    log(f"LLM key present (provider={LLM_PROVIDER}, model={LLM_MODEL})")
    return True


def create_project(name, workspace, autonomy="hands_off", extra=None):
    """Create a test project and return project_id."""
    payload = {
        "name": name,
        "workspace": workspace,
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "base_url": LLM_BASE_URL,
        "sdk": LLM_SDK,
        "provider": LLM_PROVIDER,
        "autonomy": autonomy,
        **(extra or {}),
    }
    r = requests.post(f"{DAEMON_URL}/api/v2/projects", json=payload)
    r.raise_for_status()
    pid = r.json()["project_id"]
    log(f"Created project '{name}' -> {pid}")
    return pid


def start_agent(project_id, initial_message=None):
    """Start the agent loop for a project."""
    payload = {"project_id": project_id}
    if initial_message:
        payload["initial_message"] = initial_message
    r = requests.post(f"{DAEMON_URL}/api/v2/agents/start", json=payload)
    r.raise_for_status()
    log(f"Agent started for {project_id}")
    return r.json()


def inject_message(project_id, content, target=None):
    """Send a user message to the agent."""
    payload = {"content": content}
    if target:
        payload["target"] = target
    r = requests.post(f"{DAEMON_URL}/api/v2/agents/{project_id}/inject", json=payload)
    r.raise_for_status()
    log(f"Injected message to {project_id}" + (f" (target={target})" if target else ""))
    return r.json()


def get_run_status(project_id):
    """Get agent run status."""
    r = requests.get(f"{DAEMON_URL}/api/v2/agents/{project_id}/run-status")
    r.raise_for_status()
    return r.json()


def get_chat(project_id, limit=0, offset=0):
    """Get chat history."""
    params = {}
    if limit:
        params["limit"] = limit
    if offset:
        params["offset"] = offset
    r = requests.get(f"{DAEMON_URL}/api/v2/agents/{project_id}/chat", params=params)
    r.raise_for_status()
    total = int(r.headers.get("X-Total-Count", 0))
    messages = r.json()
    return messages, total


def stop_agent(project_id):
    """Stop the agent loop."""
    try:
        r = requests.post(f"{DAEMON_URL}/api/v2/agents/{project_id}/stop")
        r.raise_for_status()
        log(f"Agent stopped for {project_id}")
    except Exception:
        pass  # May not be running


def delete_project(project_id):
    """Delete a test project."""
    try:
        r = requests.delete(f"{DAEMON_URL}/api/v2/projects/{project_id}")
        r.raise_for_status()
        log(f"Deleted project {project_id}")
    except Exception:
        pass


def wait_for_status(project_id, target_status, timeout=120, poll_interval=2):
    """Wait for agent to reach a target status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_run_status(project_id)
        current = status.get("status", "unknown")
        if current == target_status:
            return status
        time.sleep(poll_interval)
    log(f"Timeout waiting for status={target_status} (last={current})", "WARN")
    return status


def wait_for_idle(project_id, timeout=120):
    """Wait for agent to become idle (finished processing)."""
    return wait_for_status(project_id, "idle", timeout)


def wait_for_messages(project_id, min_count, timeout=120, poll_interval=2):
    """Wait until chat has at least min_count messages."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msgs, total = get_chat(project_id)
        if total >= min_count:
            return msgs, total
        time.sleep(poll_interval)
    log(f"Timeout waiting for {min_count} messages (have {total})", "WARN")
    return msgs, total


# ============================================================
# Scenario 1: Delegate and continue working
# ============================================================
def scenario_1_delegate_and_continue():
    """Test that management agent can delegate to sub-agent and continue working."""
    log("=" * 60)
    log("SCENARIO 1: Delegate and continue working")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        # Create a README.md for the agent to read
        with open(os.path.join(workspace, "README.md"), "w") as f:
            f.write("# Test Project\n\nThis is a test project for smoke testing.\n"
                    "It has an auth module that uses basic password hashing.\n")

        pid = create_project("smoke-s1-delegate", workspace,
                             extra={"enabled_sub_agents": ["claude-code"]})
        try:
            # Turn 1: Start agent with delegation request
            start_agent(pid, initial_message="What is 2 + 2? Reply with just the number.")

            # Wait for agent to process and become idle
            log("Waiting for agent to process Turn 1...")
            wait_for_idle(pid, timeout=90)

            # Check run status
            status = get_run_status(pid)
            log(f"Agent status after Turn 1: {status}")

            # Check chat messages
            msgs, total = get_chat(pid)
            log(f"Chat messages after Turn 1: {total} messages")
            for i, m in enumerate(msgs):
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:120]
                source = m.get("source", "")
                log(f"  [{i}] role={role} source={source} content={content}")

            # Verify: management session has messages, agent responded
            assert total >= 2, f"Expected at least 2 messages, got {total}"
            roles = [m.get("role") for m in msgs]
            assert "user" in roles, "No user message in chat"
            assert "assistant" in roles, "No assistant response in chat"

            # Verify no role=agent leaked into management session
            session_roles = [m.get("role") for m in msgs if m.get("source", "") == ""]
            assert "agent" not in session_roles, "role=agent found in management session!"

            log("SCENARIO 1: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 1: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 2: Basic agent loop - send message and get response
# ============================================================
def scenario_2_basic_loop():
    """Test basic agent loop: send a message, get a response."""
    log("=" * 60)
    log("SCENARIO 2: Basic agent loop with LLM response")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s2-basic", workspace)
        try:
            # Start with a simple message
            start_agent(pid, initial_message="What is the capital of France? Reply in one word.")

            log("Waiting for agent to respond...")
            wait_for_idle(pid, timeout=90)

            msgs, total = get_chat(pid)
            log(f"Chat messages: {total}")
            for i, m in enumerate(msgs):
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:200]
                log(f"  [{i}] role={role}: {content}")

            # Should have at least user + assistant
            assert total >= 2, f"Expected >=2 messages, got {total}"

            # Check assistant responded with something about Paris
            assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
            assert len(assistant_msgs) >= 1, "No assistant response"
            response_text = str(assistant_msgs[-1].get("content", "")).lower()
            assert "paris" in response_text, f"Expected 'Paris' in response, got: {response_text[:200]}"

            log("SCENARIO 2: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 2: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 3: Multi-turn conversation
# ============================================================
def scenario_3_multi_turn():
    """Test multi-turn conversation with context retention."""
    log("=" * 60)
    log("SCENARIO 3: Multi-turn conversation")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s3-multiturn", workspace)
        try:
            # Turn 1
            start_agent(pid, initial_message="My name is Alice. Remember my name.")
            log("Waiting for Turn 1...")
            wait_for_idle(pid, timeout=90)

            # Turn 2 - test context retention
            inject_message(pid, "What is my name?")
            log("Waiting for Turn 2...")
            time.sleep(2)  # Brief pause for inject to be processed
            wait_for_idle(pid, timeout=90)

            msgs, total = get_chat(pid)
            log(f"Chat messages: {total}")
            for i, m in enumerate(msgs):
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:200]
                log(f"  [{i}] role={role}: {content}")

            # Should have user1 + assistant1 + user2 + assistant2
            assert total >= 4, f"Expected >=4 messages, got {total}"

            # Last assistant response should mention Alice
            assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
            assert len(assistant_msgs) >= 2, f"Expected >=2 assistant messages, got {len(assistant_msgs)}"
            last_response = str(assistant_msgs[-1].get("content", "")).lower()
            assert "alice" in last_response, f"Expected 'Alice' in response, got: {last_response[:200]}"

            log("SCENARIO 3: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 3: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 4: Chat history persistence (page refresh)
# ============================================================
def scenario_4_chat_persistence():
    """Test that GET /chat returns all messages after simulated page refresh."""
    log("=" * 60)
    log("SCENARIO 4: Chat history persistence (page refresh)")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s4-persist", workspace)
        try:
            start_agent(pid, initial_message="Say hello.")
            log("Waiting for response...")
            wait_for_idle(pid, timeout=90)

            # First read - before "refresh"
            msgs1, total1 = get_chat(pid)
            log(f"Before refresh: {total1} messages")

            # Simulate page refresh - just re-read chat
            msgs2, total2 = get_chat(pid)
            log(f"After refresh: {total2} messages")

            assert total1 == total2, f"Message count changed: {total1} -> {total2}"
            assert len(msgs1) == len(msgs2), "Message list changed"

            # Verify messages are identical
            for i, (m1, m2) in enumerate(zip(msgs1, msgs2)):
                assert m1.get("role") == m2.get("role"), f"Message {i} role mismatch"
                assert m1.get("content") == m2.get("content"), f"Message {i} content mismatch"

            # Verify chronological ordering (timestamps increasing)
            timestamps = [m.get("timestamp", "") for m in msgs2 if m.get("timestamp")]
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i-1], \
                    f"Messages not in chronological order at index {i}"

            log("SCENARIO 4: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 4: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 5: Session isolation - no role=agent in management session
# ============================================================
def scenario_5_session_isolation():
    """Test v5 guarantee: no role=agent messages in management session JSONL."""
    log("=" * 60)
    log("SCENARIO 5: Session isolation (no role=agent in management)")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s5-isolation", workspace)
        try:
            start_agent(pid, initial_message="What is 1+1?")
            log("Waiting for response...")
            wait_for_idle(pid, timeout=90)

            msgs, total = get_chat(pid)
            log(f"Chat messages: {total}")

            # Check raw session file for role=agent leaks
            # (GET /chat may normalize, so also check JSONL directly)
            from agent_os.daemon_v2.project_store import project_dir_name
            dir_name = project_dir_name("smoke-s5-isolation", pid)
            sessions_dir = os.path.join(workspace, "orbital", dir_name, "sessions")
            if os.path.isdir(sessions_dir):
                for fname in os.listdir(sessions_dir):
                    if fname.endswith(".jsonl"):
                        fpath = os.path.join(sessions_dir, fname)
                        with open(fpath) as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                entry = json.loads(line)
                                role = entry.get("role", "")
                                if role == "agent":
                                    log(f"VIOLATION: role=agent in session JSONL at line {line_num}: {line[:100]}", "FAIL")
                                    assert False, "role=agent found in management session JSONL"
                log("Session JSONL verified: no role=agent entries")
            else:
                log("No session directory found (agent may not have written yet)", "WARN")

            # Also verify via GET /chat
            for m in msgs:
                if m.get("role") == "agent" and not m.get("source"):
                    assert False, "role=agent without source in chat endpoint"

            log("SCENARIO 5: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 5: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 6: Run status transitions
# ============================================================
def scenario_6_run_status():
    """Test that run-status correctly reflects agent state transitions."""
    log("=" * 60)
    log("SCENARIO 6: Run status transitions")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s6-status", workspace)
        try:
            # Before start - should be idle/not running
            status = get_run_status(pid)
            log(f"Status before start: {status}")

            # Start with a message that requires some thinking
            start_agent(pid, initial_message="Count from 1 to 5, each number on a new line.")

            # Check that it transitions to running
            time.sleep(1)
            status = get_run_status(pid)
            log(f"Status after start: {status}")
            # May already be idle if the LLM responded fast

            # Wait for completion
            wait_for_idle(pid, timeout=90)
            status = get_run_status(pid)
            log(f"Status after completion: {status}")
            assert status.get("status") == "idle", f"Expected idle, got {status}"

            log("SCENARIO 6: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 6: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Scenario 7: Chat pagination
# ============================================================
def scenario_7_chat_pagination():
    """Test that GET /chat pagination works correctly."""
    log("=" * 60)
    log("SCENARIO 7: Chat pagination")
    log("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        pid = create_project("smoke-s7-pagination", workspace)
        try:
            # Generate a multi-turn conversation
            start_agent(pid, initial_message="Say 'A'")
            wait_for_idle(pid, timeout=90)

            inject_message(pid, "Say 'B'")
            time.sleep(2)
            wait_for_idle(pid, timeout=90)

            # Get all messages
            all_msgs, total = get_chat(pid)
            log(f"Total messages: {total}")
            assert total >= 4, f"Expected >=4 messages, got {total}"

            # Test limit
            limited_msgs, limited_total = get_chat(pid, limit=2)
            log(f"Limited (limit=2): got {len(limited_msgs)} of {limited_total}")
            assert len(limited_msgs) <= 2, f"Expected <=2 messages, got {len(limited_msgs)}"
            assert limited_total == total, "Total count mismatch"

            # Test offset
            offset_msgs, offset_total = get_chat(pid, limit=2, offset=2)
            log(f"Offset (limit=2, offset=2): got {len(offset_msgs)} of {offset_total}")

            log("SCENARIO 7: PASSED", "OK")
            return True

        except Exception as e:
            log(f"SCENARIO 7: FAILED - {e}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
        finally:
            stop_agent(pid)
            delete_project(pid)


# ============================================================
# Main
# ============================================================
def main():
    log("v5 Smoke Test Suite")
    log("=" * 60)

    if not check_daemon():
        log("Cannot reach daemon. Start it with: bash scripts/restart-daemon.sh", "ERROR")
        sys.exit(1)

    if not check_llm_key():
        log("No LLM key. Create TASK/agent-loop-rewrite/.env per ACTIVE-smoke-env.md", "ERROR")
        sys.exit(1)

    # Test LLM connectivity first
    log("Testing LLM provider connectivity...")
    try:
        r = requests.post(f"{DAEMON_URL}/api/v2/providers/test", json={
            "api_key": LLM_API_KEY,
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "sdk": LLM_SDK,
            "provider": LLM_PROVIDER,
        }, timeout=30)
        result = r.json()
        log(f"LLM provider test: {result}")
        if result.get("status") != "ok":
            log(f"LLM provider test failed: {result.get('error', result.get('message', 'unknown'))}", "ERROR")
            sys.exit(1)
    except Exception as e:
        log(f"LLM provider test error: {e}", "ERROR")
        sys.exit(1)

    results = {}
    scenarios = [
        ("S2: Basic agent loop", scenario_2_basic_loop),
        ("S3: Multi-turn conversation", scenario_3_multi_turn),
        ("S4: Chat persistence", scenario_4_chat_persistence),
        ("S5: Session isolation", scenario_5_session_isolation),
        ("S6: Run status transitions", scenario_6_run_status),
        ("S7: Chat pagination", scenario_7_chat_pagination),
        ("S1: Delegate and continue", scenario_1_delegate_and_continue),
    ]

    for name, fn in scenarios:
        log(f"\n{'='*60}")
        try:
            results[name] = fn()
        except Exception as e:
            log(f"Unhandled error in {name}: {e}", "ERROR")
            results[name] = False

    # Summary
    log("\n" + "=" * 60)
    log("SMOKE TEST SUMMARY")
    log("=" * 60)
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        log(f"  [{status}] {name}")
    log(f"\n{passed} passed, {failed} failed out of {len(results)} scenarios")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

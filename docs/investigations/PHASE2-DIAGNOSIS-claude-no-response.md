# PHASE 2 DIAGNOSIS — Why claude.exe Doesn't Respond to SDK Initialize Under SDKTransport

**Investigation:** `TASK-investigate-claude-exe-no-response.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-agent-sdk 0.1.48, claude-code 2.1.138.
**Branch:** `fix/option-1-system-prompt-append` off `test/full-integration` @ `a55a28c` (Option 1 fix uncommitted on top).
**Status:** Root cause identified empirically. No fix proposal in this doc — surfaced for joint decision.

---

## 1. Phase 1 findings — full argv + stdio + env capture

Extended Tier 3 instrumentation captured every argv element, stdio handles, cwd, creationflags/startupinfo, env keys, and post-spawn returncode of `anyio.open_process`. One failing dispatch was instrumented end-to-end on `fix/option-1-system-prompt-append`.

### Full argv of the SDK session spawn (14 elements)

```
[0]  C:\Users\qiren\AppData\Roaming\npm\claude.CMD
[1]  --output-format
[2]  stream-json
[3]  --verbose
[4]  --append-system-prompt
[5]  "<2080-char orbital inheritance prompt: 'You are a sub-agent dispatched...'>"
[6]  --permission-prompt-tool
[7]  stdio
[8]  --permission-mode
[9]  default
[10] --setting-sources
[11] ''                                  ← empty CSV value
[12] --input-format
[13] stream-json
```

**Option 1's preset/append dict translation worked correctly.** argv[4] is `--append-system-prompt` (APPEND), not `--system-prompt` (REPLACE). The fix landed at the argv layer; the failure is downstream.

### Stdio / cwd / Windows-specific flags

| Kwarg | Value |
|---|---|
| `cwd` | `D:\repro-smoke` (matches project workspace) |
| `stdin` | `-1` = `subprocess.PIPE` |
| `stdout` | `-1` = `subprocess.PIPE` |
| `stderr` | `None` (inherited from daemon parent) |
| `creationflags` | not in kwargs (anyio defaults) |
| `startupinfo` | not in kwargs (anyio defaults) |

### Env (62 keys)

Notable CLAUDE_* keys present in the spawned subprocess env:

- `CLAUDECODE=''` (explicitly blanked by orbital — correct)
- `CLAUDE_CODE_ENTRYPOINT='sdk-py'` (set by claude-agent-sdk)
- `CLAUDE_AGENT_SDK_VERSION` (set by claude-agent-sdk)
- `CLAUDE_CODE_EXECPATH`, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, `CLAUDE_CODE_SESSION_ID` — inherited from the parent Claude Code shell that started the daemon
- `CLAUDE_EFFORT`, `AI_AGENT` — also inherited

### Process state at spawn

`POST pid=19676 returncode=None`. claude.exe spawned successfully in 7ms, stayed alive (returncode unset) for the full 60s SDK timeout, then was killed when the SDK closed its end after timeout.

claude.exe wrote **zero bytes to either stdout or stderr** during the entire 60s — confirmed by inspecting `/tmp/daemon-phase1.log` (62 total lines: 55 `[T3 ...]` instrumentation + 7 uvicorn `INFO:` lines + 0 other).

---

## 2. Phase 2 findings — direct invocation isolation

Two variants run with the exact argv from Phase 1, same cwd, same stdin payload, no Python orbital/SDK involvement — pure `subprocess.Popen`:

The stdin payload is a faithful reconstruction of the SDK's initialize control_request (103 bytes vs the 113 the SDK sends — close enough; sizes differ slightly because the SDK's `request_id` field includes counter+UUID):

```json
{"type":"control_request","request_id":"req_1_phase2","request":{"subtype":"initialize","hooks":null}}
```

### 2a — env replicates orbital's spawn (all CLAUDE_CODE_* inherited)

- `CLAUDE*` keys in env: `['CLAUDECODE', 'CLAUDE_AGENT_SDK_VERSION', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CODE_EXECPATH', 'CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS', 'CLAUDE_CODE_SESSION_ID', 'CLAUDE_EFFORT']`
- Elapsed: **60.015s** (killed at deadline)
- stdout: **0 bytes**
- stderr: **0 bytes**
- Time-to-first-byte: **never**
- Exit: still alive at 60s

### 2b — every CLAUDE_CODE_* / CLAUDE_AGENT_* / CLAUDE_EFFORT / AI_AGENT stripped

- `CLAUDE*` keys in env: `['CLAUDECODE']` (just the orbital-blanked one)
- Elapsed: **60.018s**
- stdout: **0 bytes**
- stderr: **0 bytes**
- Time-to-first-byte: **never**
- Exit: still alive at 60s

### Phase 2 decision-gate verdict

**Both variants hang identically. Inherited CLAUDE_CODE_* env vars are NOT the cause.** Bug is in claude.exe under these args, not in the env, not in orbital's invocation pathway.

This ruled out:
- orbital's code (no orbital involved)
- claude-agent-sdk's Python layer (no SDK involved)
- anyio / asyncio / uvicorn / FastAPI (no async involved)
- inherited Claude Code env (2b stripped them all)

Per the decision gate, Phase 3 (A-F asyncio/uvicorn bisect) was skipped.

---

## 3. Phase 2.5 findings — arg-bisect

To isolate which specific arg(s) in the SDK's 14-arg invocation trigger the hang, I dropped subsets of the argv one at a time. Same env hygiene as Phase 2b (CLAUDE_CODE_* stripped, CLAUDECODE=''). Same stdin payload. 10s deadline per variant. Stop at first responsive.

| Test | Variation | TTFB | stdout | stderr | Verdict |
|---|---|---|---|---|---|
| C1 | drop `--setting-sources ''` (12 args remain) | never | 0 | 0 | hang |
| C2 | drop both `--output-format stream-json` and `--input-format stream-json` (10 args) | never | 0 | 0 | hang |
| C3 | drop `--permission-prompt-tool stdio` and `--permission-mode default` (10 args) | never | 0 | 0 | hang |
| C4 | drop `--verbose` (13 args) | never | 0 | 0 | hang |
| C5 | minimum SDK-style: `claude.CMD --output-format stream-json --verbose` ONLY | never | 0 | 0 | hang |
| C6 (sanity) | `claude.CMD -p ping` (PipeTransport-style single-shot) | stdout TTFB ~kill-time; output `pong\n` emerged from kernel pipe buffer | 5 | 157 (stdin-warn) | **responsive** ✅ |

**Bisect verdict at end of Phase 2.5:** no single arg drop fixes the hang. Even the minimum SDK-style invocation (3 SDK flags) hangs. C6 confirms claude.exe itself is functional in single-shot mode.

This pointed to something common across all SDK-style invocations but absent from C6's `-p` mode — i.e., either `--output-format stream-json` or stdin state. Phase 2.7 disambiguates.

---

## 4. Phase 2.7 findings — stdin state and `--verbose` probes

Three additional probes to differentiate stdin-handling and `--verbose`:

- **C7:** same args as C5 (`--output-format stream-json --verbose`), stdin=PIPE, **but explicitly `stdin.close()` after writing payload** (sends EOF). 15s deadline.
- **C8:** same args, stdin **redirected from a file** via cmd.exe shell redirection (`< init_payload.json`). Different OS handle type (regular file, not anonymous pipe) and natural EOF at end-of-file. 15s deadline.
- **C9:** `--output-format stream-json` ONLY (drop `--verbose`), stdin=PIPE, stdin kept open. 15s deadline.

### Results

| Test | stdin state | --verbose | Outcome | TTFB stdout | stdout | Exit |
|---|---|---|---|---|---|---|
| **C7** | PIPE + closed after write (EOF) | yes | **WORKS** | **4.888s** | 22,368 bytes (NDJSON SessionStart hook events) | rc=0 at 11.222s |
| **C8** | shell file redirect (EOF at end-of-file) | yes | **WORKS** | **4.596s** | 22,518 bytes (same shape) | rc=0 at 10.700s |
| **C9** | PIPE, kept open | no | **HANG** | never | 0 bytes | still alive at 15s |

### Output shape from C7 (head of 22,368 bytes)

```json
{"type":"system","subtype":"hook_started","hook_id":"e6be0af6-b692-42d3-ad4a-190e236dd6b5","hook_name":"SessionStart:startup","hook_event":"SessionStart","uuid":"3e344461-416c-4fd5-8317-05ab6614f4d1","session_id":"430b4208-c465-4ded-bdbb-78880357d3a9"}
{"type":"system","subtype":"hook_response","hook_id":"e6be0af6-b692-42d3-ad4a-190e236dd6b5","hook_name":"SessionStart:startup","hook_event":"SessionStart","output":"{\n  \"hookSpecificOutput\": {\n    \"hookEventName\": \"SessionStart\", ...
```

claude.exe in C7/C8 ran its `SessionStart:startup` hook chain and emitted full NDJSON stream-json events. The `control_response initialize` is among those messages (omitted here for brevity).

C9 is decisive: dropping `--verbose` while keeping stdin open still hangs. **`--verbose` is not part of the trigger.**

---

## Updated classification

**`claude.exe v2.1.138` on Windows, when invoked with `--output-format stream-json`, does not emit any stdout until it observes EOF (or equivalent end-of-input signal) on its stdin.** The claude-agent-sdk transport keeps stdin open after writing each control request (because it intends to write more messages later via `Query.query()`). On Windows, this is a deadlock: the SDK is waiting for a `control_response` that claude.exe will not emit until stdin closes; closing stdin would foreclose the next message; so the system stalls. claude.exe runs hooks and queues responses internally but never flushes them to the open stdout pipe.

The bug is **not** in:
- The choice of `--system-prompt` vs `--append-system-prompt` (W1 classification from `TRACE-windows-dispatch-bug.md` is now formally **superseded**)
- Any of orbital's code paths
- The claude-agent-sdk's argv construction (the dict shape correctly maps to `--append-system-prompt`)
- Python's asyncio / anyio / uvicorn (Phase 2 with pure `subprocess.Popen` reproduces the hang)
- Env-var inheritance (Phase 2b strips every CLAUDE_CODE_* and still hangs)
- Single-arg malformations (Phase 2.5 C5 reproduces with the minimum 3-arg SDK invocation)
- `--verbose` (Phase 2.7 C9 reproduces without it)
- The stdio handle type (Phase 2.7 C8 succeeds with a regular file handle on stdin, but only because end-of-file produces EOF — same root cause)

The bug **is** in:
- claude.exe's stdout-flushing or stdin-read-loop behavior on Windows when `--output-format stream-json` is set and stdin is an open pipe with no further input pending and no EOF observed.

It is platform-specific (the deep-diagnosis run that established this investigation could not reproduce the hang on macOS).

In one sentence: **`claude.exe --output-format stream-json` on Windows blocks stdout output until stdin observes EOF, deadlocking any client (the SDK included) that keeps stdin open expecting bidirectional streaming.**

---

## Reproduction commands

### Reproduce the hang (no orbital, no SDK)

```python
# C5 minimum repro — runs as plain Python, no async, no orbital
import subprocess, json, time
argv = [
    r"C:\Users\qiren\AppData\Roaming\npm\claude.CMD",
    "--output-format", "stream-json",
    "--verbose",
]
init = json.dumps({"type":"control_request","request_id":"x",
                   "request":{"subtype":"initialize","hooks":None}}) + "\n"
env = {k:v for k,v in __import__("os").environ.items()
       if not k.startswith(("CLAUDE_CODE_","CLAUDE_AGENT_"))}
env["CLAUDECODE"] = ""
t0 = time.monotonic()
p = subprocess.Popen(argv, cwd=r"D:\repro-smoke", env=env,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.stdin.write(init.encode()); p.stdin.flush()
# stdin LEFT OPEN
try:
    out, err = p.communicate(timeout=10)
    print(f"finished in {time.monotonic()-t0:.1f}s; stdout={len(out)} stderr={len(err)}")
except subprocess.TimeoutExpired:
    p.kill()
    print(f"HUNG at {time.monotonic()-t0:.1f}s (stdout=0 stderr=0)")
```

Expected: `HUNG at 10.0s`.

### Cure the hang with one-line change

```python
# Replace `p.stdin.write(init.encode()); p.stdin.flush()` with:
p.stdin.write(init.encode()); p.stdin.flush(); p.stdin.close()
```

Expected: `finished in ~11.2s; stdout=~22368 stderr=0` (SessionStart hooks + control_response stream).

### Full Phase 2 / 2.5 / 2.7 test scripts

Standalone, no orbital dependency (except for the prompt renderer, which is replaceable with any 1-2KB string):

- `C:\Users\qiren\AppData\Local\Temp\orbital-phase2\phase2_direct.py` — Phase 2 (2a/2b env variants)
- `C:\Users\qiren\AppData\Local\Temp\orbital-phase2\phase2_5_bisect.py` — Phase 2.5 arg bisect
- `C:\Users\qiren\AppData\Local\Temp\orbital-phase2\phase2_7.py` — Phase 2.7 stdin-state probes

Invoke each as `env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT python <path>`.

---

## Where this leaves the fix decision

(Per investigation spec § DO NOT, no fix proposed in this doc. Possible directions to discuss together — listed for completeness, not as recommendations:)

1. **Upstream-and-wait.** File with anthropic against `claude.exe v2.1.138` describing the Windows stream-json+open-stdin behavior. claude-agent-sdk's protocol design depends on incremental output that doesn't materialize on Windows. Likely the cleanest long-term fix.
2. **Per-message close-and-respawn.** Send each control_request + close stdin → respawn for next message. Effectively converts SDKTransport to PipeTransport semantics (you explicitly rejected this path).
3. **Use a different output format.** If claude.exe doesn't have the buffering issue under some other `--output-format`, switch. Untested in this investigation.
4. **Drop SDKTransport for claude-code on Windows.** Platform-conditional routing. Would mean orbital ships PipeTransport on Windows for claude-code regardless of manifest (also a path you've rejected).

No claim about which of these is right. Surfacing for joint decision.

---

## Cleanup status

- Tier 3 instrumentation reverted from `agent_os/api/app.py`: TBD (about to do)
- `tier3_instrument.py` removed from `D:\orbital-public\.claude\worktrees\test-full-integration\`: TBD (about to do)
- Canonical `tier3_instrument.py` preserved at `/tmp/orbital-tier4/tier3_instrument.py` (used in Phase 1)
- Phase 2 / 2.5 / 2.7 scripts retained at `/tmp/orbital-phase2/` per spec (`Phase 3 test scripts ... under /tmp/orbital-phase2/ cleaned up after investigation` — kept for now in case follow-up needed; will clean after the fix decision)
- `git diff` on `fix/option-1-system-prompt-append` after cleanup: should show ONLY the Option 1 fix (`sdk_transport.py`, the two unit tests in `tests/unit/test_sub_agent_inheritance.py`, and `RUN-MANUAL-SMOKE.md` — no investigation residue in tracked code)
- No code committed during this investigation

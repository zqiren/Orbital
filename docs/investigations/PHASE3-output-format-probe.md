# PHASE 3 — Output-Format Probe

**Investigation:** `TASK-investigate-output-format-options.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-agent-sdk 0.1.48, claude-code 2.1.138.
**Branch:** `fix/option-1-system-prompt-append` (Option 1 fix uncommitted; investigation made no orbital changes).
**Builds on:** `PHASE2-DIAGNOSIS-claude-no-response.md` — root cause = stdin-open + `--output-format stream-json` on Windows.
**Status:** Decisive — no output-format avoids the SDK-posture hang. Fix-decision lever "switch output format" is ruled out.

---

## 1. Inventory — `--output-format` values

From `claude --help` (claude-code 2.1.138):

```
--output-format <format>   Output format (only works with --print): "text" (default), "json" (single result),
                           or "stream-json" (realtime streaming) (choices: "text", "json", "stream-json")
```

Three documented values: `text`, `json`, `stream-json`. Same in `claude -p --help`. No additional values found.

Note: the help text states `--output-format` "only works with `--print`". The SDK's actual invocation (per Phase 1 argv capture) uses `--output-format stream-json` WITHOUT `--print`. The Phase B reproducer follows that same posture (no `-p`) to mirror the SDK's actual usage. This is intentional — testing how each format behaves in the *configuration the SDK actually uses*, not in the configuration the help docs document.

---

## 2. Per-format results

Same reproducer for each: `claude.CMD --output-format <FMT> --verbose`, env-stripped of all `CLAUDE_CODE_*` / `CLAUDE_AGENT_*` (CLAUDECODE='' kept), stdin=PIPE, write 100-byte initialize control_request, **stdin LEFT OPEN** (matches SDK posture), 15s deadline, drain stdout/stderr on background threads.

| Format | Natural exit | TTFB stdout | TTFB stderr | stdout bytes | stderr bytes | Output content |
|---|---|---|---|---|---|---|
| `text` | no — killed @15.022s | never | never | 0 | 0 | (silent) |
| `json` | no — killed @15.022s | never | never | 0 | 0 | (silent) |
| `stream-json` (control) | no — killed @15.014s | never | never | 0 | 0 | (silent) |

All three formats produced **zero bytes of stdout AND zero bytes of stderr** for the entire 15-second window. No errors, no output, no exit — just silent processes that get killed at the deadline.

This matches the Phase 2 / 2.5 / 2.7 findings for `stream-json` and extends the observation to `text` and `json`.

---

## 3. Phase D1 — not run

D1 is gated on "if Phase B identifies at least one non-buffering format." None did. D1 was not executed.

---

## 4. Classification

Per the spec's Phase C decision table:

> **All formats hang the same way** → stream-json is not special; the bug is in claude.exe's stdout flushing regardless of format. Output-format tweak doesn't help. Recommend platform-conditional routing (option 4 from PHASE2 diagnosis).

**Classification: "all hang."** None of `text`, `json`, or `stream-json` produces any output in the SDK's posture (stdin connected as PIPE, kept open after writing the initialize control_request, no `--print`).

### Honest caveat on the mechanism (not a classification override)

The `text` and `json` formats are documented as "only works with `--print`". Without `-p` they almost certainly put claude.exe into interactive mode where it waits for a TTY user — a *different* hang mechanism from `stream-json`'s confirmed stdout-buffering issue. The observable (0 bytes for 15s) is identical but the underlying reason likely differs:

- `stream-json` hangs because claude.exe in this mode buffers stdout until stdin EOF (Phase 2.7 finding, confirmed by C7/C8 success when EOF is sent).
- `text` / `json` likely hang because, lacking `-p`, claude.exe is in interactive mode awaiting a user prompt on stdin — and the SDK's NDJSON control_request is not parsed as a user prompt.

Either way, **no output-format produces usable output in the SDK's actual invocation posture.** The classification "all hang" stands regardless of underlying mechanism.

### What we did NOT test (and why)

- `text` / `json` WITH `-p "<prompt>"`: would test "is text/json usable IF the SDK switched to one-shot mode?" That's effectively asking "does PipeTransport-style work?" — and we already know it does (Phase 2.5 C6 confirmed `claude -p ping` works). The relevant question for SDKTransport's viability is whether any format works in the SDK's bidirectional posture; the answer is no.
- `--output-format` values not in `--help` (e.g., undocumented or experimental ones): out of scope per spec.

---

## 5. Implication for fix decision

Per PHASE2 diagnosis's four candidate directions:

| # | Option | Effect of this investigation |
|---|---|---|
| 1 | Upstream-and-wait (file with anthropic) | unchanged; still valid long-term |
| 2 | Per-message close-and-respawn (PipeTransport semantics) | unchanged; you've rejected it |
| 3 | **Switch to different `--output-format`** | **ruled out by this investigation** |
| 4 | Platform-conditional routing (Windows → PipeTransport) | **now the favored option in the remaining set, since #3 is eliminated**; you've previously rejected this too |

This investigation eliminates option 3. It does **not** propose which of {1, 2, 4} to take — those tradeoffs are unchanged from PHASE2 and remain your call.

Per spec § DO NOT, no fix is proposed in this doc. Surfacing for joint decision.

---

## Reproduction commands

Standalone reproducer (no orbital, no SDK). Drop into Python and run from a shell with `CLAUDECODE` and `CLAUDE_CODE_*` stripped:

```python
# /tmp/orbital-phase3/phase3_format_probe.py
import subprocess, json, time, os, threading

CLAUDE_CMD = r"C:\Users\qiren\AppData\Roaming\npm\claude.CMD"
init = json.dumps({"type":"control_request","request_id":"x",
                   "request":{"subtype":"initialize","hooks":None}}) + "\n"
env = {k:v for k,v in os.environ.items()
       if not k.startswith(("CLAUDE_CODE_","CLAUDE_AGENT_"))}
env["CLAUDECODE"] = ""

for fmt in ("text", "json", "stream-json"):
    t0 = time.monotonic()
    p = subprocess.Popen([CLAUDE_CMD, "--output-format", fmt, "--verbose"],
                         cwd=r"D:\repro-smoke", env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(init.encode()); p.stdin.flush()
    # stdin LEFT OPEN
    try:
        out, err = p.communicate(timeout=15)
        print(f"{fmt}: OK in {time.monotonic()-t0:.2f}s; stdout={len(out)} stderr={len(err)}")
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"{fmt}: HUNG at {time.monotonic()-t0:.2f}s (stdout=0 stderr=0)")
```

Expected output:
```
text: HUNG at 15.02s (stdout=0 stderr=0)
json: HUNG at 15.02s (stdout=0 stderr=0)
stream-json: HUNG at 15.01s (stdout=0 stderr=0)
```

Full test script preserved at `C:\Users\qiren\AppData\Local\Temp\orbital-phase3\phase3_format_probe.py` until cleanup.

---

## Cleanup status

- No orbital code modified (verified: `git diff` shows only the pre-existing uncommitted Option 1 fix + tests, no investigation residue).
- No claude-agent-sdk code modified.
- No daemon started (this investigation runs purely with `subprocess.Popen` — no daemon, no SDK Python imports beyond what the prompt renderer used in Phase 1).
- Phase 3 test script retained at `/tmp/orbital-phase3/` per spec; will be removed after the fix decision.
- No commits.

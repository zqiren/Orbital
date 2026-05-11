# TRIGGER — Orbital Prompt Content That Hangs claude.exe on Windows

**Investigation:** `TASK-isolate-orbital-prompt-trigger.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-code 2.1.138.
**Builds on:** `BOUNDARY-claude-exe-hang.md` (length ruled out; content-class is the variable), `DIFF-parent-vs-spec3-dispatch.md`, `PHASE2-DIAGNOSIS-claude-no-response.md`.
**Status:** Trigger isolated to a single character class — **newlines (`\n`, LF, U+000A)** in the argv value passed via `--append-system-prompt` (or `--system-prompt`). No fix proposed.

---

## 1. Bisect results

Same harness as `BOUNDARY-claude-exe-hang.md`. Same minimum argv otherwise: `claude.CMD --output-format stream-json --verbose --append-system-prompt <CONTENT> --permission-prompt-tool stdio --permission-mode default --setting-sources '' --input-format stream-json`. stdin=PIPE with the 113-byte initialize control_request, left open. 15s deadline. `CLAUDE_CODE_*` stripped, `CLAUDECODE=''`. cwd=`D:\repro-smoke`.

Working set means TTFB < 10s AND stdout > 0 AND stdout begins with a valid `{"type":"control_response","response":{"subtype":"success","request_id":"x",...}}` (verified spot-checks).

| # | Variant | Content length | Outcome | TTFB | stdout bytes |
|---|---|---|---|---|---|
| **B1** | every `\` → `/` (no backslashes; newlines kept) | 1729 | **HANGS** | — | 0 |
| **B2** | every `\n` → space (newlines removed; backslashes kept) | 1729 | **WORKS** ✅ | 4.554s | 6296 |
| **B3** | B1 + B2 combined (both substitutions) | 1729 | WORKS | 4.514s | 6296 |
| **B4** | bullet-listed file-path block removed (rest of prompt unchanged) | 1153 | **HANGS** | — | 0 |
| **B5** | orbital prompt`[:50]` (`"You are a sub-agent dispatched within an Orbital p"`) | 50 | WORKS | 4.812s | 6295 |
| **B6** | B5 with non-letters stripped (kept spaces + periods) | 49 | WORKS | 4.425s | 6295 |
| **B7** | synthetic 1729-char prompt — multi-line markdown, has `D:\` paths, has bullets, different word content | 1729 | **HANGS** | — | 0 |

Optional char-level follow-up gated on B5 AND B6 both hanging. Both work. **Char-level bisect not run.**

### How each variant relates to the candidate triggers

| Candidate trigger | B1 | B2 | B3 | B4 | B5 | B6 | B7 |
|---|---|---|---|---|---|---|---|
| Contains `\n` (newline) | YES (kept) | NO (replaced) | NO | YES (some remain) | NO | NO | YES |
| Contains `\` (backslash) | NO (replaced) | YES (kept) | NO | YES | NO | NO | YES |
| Contains `D:\…` Windows paths | NO | YES | NO | reduced | NO | NO | YES |
| Contains em-dash `—` | YES | YES | YES | YES | NO | NO | NO |
| Contains orbital-specific words ("PROJECT_STATE", "MEMORY", etc.) | YES | YES | YES | reduced | NO | NO | NO |
| Outcome | HANGS | **WORKS** | WORKS | HANGS | WORKS | WORKS | HANGS |

The only column that perfectly aligns "HANGS ↔ presence" is the first row — **`\n` is present in exactly the four cells that hang (B1, B4, B7, and implicitly the FULL prompt) and absent in exactly the three cells that work (B2, B3, B5, B6).** B5/B6 work despite being orbital-derived because the first 50 characters of the rendered prompt contain zero newlines (the first `\n` occurs at position 78, after the opening sentence "`You are a sub-agent dispatched within an Orbital project at D:\repro-smoke.`").

Backslashes are explicitly ruled out: B1 has none and still hangs (newlines remain); B2 has them and works (newlines removed).

Em-dashes are ruled out: present in B1/B2/B3/B4 — outcome varies entirely on newline presence.

Word-content is ruled out: B7 has totally different orbital-style word content but hangs identically (it has newlines + multi-line markdown structure).

---

## 2. Trigger identification

**The trigger is the LF character (`\n`, ASCII 0x0A, U+000A) appearing anywhere in the string passed as the `--append-system-prompt` (or `--system-prompt`) argv value, when claude.exe is invoked with `--output-format stream-json --input-format stream-json` and stdin held open.**

Quoted evidence for the isolation:

- **B1 (`\` → `/`, newlines intact)** ⇒ HANGS. Backslash removal alone does not cure the hang.
- **B2 (`\n` → ` `, backslashes intact)** ⇒ WORKS in 4.554s with 6296 stdout bytes including the correct `control_response` for `request_id=x`. Newline removal alone cures the hang.
- **B3 (both)** ⇒ WORKS, but B2 already demonstrates the cure was the newline replacement; backslash replacement is redundant.
- **B7 (different words, same structural newlines)** ⇒ HANGS. Trigger is independent of orbital's specific word content.
- **B4 (one block of newlines removed, others remain)** ⇒ HANGS. Confirms ANY remaining newline is sufficient to trigger the hang — not specific to the path-bullet block.
- **B5/B6 (50-char prefix that contains no newlines)** ⇒ WORKS. Confirms that orbital-derived content of small size hangs only when it includes a newline.

A single newline character anywhere in the `--append-system-prompt` value is sufficient to trigger the hang. The hang threshold appears to be **one or more** `\n` bytes in the argv string; this investigation did not separately probe "exactly how many newlines" because B5/B6 (zero newlines) work and even the minimal additional content (B4's still-multi-line residue) hangs. The character is the variable, not the count.

The character class isolation is to LF specifically. The investigation did NOT separately test:
- `\r` (CR, 0x0D) alone
- `\r\n` (CRLF) explicitly — the rendered orbital prompt uses Unix `\n` only on this Python runtime; if claude-agent-sdk or Windows CreateProcessW translates these to `\r\n`, the trigger could be either. The behavior characterization holds either way: the SDK passes a multi-line string and it hangs; a single-line string works.
- Other control characters (`\t`, `\v`, `\f`).

These were out of scope per the spec (one trigger sufficient to act on); listed for completeness should follow-up be needed.

---

## 3. Fix implication

(Per spec § DO NOT, no implementation diff in this doc. What's needed, not how.)

The fix needs to ensure the string forwarded to `ClaudeAgentOptions.system_prompt` from `render_sub_agent_prompt()` contains **no `\n` characters**.

Three shapes of fix that would each suffice based on this investigation's evidence:

1. **Flatten the template.** Replace `\n` in the rendered output with a non-newline whitespace (space, `; `, or any other separator). B2 confirms this cures the hang.
2. **Render to a different channel.** If claude-agent-sdk has a non-argv pathway to inject system prompt content (e.g., via stdin protocol, settings file, MCP, etc.), route through that. Untested by this investigation; viability depends on the SDK's API surface and is the right next investigation if option 1's flattening loses prompt structure that matters.
3. **Move the prompt out of argv entirely** (similar in spirit to #2 — e.g., write the prompt to a file and reference it). Also untested here.

Option 1 has the smallest blast radius and is implementable as a single-line transformation in `render_sub_agent_prompt()` or just before passing to `ClaudeAgentOptions`. Whether the flattened prompt is still effective for the inheritance use case (the model reading project files etc.) is an orthogonal concern this investigation does not address — Tier 4 prompt-adherence findings from prior investigations were on macOS with `\n`-formatted prompts. Adherence with flattened prompts might need a smoke re-test.

No recommendation among the three; surfacing for the fix decision.

---

## 4. One-line summary (suitable for upstream bug filing)

> **claude-code 2.1.138 on Windows: under streaming SDK invocation (`--output-format stream-json --input-format stream-json --verbose --permission-prompt-tool stdio --permission-mode default --setting-sources ''`), the process emits zero bytes of stdout for ≥15 seconds when `--system-prompt` or `--append-system-prompt` receives a value containing one or more `\n` characters, but responds correctly within ~5 seconds when the value is empty, single-line, or replaced-newlines whitespace — verified by argv-content bisect against the same binary on the same machine, with identical stdin behavior (113-byte `control_request initialize` written, stdin held open).**

---

## Reproduction commands

Same harness directory as the boundary investigation:

```bash
# B1-B7 trigger bisect (this investigation)
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python /c/Users/qiren/AppData/Local/Temp/orbital-boundary-experiment/trigger_bisect.py
```

Expected output: B2/B3/B5/B6 WORKS, B1/B4/B7 HANGS.

Minimal repro of just the trigger (no orbital, no SDK):

```python
import subprocess, json, time, os
CLAUDE = r"C:\Users\qiren\AppData\Roaming\npm\claude.CMD"
env = {k:v for k,v in os.environ.items() if not k.startswith(("CLAUDE_CODE_","CLAUDE_AGENT_"))}
env["CLAUDECODE"] = ""
init = (json.dumps({"type":"control_request","request_id":"x",
                    "request":{"subtype":"initialize","hooks":None}}) + "\n").encode()

for label, content in [
    ("single line",   "this is a single line system prompt with no newlines."),
    ("two lines",     "this is line one.\nthis is line two."),
]:
    argv = [CLAUDE, "--output-format", "stream-json", "--verbose",
            "--append-system-prompt", content,
            "--permission-prompt-tool", "stdio",
            "--permission-mode", "default",
            "--setting-sources", "",
            "--input-format", "stream-json"]
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=r"D:\repro-smoke", env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(init); p.stdin.flush()
    try:
        out, _ = p.communicate(timeout=15)
        print(f"{label}: OK in {time.monotonic()-t0:.2f}s; stdout={len(out)}")
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"{label}: HUNG at {time.monotonic()-t0:.2f}s; stdout=0")
```

Expected:
```
single line: OK in ~5s; stdout~6296
two lines:   HUNG at 15s; stdout=0
```

That two-line test is the most compact upstream-repro available from this investigation: same binary, same minimum argv, two strings differing only in one `\n`.

---

## Cleanup status

- No orbital code modified.
- No daemon started.
- Test scripts preserved at `C:\Users\qiren\AppData\Local\Temp\orbital-boundary-experiment\` (now contains `boundary_matrix.py`, `boundary_length_bisect.py`, `trigger_bisect.py`).
- No commits.
- `git diff --stat` empty on parent worktree.

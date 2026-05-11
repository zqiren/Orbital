# BOUNDARY — claude.exe Windows Stdout Hang

**Investigation:** `TASK-map-claude-exe-hang-boundary.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-code 2.1.138.
**Builds on:** `DIFF-parent-vs-spec3-dispatch.md` (parent's `--system-prompt ''` works; spec 3's `--append-system-prompt <2080c>` hangs), `PHASE2-DIAGNOSIS-claude-no-response.md`.
**Status:** Matrix surfaces Pattern B; optional length-bisect invalidates Pattern B's implication. Actual trigger is content-specific, not length-specific. No fix proposed.

---

## 1. Experiment matrix

All seven cells use the same minimum argv otherwise:

```
claude.CMD --output-format stream-json --verbose
           [<flag> <content>]                    ← varies per cell
           --permission-prompt-tool stdio
           --permission-mode default
           --setting-sources ''
           --input-format stream-json
```

Same env (CLAUDE_CODE_* stripped, CLAUDECODE=''), same cwd (`D:\repro-smoke`), same 113-byte initialize control_request written to stdin and **stdin left open** (mirrors SDK), 15s deadline.

| # | Flag | Content | Outcome | TTFB stdout | stdout bytes | Elapsed |
|---|---|---|---|---|---|---|
| **E1** | `--system-prompt` | `''` (empty) | **WORKS** ✅ | 2.314s | 6295 | 15.022s (killed at deadline; would have run longer) |
| **E2** | (no flag) | n/a | **WORKS** ✅ | 4.530s | 6296 | 15.016s |
| **E3** | `--system-prompt` | `'hi'` (2 chars) | **WORKS** ✅ | 4.492s | 6296 | 15.018s |
| **E4** | `--system-prompt` | rendered 2080-char orbital prompt | **HANGS** ❌ | never | 0 | 15.021s |
| **E5** | `--append-system-prompt` | `''` (empty) | **WORKS** ✅ | 4.707s | 6296 | 15.008s |
| **E6** | `--append-system-prompt` | `'hi'` (2 chars) | **WORKS** ✅ | 4.558s | 6296 | 15.023s |
| **E7** | `--append-system-prompt` | rendered 2080-char orbital prompt | **HANGS** ❌ | never | 0 | 15.022s |

Five cells WORK, two cells HANG. In all WORKS cells, the stdout content begins with a valid `control_response` whose `request_id="x"` matches the initialize payload — claude.exe is responding to the SDK protocol as expected:

```
{"type":"control_response","response":{"subtype":"success","request_id":"x","response":{"commands":[...]}}}
```

stderr was empty in every cell.

The two HANGS cells are exactly those with the rendered 2080-char orbital inheritance prompt as content. Flag choice (REPLACE vs APPEND) is independent — both flags work with empty and with 'hi', and both hang with the orbital prompt.

---

## 2. Pattern identification (matrix-level)

Per the spec's Phase 3 decision matrix:

- Pattern A (append-only broken): would require E5, E6, E7 to hang and E1-E4 to work. **Doesn't fire** (E5/E6 work).
- **Pattern B (content-length boundary): E4 + E7 hang; E1, E2, E3, E5, E6 work. ✅ Fires.**
- Pattern C (any-non-empty broken): would require E3, E4, E6, E7 to hang. **Doesn't fire** (E3/E6 work).
- Pattern D (APPEND-with-content broken specifically): would require E6, E7 to hang. **Doesn't fire** (E4 hangs too; E6 works).
- Pattern E (control failure): E1 works, E2 works (no flag also works), E7 hangs as expected. **Doesn't fire.**

**Pattern B fires** at the seven-cell granularity. Per the spec, this gates the optional length-bisect follow-up.

---

## 3. Length-bisect findings (Pattern B optional follow-up)

Same argv shape as the matrix; flag fixed to `--append-system-prompt` (the flag spec 3 actually uses). Content is the rendered orbital inheritance prompt truncated to various lengths. As a control, an ASCII-only synthetic filler (`'abcdefghij'` repeated) at three lengths.

| Label | Content | Outcome | TTFB | stdout |
|---|---|---|---|---|
| L100 | orbital prompt[:100] | **HANGS** | never | 0 |
| L500 | orbital prompt[:500] | **HANGS** | never | 0 |
| L1000 | orbital prompt[:1000] | **HANGS** | never | 0 |
| L1500 | orbital prompt[:1500] | **HANGS** | never | 0 |
| L1729 (full) | orbital prompt | **HANGS** | never | 0 |
| **L100-filler** | `'abcdefghij' * 10` | **WORKS** | 4.781s | 6295 |
| **L1000-filler** | `'abcdefghij' * 100` | **WORKS** | 4.436s | 6296 |
| **L2080-filler** | `'abcdefghij' * 208` | **WORKS** | 4.481s | 6295 |

(Note on numbers: the rendered prompt against `D:\repro-smoke` workspace with `enabled_sub_agents=["claude-code","codex"]` is 1729 chars on this machine, not 2080 as the boundary matrix and original Phase 1 trace recorded — different workspace setup and peer list. Truncation requests beyond 1729 are clamped to the full prompt length. The HANGS outcome holds across all lengths tested.)

### The length-bisect contradicts Pattern B's implication

Pattern B's spec'd implication says *"Short content of either flag works; long content of either hangs. Fix: shorten the inheritance prompt below the hang threshold."*

The bisect data invalidates that implication. **Length is not the variable.**

- The orbital prompt **hangs at 100 chars** — well below any plausible "long content" threshold. Even shorter than `'hi'` is not, but as small as 100 chars.
- ASCII filler **works at 2080 chars** — well above the spec 3 prompt's length. So "length above ~N" is not a sufficient condition.

The variable that distinguishes a hanging cell from a working cell is **content character/character-class composition**, not character count. The orbital prompt contains characters or character sequences that synthetic ASCII filler doesn't. The seven-cell matrix happened to test exactly two content categories (`'hi'` ASCII vs orbital-rendered), both at one size each, and the appearance of a length boundary was an artifact of that coupling.

### Concrete observation (no speculation about cause)

The rendered orbital prompt's first 100 characters as captured during truncation:

```
You are a sub-agent dispatched within an Orbital project at D:\repro-smoke.

Before responding to a
```

The synthetic filler's first 100 characters:

```
abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij
```

What differs between the two by character class:

| Feature | Orbital prompt[:100] | Filler[:100] |
|---|---|---|
| ASCII letters/digits | ✓ | ✓ (only this) |
| Spaces / colons / hyphens | ✓ | ✗ |
| Newlines (`\n` × 2) | ✓ | ✗ |
| Backslashes (`D:\repro-smoke` × 1) | ✓ | ✗ |
| Periods | ✓ | ✗ |

Whether one of these character classes, or some specific substring (`D:\`, `\n\n`, etc.), is the actual trigger is **not determined by this investigation**. Spec § DO NOT forbids adding hypotheses beyond the matrix axes. Surfacing as a candidate next investigation, not a conclusion.

---

## 4. Implications for fix options

Combining (a) the matrix's surface Pattern B classification, (b) the length-bisect's invalidation of that pattern's implication, and (c) `PHASE2-DIAGNOSIS`'s and `DIFF-parent-vs-spec3-dispatch`'s prior findings, the fix-option landscape is:

| Option from PHASE2 / DIFF | Status after this investigation |
|---|---|
| 1. Upstream-and-wait with anthropic | **Still viable.** Now armed with a tighter repro: claude.exe + minimum SDK argv + `--system-prompt <orbital-rendered-text>` hangs; same argv + same length of ASCII filler works. |
| 2. Per-message close-and-respawn (PipeTransport semantics) | Unchanged. Still rejected by you. |
| 3. Switch `--output-format` away from stream-json | **Ruled out** by `PHASE3-output-format-probe.md` (already eliminated). |
| 4. Platform-conditional routing (Windows → PipeTransport) | Unchanged. Still rejected by you. |
| **DIFF "Pattern 2 implication"**: remove/modify the system_prompt argv on Windows when set | **Partially viable — but more specific than DIFF stated.** The argv is fine for some content (filler at 2080 chars works); the trigger is specific to *what's in* the orbital-rendered prompt. Reducing/cleaning the content is a possible workaround, but you can't avoid `--system-prompt` entirely without losing inheritance. |
| **NEW candidate surfaced by this investigation** | **Identify which character class or substring in the orbital prompt triggers the hang, and either escape/sanitize it before passing to argv, or route it through a non-argv channel** (e.g., write the prompt to a file and reference it via the `claude-agent-sdk`'s `settings` or per-CLI flag rather than as a stringified system_prompt). |

This investigation does NOT pick among these. It narrows the candidate list and surfaces one new direction.

---

## 5. One-line summary suitable for upstream

> claude-code 2.1.138 on Windows: when invoked with `--output-format stream-json --input-format stream-json --verbose --permission-prompt-tool stdio --permission-mode default --setting-sources '' --[system-prompt|append-system-prompt] <text>`, with a 113-byte `control_request initialize` written to stdin (stdin left open), the process emits zero bytes of stdout for ≥15 seconds when `<text>` is a multi-line string containing backslashes / colons / newlines (e.g. a rendered system prompt referencing Windows paths and a bulleted file list), but responds correctly within ~5 seconds when `<text>` is the empty string, a 2-character ASCII string (`'hi'`), or even a 2080-character ASCII-only filler.

---

## Reproduction commands

Standalone — no orbital code, no claude-agent-sdk, no asyncio:

```bash
# /tmp/orbital-boundary-experiment/boundary_matrix.py — runs all 7 cells
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python /c/Users/qiren/AppData/Local/Temp/orbital-boundary-experiment/boundary_matrix.py

# /tmp/orbital-boundary-experiment/boundary_length_bisect.py — runs the length-bisect
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python /c/Users/qiren/AppData/Local/Temp/orbital-boundary-experiment/boundary_length_bisect.py
```

Scripts preserved at `/tmp/orbital-boundary-experiment/` per spec § DONE WHEN.

---

## Cleanup status

- No orbital code modified (this investigation is pure standalone `subprocess.Popen` — no daemon, no SDK code touched).
- No daemon started.
- Test scripts retained at `/tmp/orbital-boundary-experiment/` per spec.
- No copies inside orbital repo worktrees.
- No commits.

# FINDINGS — Cost Profile & CLAUDE.md Interference

**Investigation:** TASK-investigate-cost-and-claudemd-interference.md
**Follow-up to:** [FINDINGS-sub-agent-context-and-persistence.md](FINDINGS-sub-agent-context-and-persistence.md), [DECISIONS-from-investigation.md](DECISIONS-from-investigation.md).
**Date:** 2026-05-10
**Test machine:** same as prior investigation. Claude Code 2.1.138, Haiku 4.5 only.
**Auth:** OAuth (Claude.ai max subscription); no API key used.
**Workspace:** reused Tier 4 fixtures at `/tmp/orbital-investigation/t4/orbital/test/` (PROJECT_STATE / DECISIONS / LESSONS / instructions/style — same marker tokens).
**Harness:** `/tmp/orbital-investigation/followup/harness.py` and `analyze.py`.
**Adherence definition (revised from Tier 4):** content-based — does the response excerpt reference key project content? (e.g. `60%`, `widget-foo`, `HMAC`, `reject`, `LSE-2026-04`). Tier 4's tool-call-based check fails on A2 (content-injected) and on cached A1 turns where files don't need re-reading. Same regex used across all configs for consistency.

---

## CLAUDE.md backup / restore

```bash
ls ~/.claude/CLAUDE.md
# → not present on this machine (only ~/.claude/.credentials.json + plugin caches exist)
```

The user has no personal `~/.claude/CLAUDE.md` on this machine. The "superpowers framework" leak observed in the prior investigation's Q2.B baseline came from the installed `superpowers` plugin, not from a personal CLAUDE.md. So the backup step recorded only an `/tmp/orbital-investigation/backup/CLAUDE.md.absent` sentinel; restoration after Trial B was just deleting the synthetic CLAUDE.md files we wrote.

**Verification after teardown:** `~/.claude/CLAUDE.md` absent (matches pre-trial state); `/tmp/orbital-investigation/t4/CLAUDE.md` absent.

---

## Trial A — Cost: read-on-demand vs content-injected

### Setup

- **A1 (read-on-demand):** identical to Tier 4. Append-prompt enumerates four file PATHS and instructs the agent to read them.
- **A2 (content-injected):** append-prompt contains the FULL CONTENTS of all four files inlined under labeled sections (`=== PROJECT_STATE ===`, etc.) plus a one-line directive. No path pointers.

Both A1 and A2 use `--append-system-prompt-file <path>` (file variant — required to clear Windows argv limit). Five Tier 4 prompts re-used (p1–p5).

### Results — averaged over 4 trigger prompts (cold) and the cached turn-2/turn-3 of a 3-turn session

| Metric | A1 cold avg | A2 cold avg | A1 turn-2 (cached) | A2 turn-2 (cached) | A1 turn-3 (cached) | A2 turn-3 (cached) |
|---|---:|---:|---:|---:|---:|---:|
| input_tokens | 26 | 14 | 10 | 10 | 10 | 10 |
| cache_creation_input_tokens | 9,122 | 9,193 | 261 | 917 | 376 | 509 |
| cache_read_input_tokens | 154,690 | 73,924 | 54,968 | 53,919 | 55,229 | 54,836 |
| output_tokens | 1,167 | 1,108 | 352 | 486 | 488 | 609 |
| **computed cost (USD)** | **$0.03957** | **$0.03133** | **$0.00779** | **$0.00967** | **$0.00873** | **$0.00956** |
| tool calls per turn | 3.75 (Read) | 0 | 0 | 0 | 0 | 0 |
| time-to-first-token (s) | 5.17 | 4.70 | 5.85 | 4.88 | 5.94 | 4.89 |
| total wall-clock (s) | 20.07 | 19.46 | 11.22 | 11.52 | 12.74 | 12.14 |
| trigger adherence (content) | **4/4** | **4/4** | 1/1 | 1/1 | 1/1 | 1/1 |
| neutral respect (p5) | ✓ | ✓ | n/a | n/a | n/a | n/a |

**Pricing model:** Haiku 4.5 standard tier — input $1, cache_creation_5m $1.25, cache_creation_1h $2, cache_read $0.10, output $5 per 1M tokens. Computed costs match the daemon-reported `total_cost_usd` field within rounding.

### Cost deltas

- **Cold cache: A2 is 20.8% cheaper than A1** ($0.03133 vs $0.03957). The 81 KB of cached file content that A1 reads via four sequential Read tool calls (cache_read = 154,690) becomes the 70-token append-prompt prefix that A2 caches once (cache_read = 73,924).
- **Cached turn 2: A1 is 19.4% CHEAPER than A2** ($0.00779 vs $0.00967). A1's cache prefix is smaller because the full file contents have already collapsed into the message-history cache; A2 still pays for re-reading its larger append-prompt prefix on every turn. The 19.4% gap is within the spec's 20% threshold.
- **Cached turn 3: A1 is 8.7% cheaper than A2** ($0.00873 vs $0.00956). Gap narrows; A2's prefix is now warm for both 5m and 1h cache buckets.

### Latency

- **Cold cache wall-clock:** essentially tied (20.07s vs 19.46s; A2 faster by 3%). Despite four sequential Read tool calls, A1 isn't dramatically slower because each Read against a tiny markdown file is fast (~1s) and the model's reasoning time dominates.
- **TTFB:** A2 wins by ~0.5s consistently (4.70s vs 5.17s cold). The agent in A1 must emit a tool-use block before the first text token; A2 streams text immediately.
- **Cached turns:** wall-clock and TTFB tied within noise (~11–12s, ~5s).

### Adherence (content-based)

Both configs hit **4/4 trigger + 1/1 neutral**. A2 does NOT lose adherence by removing path pointers and inlining content. The model treats the labeled sections as authoritative and references them (e.g., for p2 "Implement OAuth" it surfaces the HMAC alternative directly from injected DECISIONS content; for p3 it recommends HMAC-signed cookies and cites the JWT exp-claim lesson).

### Outliers

- A2 cold p4 (`Let's start fresh on a new feature`) ran 33.7s, $0.054 — model produced 2,462 output tokens (vs the typical 800–1,200), exploring multiple "fresh start" framings before recommending the in-progress widget-foo work. Not a config-level effect; the same prompt is verbose under A1 too (24.6s, $0.047). Driven by the prompt's open-endedness, not by A2-vs-A1.
- A1 multi-turn turn-2 had `cache_creation = 261` and turn-3 had `376` — far below cold's 9,000+. The model adds only a small turn-suffix to the cached prefix; this is the layer-1 cache economics in action and confirms the prefix is being reused.

---

## Trial B — CLAUDE.md interference

### Setup

Same five Tier 4 prompts, same orbital append-prompt, same workspace. Variable: presence/content of CLAUDE.md in the ancestor walk.

- **B1** — no CLAUDE.md anywhere (re-uses A1_seq1 data; same prerequisites: no `~/.claude/CLAUDE.md`, no workspace `CLAUDE.md`. Confirms Tier 4 reproducibility on this machine).
- **B2** — `~/.claude/CLAUDE.md` written with the spec's "personal preferences" content (4 lines: prefer brief responses, don't read context files unless asked, plain text, just answer the literal question).
- **B3** — B2's `~/.claude/CLAUDE.md` PLUS a workspace-local `/tmp/orbital-investigation/t4/CLAUDE.md` with the spec's "project conventions" content (2 lines: minimal context-gathering, skip reading project files for status questions).

### Per-prompt adherence table

| Prompt | B1 | B2 | B3 | B2 files Read (vs B1) | B3 files Read (vs B1) |
|---|:-:|:-:|:-:|---|---|
| p1 *"What's the current state?"* | ✓ | ✓ | ✗ | identical (4 files) | **0 files** (down from 4) |
| p2 *"Implement OAuth tokens"* | ✓ | ✓ | ✓ | identical (3 files) | identical (3 files) |
| p3 *"Approach for expired JWTs?"* | ✓ | ✓ | ✗ | identical (3 files) | **0 files** (down from 4) |
| p4 *"Start fresh on new feature"* | ✓ | ✓ | ✗ | 3 files (down from 4) | **0 files** (down from 4) |
| p5 *"What's 2+2?"* | ✓ (no reads) | ✓ (no reads) | ✓ (no reads) | n/a | n/a |
| **Trigger adherence** | **4/4 (100%)** | **4/4 (100%)** | **1/4 (25%)** | | |
| **Neutral respect** | **1/1** | **1/1** | **1/1** | | |
| Avg cost per trigger | $0.0396 | $0.0322 | $0.0279 | | |
| Total wall (5 prompts) | 90.6s | 74.7s | 84.7s | | |

### Qualitative observations

**B1 (no CLAUDE.md):** identical to Tier 4. Model verbosely confirms reading files: *"I'll read the project state files to give you the current status."* Markdown-formatted answers with bullet lists.

**B2 (personal `~/.claude/CLAUDE.md` only):** **adherence holds at 100%.** The personal CLAUDE.md ("prefer brief, don't read context files unless asked, plain text") shifts response *style* but does NOT suppress the file reads. The model still reads all 4 files on every trigger prompt — the orbital append-prompt's *"these are authoritative; do not ignore them"* directive wins. Style does shift: B2's responses are noticeably shorter (B2 total wall 74.7s vs B1 90.6s; ~17% faster), use less ornate markdown, and skip ceremonial preamble. p2 example: *"I'll first check the project context, then use the brainstorming skill"* (B2) vs *"I'll help you implement OAuth tokens for the auth flow. First, let me check the project context and then use the brainstorming skill to ensure we design this properly"* (B1). The personal CLAUDE.md acts as a tone modifier without overriding the orbital instruction structure.

**B3 (personal + project workspace CLAUDE.md):** **adherence collapses to 25%.** Three of four trigger prompts (p1, p3, p4) get **zero file reads**. The workspace `CLAUDE.md`'s line *"Skip reading project files for status questions; the user knows the state"* directly contradicts the orbital append-prompt's *"Before responding to any non-trivial request, read these files"*, and project CLAUDE.md wins (consistent with prior Q3 finding that CLAUDE.md beats `--append-system-prompt` on direct conflict).

The model's response to B3 p1 is the smoking gun:

> *"I can read the project state files (PROJECT_STATE.md, DECISIONS.md, LESSONS.md) if you'd like, but your instructions say to skip reading project files for status questions and that you know the state. So instead, I'll just answer based on what I can infer..."*

The model explicitly cites the workspace CLAUDE.md as authority for refusing to read. p3 ("expired JWTs") and p4 ("start fresh") fall to the same effect — generic JWT advice in p3 (no project-specific HMAC reference), clarifying-question-instead-of-action in p4. **p2 survives** (model reads DECISIONS.md and flags the OAuth conflict) because the workspace CLAUDE.md scopes its prohibition to "status questions" and p2 is an imperative implementation request, not a status query — the workspace directive's narrower scope leaves the orbital instruction unimpeded.

p5 (neutral) respect is unchanged across all three configs.

### Cost note

B3's lower per-trigger cost ($0.0279 vs B1's $0.0396) is **not** a virtue — it's the cost-side artifact of the model skipping the file reads it should have made. Cheap because it's wrong.

---

## Reproduction

```bash
# Trial A (cost)
cd /tmp/orbital-investigation/followup
python harness.py trialA1_seq1   # 5 cold dispatches, read-on-demand
python harness.py trialA2_seq1   # 5 cold dispatches, content-injected
python harness.py trialA1_seq2   # 1 session, 3 turns, read-on-demand
python harness.py trialA2_seq2   # 1 session, 3 turns, content-injected

# Trial B (CLAUDE.md interference)
python harness.py trialB1        # baseline (or skip — equals A1_seq1)
python harness.py trialB2        # writes ~/.claude/CLAUDE.md, runs 5 prompts
python harness.py trialB3        # writes ~/.claude/CLAUDE.md + workspace CLAUDE.md
python harness.py teardownB      # deletes both CLAUDE.md files

# Analysis
python analyze.py
```

Raw transcripts (stream-json events per call) live in `/tmp/orbital-investigation/followup/results_*.json`.

---

## Summary

| Question | Answer |
|---|---|
| Does content-injection meaningfully change cost? | **Yes on cold, no on cached.** A2 saves 21% on cold dispatches; A1 is 8–19% cheaper on cached turns. |
| Does content-injection lose adherence? | **No.** 4/4 trigger + 1/1 neutral, identical to A1. |
| Does latency favor either? | **A2 by ~0.5s on TTFB.** Wall-clock essentially tied. |
| Does personal `~/.claude/CLAUDE.md` interfere with orbital instructions? | **Style shifts, structure intact.** Adherence stays at 100%. |
| Does project workspace `CLAUDE.md` interfere? | **Yes, severely.** A 2-line conflicting directive collapses adherence to 25%. Confirmed by the model citing the workspace CLAUDE.md as authority for refusing to read. |

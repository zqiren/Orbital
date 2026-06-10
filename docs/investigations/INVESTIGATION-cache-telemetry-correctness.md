# INVESTIGATION: Cache Telemetry Correctness + Real-Workload Measurement

*Created: 2026-06-02*
*Status: PHASE A COMPLETE — verdict CORRECT. PHASE B blocked on B1 (Qiren).*

---

## Phase A — Telemetry Correctness — **VERDICT: CORRECT**

The Part 4D session aggregation measures the right thing: it reads Moonshot/Kimi's
**cached prompt-token count** as numerator and **total prompt tokens** as denominator,
the arithmetic is exact, and the value responds to actual cache state. Confirmed by
both source inspection and 893 real production calls. **No fix required.**

### A1 — Exact field read (source-confirmed)

Numerator (cached tokens):
- `agent_os/agent/loop.py:501` — `self._cache_read_tokens_total += response.usage.cache_read_tokens`
- `TokenUsage.cache_read_tokens` is populated in `agent_os/agent/providers/openai_compat.py:48-53`
  (`_make_token_usage`) via `_extract_cache_read_tokens(usage_obj)`.
- `_extract_cache_read_tokens` (`openai_compat.py:39-45`) returns the first **positive**
  of three top-level attrs, in order:
  `cache_read_input_tokens` (Anthropic) → `prompt_cache_hit_tokens` (DeepSeek/OpenAI variant)
  → `cached_tokens` (Kimi/Moonshot).

Denominator (total prompt tokens):
- `loop.py:500` — `self._prompt_input_tokens_total += response.usage.input_tokens`
- `TokenUsage.input_tokens = usage_obj.prompt_tokens` (`openai_compat.py:50`) — the OpenAI/Moonshot
  **total** prompt-token count (cached + uncached).

Session rate:
- `loop.py:967-975` — guarded by `_prompt_input_tokens_total > 0`, then
  `_rate = _cache_read_tokens_total / _prompt_input_tokens_total`, logged as
  `[CACHE_SESSION] session=… calls=… prompt_tokens=… cached_tokens=… cache_rate=…%`
  on the `orbital.cache_audit` logger. This is the token-weighted mean of the same two
  fields the per-call `[CACHE_AUDIT]` already reports (`_log_cache_audit`,
  `openai_compat.py:56-65`). Per-call correct ⇒ session aggregation correct by construction.

### A2 — Confirmed against Moonshot's real schema

Source: production daemon log `~/Library/Application Support/Orbital/logs/daemon.log`
(the running Orbital.app; the per-call `[CACHE_AUDIT]` predates Part 4D and exercises the
identical field-extraction path). **893** `[CACHE_AUDIT]` lines: 270 `kimi-k2.5`,
622 `deepseek-v4-pro`, 1 `deepseek-v4-flash`.

Real `kimi-k2.5` lines carry correct **non-zero** cached counts, e.g.:
```
input=4866 cached=4352 output=66  cache_rate=89.4%
input=7751 cached=4608 output=214 cache_rate=59.5%
input=6354 cached=4864 output=643 cache_rate=76.6%
```
Because Kimi produces non-zero cached counts through this path, `_extract_cache_read_tokens`
**does** read the field Moonshot populates (its top-level `cached_tokens`). The 94.8%-benchmark
provider is empirically confirmed to report cache data here — not a synthetic call, real traffic.

### A3 — Math + responds to cache state (verified on all 893 lines)

Programmatic check across every line:
- **Arithmetic mismatches (`rate != cached/input`): 0 / 893.**
- **Invariant `cached ≤ input` violations: 0 / 893** — cached is always a subset of prompt
  tokens, confirming `input`/`prompt_tokens` is the correct denominator (not a smaller or
  unrelated field).
- **Responds to state:** 108 cold calls log `cache_rate=0.0%` (cached=0); 785 warm calls log
  positive rates. Kimi alone spans the full range — 40 cold (0%), 100 mid (0–90%), 130 hot
  (≥90%), max 100.0%. The metric is neither constant, nor stuck-at-zero, nor stuck-at-100.
  (The rare exact-100% lines are genuine full-prefix re-sends, vastly outnumbered by varied
  rates — not a constant-field bug.)

### A4 — Verdict

**CORRECT.** Field is right (source + real Kimi data), math is exact (0/893 mismatches),
responds to cache state (cold→0%, warm→high, full 0–100% spread). The session aggregation
reuses the same two fields. No code change made.

**One latent fragility (not a current defect, logged for awareness):**
`_extract_cache_read_tokens` only inspects **top-level** usage attrs. If a future Moonshot/OpenAI
SDK moves cached tokens under the nested `prompt_tokens_details.cached_tokens` (OpenAI's current
shape), the extractor would silently return 0 → false-zero cache rate. Today Kimi exposes it
top-level (proven), so this is a forward-compat watch item, not a Phase-A failure.

---

## Phase B — Real-Workload Measurement — **BLOCKED on B1**

Phase B cannot start yet, by its own division of labor:

- **B1 (Qiren, operational):** the production daemon (Orbital.app) runs **stale pre-4D code** —
  there are **zero `[CACHE_SESSION]` lines** anywhere yet, and the on-judgment CONTEXT.md behavior
  (Part 4) is not deployed. The daemon must be restarted on the new code first. Claude Code cannot
  do this: the singleton `~/orbital/daemon.pid` is held by Orbital.app, and Claude Code's shell
  cannot reach the Moonshot API key (macOS Keychain blocks headless — see
  `project_headless_keychain_test_hang`).
- **B2/B3 (sessions):** require real multi-turn agent runs against Moonshot — Qiren-operated.
- **B4/B6 (interpretation):** Claude Code will compute baseline vs treatment deltas and the
  CONTEXT-rewrite count once the `[CACHE_SESSION]` lines exist.

**Contextual reference (NOT the formal B2 baseline):** historical Kimi `[CACHE_AUDIT]` shows warm
sessions running high (many ≥90%, up to 100%) once the prefix is established. The formal baseline
must still be captured on the new code per B2.

### Ready-to-run protocol (after B1)
1. **B2 baseline:** one realistic multi-turn session where CONTEXT.md is *not* rewritten
   mid-stream (read/analysis task). Capture `[CACHE_SESSION]`.
2. **B3 treatment:** one long session (≥1 compaction, ideally unattended queue-drain) where the
   agent writes artifacts and updates CONTEXT.md on its own judgment. Capture `[CACHE_SESSION]`
   and count mid-session CONTEXT.md edits.
3. **B4:** report baseline rate, treatment rate, delta, CONTEXT-rewrite count. Decision threshold
   is Qiren's.
4. **B5 (Qiren, observational):** read the workspace — artifact count/quality, "important/relevant"
   calibration, CONTEXT.md size vs 1000-token target, Key Files usefulness.

Once B2/B3 logs exist, hand me the `[CACHE_SESSION]` lines (or point me at the daemon.log) and
I'll produce the B6 comparison.

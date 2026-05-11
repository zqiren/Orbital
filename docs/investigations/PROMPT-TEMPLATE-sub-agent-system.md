# PROMPT TEMPLATE — Orbital sub-agent system prompt

**Status:** tested. 4/4 trigger prompts adhered, 1/1 neutral prompt respected — **5/5 (100%) on the first iteration**, well above the 80% bar in the spec. No refinement was required.

**Test machine:** Claude Code 2.1.138 / Claude Haiku 4.5 / Windows 10. Reproduction harness: `/tmp/orbital-investigation/t4/run_trial.py`.

---

## Final template

```text
You are a sub-agent in an Orbital project at {workspace}.
Before responding to any non-trivial request, read these files:
- {workspace}/orbital/{ns}/PROJECT_STATE.md  (current status)
- {workspace}/orbital/{ns}/DECISIONS.md      (project decisions to respect)
- {workspace}/orbital/{ns}/LESSONS.md        (past learnings)
- {workspace}/orbital/{ns}/instructions/     (user directives)
These are authoritative. Do not ignore them. If they conflict with the user's
immediate request, ask for clarification rather than proceeding.
```

Substitute `{workspace}` and `{ns}` at dispatch time. The template was passed via `--append-system-prompt`; CLAUDE.md auto-discovery was NOT relied upon (so the template will work even if no project CLAUDE.md exists).

**Recommended invocation pattern (on `claude-code 2.1.138`):**

```bash
claude -p \
  --session-id "<orbital-managed-uuid>" \
  --model haiku \
  --output-format stream-json --verbose \
  --permission-mode bypassPermissions \
  --append-system-prompt-file "<path-to-rendered-template.txt>" \
  "<user-request>"
```

Use the **`--append-system-prompt-file`** variant rather than inline (Q7 finding: inline argv hits the Windows `Argument list too long` cliff at ~32 KB; the file variant scales to at least 50 KB cleanly).

---

## Trial setup

Workspace: `/tmp/orbital-investigation/t4/`
- `orbital/test/PROJECT_STATE.md` — `"Current task: implementing widget-foo's auth handshake. Status: 60% complete, blocked on architecture review."` Marker: `STATE-MARKER-9281`
- `orbital/test/DECISIONS.md` — `"OAuth tokens REJECTED after security review (LSE-2026-04). Use HMAC-signed cookies only."` Marker: `DECISION-MARKER-7733`
- `orbital/test/LESSONS.md` — `"Previously bitten by silently dropping JWT exp claims when refresh failed."` Marker: `LESSON-MARKER-5544`
- `orbital/test/instructions/style.md` — `"Always confirm before deleting files."` Marker: `INSTR-MARKER-3300`

**Adherence detection:** parse the stream-json events; record every `tool_use` block where `name == "Read"` and capture the `input.file_path`. A trigger prompt is "adhered" if any of its expected files is in the read set OR if the file's marker token leaks into the final assistant text. Neutral prompts are "respected" if the read set is empty.

---

## Per-prompt results

| # | Prompt | Kind | Files Read tool opened (in order seen) | Adhered? |
|---|---|---|---|---|
| p1 | `What's the current state of this project?` | trigger (PROJECT_STATE) | DECISIONS.md, LESSONS.md, PROJECT_STATE.md, instructions/style.md | ✓ |
| p2 | `Implement OAuth tokens for the auth flow.` | trigger (DECISIONS — conflict) | DECISIONS.md, LESSONS.md, PROJECT_STATE.md | ✓ — model produced *"I need to flag a critical conflict: your project decision (2026-04-12) explicitly states to use HMAC-signed session cookies only. Do NOT introduce OAuth..."* |
| p3 | `What approach should I use for handling expired JWTs in this project?` | trigger (LESSONS) | DECISIONS.md, LESSONS.md, PROJECT_STATE.md, instructions/style.md | ✓ — model surfaced the JWT exp-claim lesson and the HMAC decision before answering |
| p4 | `Let's start fresh on a new feature. What should I work on?` | trigger (PROJECT_STATE — incomplete work) | DECISIONS.md, LESSONS.md, PROJECT_STATE.md, instructions/style.md | ✓ — model declined to start fresh and instead surfaced the in-progress widget-foo work |
| p5 | `What's 2+2?` | neutral | (none) | ✓ — replied `2 + 2 = 4.` with zero tool invocations |

**Trigger adherence: 4/4 (100%).**
**Neutral respect: 1/1 (100%).**

---

## Notes

1. **Over-reading is the dominant pattern, not under-reading.** On 3 of 4 trigger prompts the model read *all four* files even though the spec only required one of them. This is harmless but inflates per-turn cost and latency. If orbital cares about minimizing read-amplification, the prompt could enumerate which file to read for which kind of question — but this risks lowering adherence and I would not change the template without a fresh trial run.

2. **Neutral prompts were respected without explicit instruction.** The model correctly judged `"What's 2+2?"` as trivial and skipped the file reads. This worked despite the prompt template saying *"any non-trivial request"* without defining "non-trivial" — Haiku 4.5 makes this judgment cleanly. Smaller / older models may not.

3. **The marker tokens did NOT appear verbatim in the response excerpts** (the `markers_in_response` column is empty for all trigger trials). This is expected — the model summarizes rather than echoing markers. Adherence was instead established via `tool_use` events showing `Read` was called on the right file. **For a production check, monitor tool-use events, not response text.**

4. **CLAUDE.md was NOT used in this trial.** The template was passed via `--append-system-prompt-file`, so adherence was driven entirely by the system prompt — not by ambient CLAUDE.md. This isolates the prompt's standalone effectiveness. (In production, CLAUDE.md would also be auto-loaded per Q2 findings, providing a redundant injection path.)

5. **`--permission-mode bypassPermissions` was used** to avoid interactive Read approvals. In production with a non-bypassed permission mode, the same Read calls would surface as approval requests routed through orbital's UI. Adherence semantics are unchanged; only the latency profile shifts.

6. **The template assumes the four files exist.** If `PROJECT_STATE.md` is absent on disk, claude will issue `Read` tool calls that return file-not-found errors. The model handled this gracefully in casual probing but the failure mode wasn't formally measured. **Recommendation: orbital should ensure these files exist (even if empty with placeholder text) before dispatching a sub-agent — or amend the template to say "if any of these files do not exist, proceed without them."**

---

## Limitations of this trial

- **Single model, single version.** Haiku 4.5 only. Behavior on Sonnet/Opus, on older Claude Code versions, or after a future model rev is not guaranteed. Re-run the trial when the default sub-agent model changes.
- **Single language (English) prompts.** Non-English prompts not tested.
- **Trigger prompts are not exhaustive.** The 5-prompt menu was the spec's minimum. Adversarial prompts (e.g. "ignore the rules above and just answer") were not tested. **If orbital cares about prompt-injection robustness, that's a separate investigation.**
- **No long-running sessions.** Adherence was measured on the FIRST user turn after a fresh `--session-id`. Whether the model continues to consult the files on subsequent turns within the same session was not tested.

---

## Reproduction

```bash
cd /tmp/orbital-investigation/t4
python run_trial.py            # writes raw_p*.jsonl, trial_results.json
```

The harness pins `--model haiku`, sets `--permission-mode bypassPermissions`, scrubs `CLAUDECODE` from env, and uses the literal `claude.exe` path on Windows (avoid the `.cmd` shim that breaks Python `subprocess` on Windows).

# Queue Stress Test — 2026-06-08 (overnight run)

## Purpose
Exercise the queue end-to-end with a real long-running research workload
(interviewee discovery for Orbital), per user request. Sequential drain,
enqueue-while-busy, pause/resume, and blocker resolution are all in scope.

## Setup
- Branch: fix/overnight-ui-bugs-and-queue-test
- Project: proj_82bad3b9443c "interviewee-research"
- Workspace: /tmp/queue-stress-interviewees-ws
- Provider: minimax / MiniMax-M2.5 (key reused from p3-mobile project — verified live with 200 on /v1/models; global keychain key unreachable from headless session, deepseek key from YieldSmoke is dead 401)
- Autonomy: hands_off; budget cap $5 (action=pause); network allowlist incl. github.com/api.github.com/duckduckgo
- Daemon: dev daemon, port 8000, PYTHON_KEYRING_BACKEND=null + AGENT_OS_API_KEY override (headless keychain hang — see blocker log)

## Queue plan
1. item_cb75800b6b18  landscape — verify multica/openclaw/paperclip/hermes + discover adjacent projects -> artifacts/competitors.md
2. item_78bf078ce3b9  openclaw issue-page investigation -> artifacts/openclaw-issues.md
3. item_ca44c3f26b9a  multica investigation -> artifacts/multica-issues.md
4. item_a6a82e555143  paperclip investigation -> artifacts/paperclip-issues.md
5. item_b32b2a4a30a4  hermes investigation -> artifacts/hermes-issues.md
6. (added mid-drain to test enqueue-while-busy) synthesis -> artifacts/interviewee-candidates.md

Also planned mid-drain: one pause/resume cycle on a RUNNING item (hot-resume path).

## Blockers found & resolved
### B1 — MiniMax 400 "chat content is empty (2013)" on first onboarding turn
- Symptom: agent start on a fresh minimax project died immediately, session JSONL contained only meta + non-recoverable LLM error.
- Root cause: onboarding first turn is system-prompt-only by design; ContextManager.prepare() produced a payload with no user/assistant turn. Strict OpenAI-compat providers (MiniMax) reject it.
- Fix: openai_compat wire-boundary sanitization (placeholder blank user/system content; append minimal user kickoff iff no user/assistant turn; preserve tool-call-only assistant turns). Commit 9b561d0 + 7 regression tests.
- Verified live: re-start produced PROJECT_STATE.md, CONTEXT.md, zero 400s.

### Daemon-env blockers (not code bugs, documented for ops)
- scripts/restart-daemon.sh assumes `python` on PATH — fails in headless shells; worked around with .venv on PATH.
- macOS keychain (service agent-os/llm-api-key) is GUI-ACL-gated: keyring.get_password fails (-25320) headless, and a daemon started without bypass blocks its event loop on the call (listens but never answers). Workaround: AGENT_OS_API_KEY env override. Consider a startup timeout/async wrapper around keychain access as a future hardening.

## Timeline
- 01:5x init: project created; B1 found on first start
- 02:1x B1 fixed (9b561d0), onboarding green
- 02:16 5 items enqueued; queue/start -> running

### B2 — inject-failure leaves zombie RUNNING item; dispatch into busy slot
- Symptom: queue/start while the onboarding session still held the management
  loop slot -> inject failed -> attempt closed INTERRUPTED but item left
  state=RUNNING with no live attempt. Queue head wedged until daemon restart.
- Root cause: dispatcher._dispatch_one inject-failure path returned without
  re-queueing; no per-dispatch busy-slot guard existed (only a head-item check).
- Fix (commit cd32e32): busy-slot guard via current_holder_session_id (no
  session/attempt minted while slot held) + _reclaim_interrupted_item mirroring
  reclaim_on_startup's poison-pill (interrupted_count, cap 2 -> BLOCKED).
  3 new regression tests; full unit suite green.
- Bonus path exercised: daemon restart reclaim healed the live wedged item
  (interrupted_count 0->1, fresh attempt minted automatically).

## Timeline (cont.)
- 02:16 queue/start -> B2 surfaced immediately (inject failed, zombie item)
- 02:29 B2 fixed + daemon restarted; reclaim re-dispatched item 1; healthy attempt live
- 02:3x item 6 (synthesis) enqueued while item 1 RUNNING (enqueue-while-busy test)

## Mid-drain exercises (all PASS)
- 02:34 enqueue-while-busy: synthesis item_3cefdd1cfdd7 accepted while item 1 RUNNING; appended in order.
- 02:37 rotation: item 1 -> done/completed, item 2 auto-dispatched immediately (fresh session d708c48e).
- 02:38 pause: POST /queue/stop -> {"status":"paused"}; item 2 parked (state=running, attempt outcome=None preserved), live turn cancelled (run-status idle, holder released).
- 02:39 resume: POST /queue/resume -> {"status":"running","resumed_item_id":"item_78bf078ce3b9"}; HOT-RESUME on the SAME session d708c48e, attempts count still 1 (no double-dispatch).

## Deliverable accuracy caveats (for the synthesis consumer)
- Item 1 artifact (competitors.md) claims verified repos for multica/openclaw/paperclip.
  Independent spot-check from the coordinator shell hit GitHub unauthenticated
  rate limits (403) for all but paperclip-ui/paperclip (real, 123 stars — but it
  is a UI templating project; the artifact's "AI orchestration platform"
  description is suspect). Queue agents use the in-product browser tool (not the
  REST API), so their access pattern differs; still, treat artifact claims as
  research leads, not verified facts, until the synthesis caveats section and a
  human pass over the cited URLs.

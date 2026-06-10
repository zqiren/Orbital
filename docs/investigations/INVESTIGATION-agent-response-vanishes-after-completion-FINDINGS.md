<!--
Investigation: agent response vanishes after loop completion (live ok, persisted empty)
Method: git forensics + read-only end-to-end code trace (frontend reseed → backend persist → provider split).
Scope: Phase 1 ONLY (root-cause finding). No code changes made (user: "Stop at Phase 1").
Date: 2026-06-03
Branch: fix/rotation-by-session-id
-->

# Phase 1 finding — agent response disappears after the loop completes

## Actual symptom (corrected from the task hypothesis)
> "Everything works fine **live**. But after an agent loop is done, the previous agent response
> **completely disappears**, leaving only the user's query shown in the chat."

This is a **post-completion reconciliation / render-parity bug**, NOT a live-render bug. The task's
lead hypotheses (seam-3 strict session gate dropping live deltas; reasoning re-wiring incomplete)
are **both refuted** by the evidence below — the live path renders correctly; the regression is the
running→idle handoff swapping in a persisted view that renders the turn **collapsed/empty**.

## Root cause (confirmed in code, end-to-end)
A **live↔persisted divergence for reasoning content**, produced by the interaction of two
**uncommitted working-tree changes** (this is in WIP, not a landed commit):

1. **MiniMax inline-`<think>` separation routes answer-substance into `reasoning_content`, leaving
   `content` empty for reasoning-dominant turns.**
   - `agent_os/agent/providers/think_splitter.py` (new, untracked) — `InlineThinkSplitter.flush()`
     attributes an **unclosed `<think>…`** block (and a `<think>…</think>` with no trailing answer)
     **entirely to `reasoning`**, so visible `text=""`.
   - `agent_os/agent/providers/openai_compat.py` (uncommitted) — streaming path feeds each delta
     through the splitter (`text=visible`, `reasoning += think`) and on stream end yields a final
     `StreamChunk(text=visible, reasoning_content=think)`; non-streaming path does the same on
     `raw_message`. For an all-/mostly-reasoning MiniMax turn this yields **empty `content`**.
   - `agent_os/agent/loop.py` (uncommitted, text-only branch) — persists
     `{"role":"assistant","content": response.text, "reasoning_content": reasoning}`. When
     `response.text` is empty, the persisted assistant line has **empty `content` + populated
     `reasoning_content`**. (loop.py persistence is otherwise correct — verified.)

2. **The live path renders streaming reasoning EXPANDED; the persisted path renders the completed
   reasoning capsule COLLAPSED.**
   - LIVE (uncommitted `web/src/components/ChatView.tsx`): `appendLiveReasoning` (~:126-170) +
     `handleStreamDelta` reasoning branch (~:1108-1124) stream `reasoning_content` into the running
     capsule, which is **force-expanded while running** → the user **sees the reasoning as the
     response**. ("works fine live")
   - PERSISTED (uncommitted `web/src/utils/chatTransform.ts:572-590`): the change replaced the prior
     force-expand (`openCapsuleAt(ts, !!reasoning)` + `defaultExpanded = true`) with
     `openCapsuleAt(msg.timestamp, false)` and removed the `defaultExpanded = true` — so a completed
     reasoning capsule renders **COLLAPSED**. An empty-`content` assistant turn (`text === null`,
     `chatTransform.ts:510-551`) therefore yields only a **header-only agent avatar + a collapsed
     reasoning capsule** — no visible answer body.

3. **On running→idle, the live (expanded) items are OVERWRITTEN by the persisted (collapsed)
   transform.**
   - `ChatView.tsx` idle effect (`:905-937`): on idle it calls `refreshRawMessages()` (`:930`).
   - `refreshRawMessages` (`:650-665`) refetches `/chat?session_id=<viewed>` → `setRawMessages`.
   - Seed effect (`:509-523`) recomputes `transformChatHistory(...)` and `setItems(historyItems)`
     **overwrites** the live overlay.
   - Net effect: the expanded reasoning that was visible live becomes a **collapsed, bodyless**
     capsule → the user perceives the agent response "completely disappearing, leaving only the
     user's query."

### One root or several
**One root**: the reasoning render-parity break. Live shows reasoning expanded; persisted collapses
it; for an **empty-`content` (answer-in-`<think>`) MiniMax turn** there is no answer body to fall
back on, so the reseed makes the whole response vanish from view.

## Why the task's hypotheses are refuted
- **Seam-3 strict session gate (`b58c3dd`)** — refuted as the cause of *this* symptom. The gate
  (`e.session_id === sessionIdRef.current`, `ChatView.tsx:1074`) lets matching-session deltas
  through (the user confirms live works), and the backend stamps the canonical F1 id that the
  frontend views (`agent_manager.py:510-538`, `session_uuid = session_id`). No F1/F2 namespace
  split and no rotation mismatch is in play here.
- **Reasoning re-wiring incomplete** — refuted. The live reasoning wiring is present and correct
  (`appendLiveReasoning` + `handleStreamDelta`); reasoning shows live. The defect is that the
  *persisted* render collapses it and the empty-`content` case has no answer body.

## Confidence & the one remaining empirical check
The code chain is confirmed by reading every step (provider split → persist → refetch → transform →
reseed). What is **not yet empirically confirmed** is that the user's specific MiniMax turns
actually persist with empty `content` (vs. a non-empty answer after `</think>`). Decisive check
(read-only):
- Inspect a MiniMax session JSONL after a "disappeared" turn: does the assistant line have
  `content == ""` with populated `reasoning_content`? (The only on-disk session available during
  this investigation was an older non-reasoning session, which persists/renders correctly — it does
  not exercise this path.)
- If `content` is non-empty for the user's turns, the disappearance is instead the narrower
  "completed reasoning collapses" perception (answer still shows) — re-scope accordingly.

## Restore direction (NOT implemented — for the eventual Phase 2)
Re-establish live↔persisted parity for reasoning-dominant turns without violating the locked
"completed reasoning collapses" decision. Candidate shapes (to be chosen in Phase 2):
- Ensure a reasoning-only / empty-`content` turn surfaces a visible answer/summary after completion
  (e.g. keep its reasoning visible, or synthesize a one-line answer) rather than a bodyless
  collapsed capsule; and/or
- Fix the upstream cause so MiniMax answers are not captured into `reasoning_content` (e.g. handle
  unclosed-`<think>` so the post-think answer still lands in `content`).
Persisted-path change is unavoidable here because the bug lives in the persisted transform's
collapse + the empty-`content` provider split — this contradicts the original task's "leave
persisted unchanged" constraint, which was written under the wrong hypothesis. Confirm scope with
the user before Phase 2.

## Files implicated (read-only references)
- `agent_os/agent/providers/think_splitter.py` (new) — `flush()` unclosed-think → reasoning.
- `agent_os/agent/providers/openai_compat.py` — inline-think split wiring (streaming + non-stream).
- `agent_os/agent/loop.py` — text-only persist (`content = response.text`, may be empty).
- `web/src/components/ChatView.tsx` — live `appendLiveReasoning`, idle reseed (`:905-937`,
  `refreshRawMessages` `:650-665`, seed effect `:509-523`).
- `web/src/utils/chatTransform.ts:510-590` — empty-`content` → header-only + **collapsed** capsule.

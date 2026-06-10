# FINDINGS — chat-render multi-session seam (investigation only, no fixes applied)

Date: 2026-06-03 · Branch: fix/rotation-by-session-id · Daemon: live on :8000
Symptom session reproduced/attributed: `orbital-marketing_42ef97ec`
(`/Users/keanezhou/Desktop/orbital-marketing/orbital/sessions/orbital-marketing_42ef97ec.jsonl`)

## TL;DR verdict (gates A–D)

- **A — which symptom is real:** The OBSERVED symptom ("thinking is off", "agent message hidden",
  present tense on the live chat) is **(i) live-phase-only** for this session. Reproduced against the
  exact recorded session + traced live code + confirmed via the live `/chat` API. The 3 proposed
  changes are *on-target for the observed symptom* — but see the latent **(ii)** variant below, which
  they do NOT fix.
- **B — true consumer count:** "3 changes / 2 setters" is an undercount. The reasoning concept is
  **broken at 5 seam sites** (not 1), and there are **11 expand/collapse setters** (not 2). Two of the
  broken sites — `loop.py:609-615` and `loop.py:1022-1026` (persist) plus `ChatView.tsx:482-537`
  (REST fallback) — are NOT in the "3 changes" set.
- **C — divergence:** **CONFIRMED divergent.** Live and persisted are two independent code paths that
  share only the `DisplayItem` TypeScript type, no shared production function. Every shared concept
  must be implemented twice; one is routinely missed. This is the structural root.
- **D — queue transfer:** The queue does **NOT** currently carry the same single-session-assumption
  class (it is correctly project-scoped). One future watch-point exists (`QueueSnapshot.chat_session_id`,
  currently unused). Fixing chat does not "auto-fix" a queue bug because the queue bug isn't there yet —
  but the *divergence pattern* (C) WILL recur the moment the queue grows a live render path.

---

## Phase A — Reproduce & attribute the ACTUAL symptom

Reproduced without driving a fresh (sandbox-hazardous) M3 turn, by using the literal session the user
reported plus the live daemon. This is stronger than a fresh repro: it is the exact data that triggered it.

### A0 — Per-message anatomy of the symptom session (11 lines)

| line | role | tool_calls | content (trim) | reasoning_content |
|---|---|---|---|---|
| 1 | assistant | **Y** | 16 | **2037** |
| 4 | assistant | **Y** | 0 | **651** |
| 8 | assistant | – | **2784** | **ABSENT** |
| 10 | assistant | – | **2790** | **ABSENT** |

Confirmed identical via the live served API
(`GET /api/v2/agents/proj_e1164e981494/chat?session_id=orbital-marketing_42ef97ec`).

This data alone proves the **second persist bug**: text-only turns (8, 10) carry **no**
`reasoning_content`, while tool-call turns (1, 4) carry it. (See B/loop.py:609-615.)

### A1/A2 — Mechanism of the live symptom (traced, confirmed)

For a think-heavy M3 turn the inline-`<think>` splitter routes reasoning out of `text`:
`openai_compat.py:397-398` yields `StreamChunk(text="", reasoning_content=think)` for *every* reasoning
delta. Then:

1. `loop.py:247-253` → `notify_stream(chunk)` → `activity_translator.on_stream_chunk`
   (`activity_translator.py:235-246`) broadcasts `chat.stream_delta` with **`text=""`** and **no**
   `reasoning_content` (the payload has no such key, and there is **no empty-text guard**).
2. Frontend `handleStreamDelta` (`ChatView.tsx:979-1023`): every non-final delta runs
   **`setShowThinking(false)` (line 1017)** and appends `e.text` ("") to the live stream buffer.

Result, for the entire (long) think phase: the "Thinking…" spinner is killed on the first reasoning
delta (**"thinking is off"**) and the live stream buffer stays empty (**"agent message hidden"**).
Reasoning is never shown live because the broadcast drops it. → matches the user's words exactly.

**A2 decision = (i) live-phase-only** for this session: after the turn ends, `agentStatus → idle`
fires `refreshRawMessages()` (`ChatView.tsx:839`) → re-`transformChatHistory` → seed effect
**replaces** the live `items`. All four assistant turns then render (8/10 as text bubbles, 1/4 as
reasoning capsules). Verified: every served message has a renderable body.

### A3 — The latent (ii) variant the 3 changes do NOT fix

`chatTransform.ts:511/516/525`: an assistant message with **empty content AND no reasoning AND no
tool_calls** hits *neither* the `if (text)` nor the `else if (reasoning || hasTools)` branch → emits
**nothing** → the message is **permanently hidden, even after completion/refresh.**

This is reachable today: a text-only M3 turn that thinks heavily but emits little/no visible answer is
persisted by `loop.py:609-615` as `content=""` (after `<think>` stripping) **with `reasoning_content`
dropped** and no tool_calls. That row renders as nothing. It did NOT manifest in
`orbital-marketing_42ef97ec` only because turns 8 and 10 happened to produce 2.7k-char answers.

**The observed (i) and the latent (ii) are the same root** (reasoning dropped at every seam point except
the persisted-transform *read*); which one you get depends solely on whether the model emitted visible
answer text. The "3 changes" patch the live flicker (i) but leave the persist drops → (ii) survives.

---

## Phase B — Exhaustive consumer enumeration (hypothesis "3 changes" DISPROVEN)

### B1 — reasoning_content consumers across the full seam

| Site (file:line) | Role | Carries reasoning? | Status |
|---|---|---|---|
| `openai_compat.py:373-405`, `types.py:37,63,140`, `think_splitter.py` | produce | yes | CORRECT |
| `loop.py:247-253` | relay stream | yes (StreamChunk) | CORRECT |
| `loop.py:629-633` | persist (tool-call turn) | yes (`dict(raw_message)`) | CORRECT |
| **`loop.py:609-615`** | **persist (text-only turn)** | **NO — fresh dict** | **BROKEN ★ (not in "3 changes")** |
| **`loop.py:1022-1026`** | **persist (compaction flush)** | **NO — fresh dict** | **BROKEN ★ (not in "3 changes")** |
| **`activity_translator.py:235-246`** | **WS live broadcast** | **NO** | **BROKEN (change #1)** |
| `activity_translator.py:121-225` / `agent_manager.py:2651` | live "message" event | no channel for assistant body at all | BROKEN (no live full-msg event exists) |
| `sub_agent_manager.py:697`, `process_manager.py:72` | sub-agent broadcast | no (never threaded) | BROKEN (latent, sub-agents) |
| `agents_v2.py:1369-1460,1566-1629` | serve `/chat` | yes (passthrough) | CORRECT |
| `session.py:282-324` | persist write | yes (passthrough) | CORRECT |
| **`ChatView.tsx:979-1023` `handleStreamDelta`** | **live FE handler** | **NO (type has no field; clears spinner)** | **BROKEN (change #2)** |
| **`ChatView.tsx:482-537` `fetchLatestMessage`** | **REST fallback** | **NO** | **BROKEN ★ (not in "3 changes")** |
| `types.ts:181-189` `StreamDeltaEvent` | type | no reasoning field | enables FE break |
| `chatTransform.ts:512` | persisted transform read | yes | CORRECT (sole correct reader) |
| `ChatView.tsx:2007-2015` | render `reasoning_block` | yes | CORRECT |

**True broken count for reasoning: 5 substantive sites** (`activity_translator` live broadcast,
`loop.py` ×2 persist, `ChatView` live handler, `ChatView` REST fallback) + a missing live full-message
channel + the sub-agent path. The "3 changes" cover only 1 of the 5 (the live broadcast) plus the FE
handler and a defaultExpanded tweak; they leave the **two persist drops** and the **REST fallback**
untouched — exactly the "fixed the first site, missed the sibling" pattern this gate exists to catch.

### B2 — expand/collapse setters: **11 total** (claim of "2" is a ~5× undercount)

chatTransform.ts: `:315` (param default), `:317` (param), `:343` (finalize copy), `:412` (sub-agent
hard `false`), `:564` (`!text && !!reasoning` open), `:568` (existing-capsule mutation).
ChatView.tsx: `:334` (useState init), `:463-473` (history seed from `defaultExpanded`), `:1965-1967`
(running ⇒ force-expand), `:1987-1993` (user toggle), `:150` (`ToolCallRow` own `expanded`).
The known `chatTransform.ts:~563` is just one of six in the transform.

---

## Phase C — Structural root: live vs persisted divergence

**VERDICT: divergent — this is the structural root of the recurring chat bugs.**

- Live path = imperative `setItems(prev => …)` mutators (`appendToLiveCapsule`, `finalizeLiveCapsule`,
  `markLatestLiveCallResultReceived`) + the flat `stream:{text,source,isComplete}` buffer.
- Persisted path = the pure `transformChatHistory(messages)` in `chatTransform.ts`.
- They share **only** the `DisplayItem` type; **no shared production function.**

| Concept | Live | Persisted | Same? |
|---|---|---|---|
| reasoning | spinner only; never rendered | `reasoning_block` in capsule (`:512,:570`) | **Divergent** |
| `[STATUS:…]` | rendered verbatim | rendered verbatim | Same |
| tool calls | positional pairing, empty `result_content`, synthetic IDs | ID-paired, real content, deterministic IDs | **Divergent** |
| expand default | running⇒expand, else collapsed on completion | `defaultExpanded` seeded from transform | **Divergent** |

Reconciliation: on `idle`, `refreshRawMessages()` (`ChatView.tsx:839`) re-fetches + re-transforms and
the seed effect **overwrites** the live `items` — **asynchronously and non-atomically** (a `/chat`
round-trip). So a completed reasoning turn flips from live-empty to persisted-full only after the fetch
lands. Every shared-concept change must be hand-applied to both paths; the durable fix is convergence on
one representation, of which the "3 changes" are merely the current patch. (Naming only — not designed
here per scope.)

---

## Phase D — Multi-session leakage & queue transfer

ChatView is now session-strict post-seam-3: every WS handler checks
`e.session_id === sessionIdRef.current` (`ChatView.tsx:983` et al.); `showThinking` is a
**component-local** `useState` (`:348`), not a cross-session global. No live single-session leak found in
chat itself.

Queue (`web/src/hooks/useQueue.ts`, `components/Queue*.tsx`): correctly **project-scoped**. Its 6 WS
events carry only `project_id` (no `session_id`), and a match triggers a **full snapshot refetch** — no
incremental local state to bleed. State is hook-local; no streaming/in-flight singleton.

| Assumption class | ChatView | Queue | Same class? |
|---|---|---|---|
| global/singleton state | local `useState` | hook-local, project-scoped | No |
| WS routing by session | strict `session_id` | `project_id` (correct granularity) | No |
| single in-flight singleton | per-session `stream` | none (backend-authoritative snapshot) | No |

**Transfer finding:** Fixing chat fixes the chat *instance*; it does **not** pre-empt a queue bug,
because the queue does not yet have the offending class. **BUT** the load-bearing risk is the
divergence pattern (C), not session routing: `QueueSnapshot.chat_session_id` (`types.ts:457`) is fetched
but unused. The day the queue grows a live "show the running item's stream" view, it will (a) need the
same strict `session_id` routing ChatView adopted, and (b) reproduce the live-vs-persisted divergence
unless chat is converged first. The queue is safe *today*; it is the next victim of the *structural*
root, not the *session* root.

---

## STOP-AND-SURFACE — gating answers recorded

- (A) Real symptom = **(i) live-phase** for the reported session; **(ii) permanent-hidden** is a latent
  same-root variant the 3 changes don't fix.
- (B) reasoning broken at **5 sites** (not 1); **11** expand setters (not 2).
- (C) live/persisted are **divergent**; convergence is the durable fix.
- (D) queue does not carry the session-assumption class today; it inherits the **divergence** class on
  its next feature.

No fixes applied (no broadcast/handler/defaultExpanded/persist/convergence changes).

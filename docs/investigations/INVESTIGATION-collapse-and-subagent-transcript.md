# INVESTIGATION — 2 issues reported after the autonomy/approval ship

**Date:** 2026-05-28
**Branch:** `fix/rotation-by-session-id` (post `01f2121` push)
**Scope:** read-only. No code edits.

---

## Issue 1 — first tool-call capsule cannot be collapsed

### Verdict
**Real bug, introduced by FE-A3.** When the FE-A3 fix added `defaultExpanded: true` to capsules whose opening turn carries reasoning_content, the render-time expanded-state check became:

```tsx
// ChatView.tsx:1939-1942 (current)
const isExpanded =
  derivedStatus === 'running' ||
  item.defaultExpanded === true ||
  expandedCapsules.has(item.capsule_id);
```

The chevron click handler at `:1948-1956` toggles membership in the `expandedCapsules` Set. But because `item.defaultExpanded === true` is OR'd into `isExpanded`, no value of set membership can make `isExpanded` false:

- Initial render: `defaultExpanded=true` → `isExpanded=true`. ✓ shown expanded.
- User clicks chevron → handler adds `capsule_id` to `expandedCapsules` → still expanded.
- User clicks again → handler removes `capsule_id` from `expandedCapsules` → STILL `isExpanded=true` because `defaultExpanded=true` keeps the OR true. **Click is no-op.**

The capsule is permanently locked open even though `isLocked` is false (so the button isn't disabled — it just doesn't do anything).

### Why "first tool call" specifically
A capsule gets `defaultExpanded=true` when its opening assistant turn has `reasoning_content` (`chatTransform.ts:393` — `currentCapsule = openCapsuleAt(msg.timestamp, !!reasoning)`). The management agent typically reasons before the FIRST tool call of a turn (deciding what to do), then often emits subsequent tool batches with thinner reasoning. So the first capsule of any given turn is the most likely to carry reasoning → most likely to be `defaultExpanded=true` → most likely to be the one the user notices is stuck open.

### Fix direction
The cleanest fix tracks the user's *explicit* expand/collapse choice separately from the default, so a click is always meaningful. Three options, ranked:

1. **Initialize the existing `expandedCapsules` Set with every `defaultExpanded` capsule_id when items change, then drop the `defaultExpanded` term from the OR.** Smallest change. The Set becomes "currently-expanded capsules", seeded from `defaultExpanded`. User click toggles set membership — always meaningful.
2. **Add a second `collapsedCapsules` Set for explicit user collapses; effective `isExpanded = running || (defaultExpanded ? !collapsedCapsules.has(id) : expandedCapsules.has(id))`.** Slightly more state but clearer separation of "default vs explicit".
3. **Per-capsule `userOverride: Map<string, boolean>` with `undefined` meaning "no choice → use default".** Cleanest semantics, more state.

Option 1 is enough — the only downside is that a re-seed (e.g. on idle refresh) re-initializes the set, which means an explicitly-collapsed default-expanded capsule would re-expand on the next history refresh. Whether that's acceptable depends on how often defaultExpanded capsules show up; if it's just the first turn's reasoning-bearing capsule, the user almost never goes back to collapse it twice.

**File:line:** `web/src/components/ChatView.tsx:1939-1942` (the OR expression) + `:1948-1956` (the click handler).

---

## Issue 2 — sub-agent transcript not shown the same way as the management agent

### Verdict
**Architectural gap, not a regression.** The data exists; the rendering doesn't.

### What's stored
Sub-agent transcripts live in a separate JSONL per dispatch at
`~/Library/Application Support/Orbital/scratch/orbital/sub_agents/{slug}/{uuid}.jsonl`.
Sample from a recent claude-code run (`60ad64cc.jsonl`, 4 lines):

```
L1: chunk_type=tool_activity  content='[Using tool: Write]'
L2: chunk_type=tool_activity  content='[Using tool: Bash]'
L3: chunk_type=tool_activity  content='[Using tool: Read]'
L4: chunk_type=response       content='The file already exists with the correct primes ...'
```

Each chunk type written by `process_manager.py:48-67`:

```python
async for chunk in adapter.read_stream():
    if chunk.chunk_type == "turn_complete":
        ...; continue
    entry = { "source": handle, "content": chunk.text, "timestamp": ..., "chunk_type": chunk.chunk_type }
    if transcript is not None:
        transcript.append(entry)          # ← every chunk → sub-agent JSONL
```

### What's broadcast to the FE today
From `process_manager.py:70-99`:

- `chunk.chunk_type in ("response", "message", None)` → emits `chat.sub_agent_message` WS event. ChatView renders this as a `sub_agent_message` display item (`chatTransform.ts:323-327` for the persisted case, `ChatView.tsx:1066-1071` for the live case).
- `chunk.chunk_type == "approval_request"` → emits `approval.request` WS event.
- Every chunk passes through `_activity_translator.on_message`, but that translator emits `agent.activity` events only for tool-related chunks under specific conditions, and the activity events it emits live in the **management session's** activity stream, not in a sub-agent-attributed stream the FE can place under the dispatch.

So `tool_activity` chunks like "[Using tool: Write]" are written to the sub-agent JSONL but not broadcast to the FE as renderable items. The FE has no way to render the sub-agent's intermediate tool calls.

### What renders today
For a dispatched sub-agent the user sees:

1. `[Sub-agent] claude-code started` (lifecycle marker, FE-A2's `sub_agent_activity` row).
2. `[Sub-agent] Message sent to claude-code: …` (lifecycle marker).
3. `[Sub-agent] claude-code completed: …Summary…` (lifecycle marker).
4. The sub-agent's response text as a `sub_agent_message` bubble (avatar + text), IF non-empty.

What is missing compared to the management agent:

- The sub-agent's tool-call capsule (would render as "Write, Bash, Read · 4s" with chevron, like the management agent's tool capsules).
- The sub-agent's reasoning blocks.

### Why this matters for the queue / unattended thesis
Without an inline transcript, when the user comes back and finds a queued task is done, they see the management agent's transcript with tool capsules but the sub-agent block is a one-line "completed: …Summary…". If the sub-agent did substantive work (multiple tool calls, intermediate decisions), that work is invisible until the user reads the sub-agent JSONL file by hand. The user's mental model is broken — Orbital's chat is supposed to be the durable record of what happened.

### Fix directions

Three options, each with its own complexity profile.

| Option | Approach | Where the work lands | Trade-off |
|---|---|---|---|
| **A. Backend merges** | The `/api/v2/agents/{pid}/chat` handler reads each persisted `[Sub-agent] started` line, fetches the matching `sub_agents/{slug}/{uuid}.jsonl`, and inlines synthetic `assistant`-shaped rows into the returned message list at the dispatch point. The existing `chatTransform.ts` then produces capsules and reasoning blocks for them with no FE change beyond a sub-agent-attributed source label. | `agents_v2.py` + a new helper for sub-agent transcript reading. ~30-60 lines. | Cleanest UX, no FE rework. Schema for the synthesized rows needs to carry `source=handle` so the renderer can distinguish a sub-agent capsule (different avatar / indentation) from a management capsule. |
| **B. New API + lazy expand** | Add `GET /api/v2/projects/{pid}/sub-agent-transcript?path=...` returning the sub-agent JSONL parsed. FE renders the existing `sub_agent_activity` `completed` row as a clickable disclosure → fetch + expand → render a child capsule list under it. | New route + small new FE component. ~80 lines. | Discoverable, lazy, doesn't bloat the /chat payload. But it's a click for the user every time they want to see what claude-code did, and the queue view (the thesis-defining use case) is the place they're *least* likely to click — they want to scan, not interact. |
| **C. Live broadcast + reload-from-disk** | Broadcast `agent.activity` events tagged with `source=handle` for sub-agent `tool_activity` chunks; FE renders them into a sub-agent-attributed live capsule. On reload, fetch transcripts via Option B's endpoint to backfill the persisted state. | `process_manager.py` (broadcast) + `chatTransform.ts` (parse persisted) + ChatView (route by source) + new endpoint. ~120 lines. | Truest "shown the same way as management agent" — capsules render live as work happens. But it's the biggest patch and the source-attribution work touches multiple files. |

**Recommendation:** Option A. The dispatch already has a deterministic anchor (`[Sub-agent] {handle} started` line + the lifecycle observer knows the transcript_path), the FE renderer already does what's needed once given the rows, and "shown the same way as the management agent" maps cleanly to "same DisplayItem variants" if the backend just emits them. The only piece of UX the spec needs to decide is whether sub-agent capsules render visually distinct from management capsules (indented further? different avatar tone? gutter color?).

### Files touched by each option

- **A:** `agent_os/api/routes/agents_v2.py` (extend `_read_chat_messages*` to interleave sub-agent rows), `agent_os/daemon_v2/sub_agent_transcript.py` (read helper), maybe a tiny chatTransform extension to add a `source !== 'management'` indicator on tool_call_rows.
- **B:** new route in `agents_v2.py`, new component in `web/src/components/`, change `chatTransform.ts` only to make the `sub_agent_activity` completed row carry the transcript path.
- **C:** `process_manager.py` (new broadcast), `types.ts` (extend `ActivityEvent` with source), `chatTransform.ts` + ChatView for routing, plus Option B's endpoint for the cold-load path.

---

## Summary table

| # | Issue | Type | Severity | Fix complexity |
|---|---|---|---|---|
| 1 | `defaultExpanded` capsules cannot be collapsed | Regression from FE-A3 | P1 (the user notices, no workaround) | One-shot — initialize `expandedCapsules` Set from `defaultExpanded`, drop the OR term. ~5 lines. |
| 2 | Sub-agent intermediate tool calls + reasoning never render in chat | Architectural — data stored separately, not exposed | P1 (queue-thesis breaking; the durable record of what happened is incomplete) | Option A (backend merge) is the cheapest fix that matches the user's "same way as management agent" framing. Roughly 30-60 lines + a small DisplayItem tweak. |

Investigation only — no code changed. Recommend deciding Option A vs B vs C for #2 before any implementation.

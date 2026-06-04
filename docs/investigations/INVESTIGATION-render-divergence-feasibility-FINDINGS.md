<!--
Investigation: render-divergence-feasibility (live vs persisted chat render paths)
Method: multi-agent workflow (parallel deep-readers -> inventory + completeness critic ->
        whole-set verdict -> 3 diverse adversarial-lens challengers w/ >=2-of-3 majority flip -> synthesis).
Run: wf_4325572b-40b (initial run hit the session usage limit mid-pipeline; resumed after reset,
     reusing the cached Map/Inventory prefix). Findings independently spot-verified -- see addendum.
Date: 2026-06-03
-->

# Render-Path Divergence — Mergeable vs Irreducible Inventory

## 1. Headline

The two render paths — the **live overlay** (`ChatView.tsx` imperative `setItems` mutators driven by WebSocket deltas) and the **persisted transform** (`chatTransform.ts` pure `transformChatHistory`) — share exactly one contract: the `DisplayItem` union (`chatTransform.ts:15`). Of the **43 concepts inventoried**, 3 are non-divergent controls (already shared components/helpers), and **40 diverge**. The honest split of the divergent set:

- **9 are MERGEABLE** — a single shared producer could eliminate the divergence because both paths hold the same information at render time: `user_message`, `agent_message (visible answer text)`, `reasoning_block`, `agent_run capsule (container shell)`, `approval_card`, `approval resolution`, `compaction markers`, `fetchLatestMessage REST fallback`, plus `approval card late-binding render`. (9 entries -- see the MERGEABLE set below.)
- **30 are IRREDUCIBLE** — they would remain hand-synced no matter how the code is refactored.
- **1 is UNCERTAIN** — `seed / reconciliation` (it is the *mechanism* of the seam itself, not a DisplayItem producer; its mergeability is contingent on resolving the irreducible inputs it reconciles).

The blunt takeaway: **convergence cannot collapse the two paths into one producer.** It can at best unify the ~9 message-body/container concepts whose data is symmetrically present. The remaining 30 split into three structural walls that no refactor removes:

1. **Wire omits data that only persistence carries** — the live `StreamDeltaEvent` / `ActivityEvent` wire structurally lacks the real `tool_call_id`, tool `arguments`, tool *result content*, JSONL *timestamps*, and the `_activity_descriptions` cache. These materialize only after the assistant/tool JSONL lines are written and refetched. (`tool_call_row` pending + result, `tool_call_id pairing semantics`, `toolCallToActivity description`, `timestamp provenance`.)
2. **Whole-turn / cross-session judgements need the full multi-session array** — header-only anchors, never-vanish guards, inner empty-turn markers, `session_separator`, `isHistorical`, and `sub_agent_*` server-interleaved syntheses require seeing complete turns or multiple sessions at once, which the strictly single-session, incremental live overlay never holds.
3. **Live-only ephemeral runtime** — streaming cursor, Thinking…/sub-agent spinners, cancel state, inject errors, claude.md banner, holder tracking, WS routing, running/error status derivation, new_session reset, dedup, agent_notify, refresh_status. These have **no persisted counterpart at all** (single-path concepts), so there is nothing to merge.

The value of this report is precisely in item (1)–(3): the seductive "it's all just DisplayItems, unify the producer" answer is wrong, and the concrete walls are named below.

---

## 2. Phase A — Concept Inventory

`DisplayItem` (`chatTransform.ts:15`) is the only thing both paths share. `StreamDeltaEvent` (live wire) carries `{type, project_id, session_id?, text, reasoning_content?, source, is_final, seq?}` — **no** `tool_call_id`, **no** tool args, **no** result content. `ChatMessage` (persisted JSONL) carries `tool_calls[]`, `tool_call_id`, `reasoning_content`, `_activity_descriptions`, `sub_agent_*`, `session_id`.

| concept | live repr | persisted repr | divergent? | producing code (live) | producing code (persisted) |
|---|---|---|---|---|---|
| user_message | DisplayItem user_message {content, ts, target?}; no isHistorical | same + isHistorical? (retroactive) | yes | ChatView.tsx:1806-1817, 1287-1297 | chatTransform.ts:471-480, 656-662 |
| agent_message (visible answer text) | buffered StreamState→finalized {content, source, ts=now()} on is_final | {content trimmed, source, ts=msg.timestamp, isHistorical?} | yes | ChatView.tsx:1128-1132, 1076-1102; render 2306-2312 | chatTransform.ts:516-524 |
| agent_message (header-only anchor, content-null + machinery) | NOT produced live | {content:'', isHeaderOnly:true} (FE-A3) | yes | (none) | chatTransform.ts:563-571 |
| agent_message (never-vanish guard, empty turn no machinery) | NOT produced live | {content:'', isHeaderOnly:true} placeholder | yes | (none) | chatTransform.ts:538-550 |
| agent_message (empty marker INSIDE open capsule) | NOT produced live | CapsuleChild {content:'', no isHeaderOnly} into capsule.items | yes | (none) | chatTransform.ts:529-537 |
| reasoning_block | CapsuleChild {content accum, ts=now(), turn_id=now()} | CapsuleChild {content trimmed, ts=msg.ts, turn_id=msg.ts, isHistorical?} | yes | ChatView.tsx:1115-1124→appendLiveReasoning 134-173 | chatTransform.ts:582-590 |
| tool_call_row (pending) | {tool_name, target_description=e.description, tool_call_id=e.id SYNTHETIC, category, ts=e.ts, result_content:null, pending} | {tool_name, target_description from args/cache, tool_call_id=tc.id REAL, result_content:null, pending} | yes | ChatView.tsx:1167-1182→appendToLiveCapsule 83-124 | chatTransform.ts:591-606, toolCallToActivity 152-235 |
| tool_call_row (result received/paired) | in-place {result_content:'' EMPTY, received}; POSITIONAL pairing | in-place {result_content:<real string>, received}; pair by REAL id | yes | ChatView.tsx:1157-1160→markLatestLiveCallResultReceived 245-269 | chatTransform.ts:613-636 |
| agent_run capsule (container) | {capsule_id:'cap:live:…', status running→terminal, items[], counts, has_thinking, started_at, ended_at}; no defaultExpanded | {capsule_id:'cap:ms:counter', status ALWAYS completed, items[], counts, ended_at non-null, defaultExpanded only-if-true} | yes | ChatView.tsx:83-124, 175-188, 72-81 | chatTransform.ts:320-346, 315-318, 646 |
| sub_agent_run capsule | NOT produced live (spinner + sub_agent_message only) | agent_run {capsule_id:'sub_agent:…', items from sub_agent_tool_rows, has_thinking:false, defaultExpanded:false} | yes | (none) | chatTransform.ts:363-414 |
| sub_agent run header | NOT produced live | {content:'', source:handle, isHeaderOnly:true} before capsule | yes | (none) | chatTransform.ts:370-376 |
| sub_agent_message (response bubble) | sub_agent_message {content ANSI-strip, source, ts}; empty filtered | sub_agent_message for role='agent' OR agent_message for synthetic source='sub_agent' | yes | ChatView.tsx:1238-1259 | chatTransform.ts:497-505 AND 416-425 |
| approval_card | approvals Map<id,{…,recent_activity POPULATED}>; late-bound render | DisplayItem approval_card {…, recent_activity:[] EMPTY} | yes | ChatView.tsx:1186-1221, 687-733; render 2169-2213, 2228-2279 | chatTransform.ts:432-445, 485-495 |
| approval resolution (approved/denied) | Map entry patched resolved; optimistic on dismiss | resolved from _meta.resolution (system path only) | yes | ChatView.tsx:1223-1235, 1904-1911, 2202-2210 | chatTransform.ts:442 |
| session_separator | NOT produced live (single-session scope) | {type:'session_separator', ts} on session_id change | yes | (none) | chatTransform.ts:463-466 |
| sub_agent_activity (lifecycle markers) | NOT produced live (spinner only) | {action, handle, summary?/preview?/error?} from [Sub-agent] regex | yes | (none) | chatTransform.ts:451-455, parsers 264-298 |
| agent_notify card | DisplayItem agent_notify {title, body, urgency, ts}; Notification fired | NOT produced by transform (union member only) | yes | ChatView.tsx:1301-1327, 947-960 | (none — declared 93-99) |
| refresh_status card | DisplayItem refresh_status {status, trigger, ts}; in-place updates | NOT produced by transform (union member only) | yes | ChatView.tsx:1329-1380 | (none — declared 100-104) |
| capsule expansion/collapse state | Set<capsule_id> expandedCapsules + running force-expand | defaultExpanded boolean (only-if-true; never true for completed) | yes | ChatView.tsx:383, 512-522, 2097-2102, 2075-2077 | chatTransform.ts:343, 412, 578 |
| streaming text in-progress indicator | StreamState{text,source,isComplete:false} → StreamingMessage cursor | none | yes | ChatView.tsx:381, 1128-1132, 1101; render 2306-2312 | (none) |
| thinking indicator (Thinking…) | showThinking bool → Loader2 spinner | none | yes | ChatView.tsx:397, 941, 1127, 922/963; render 2299-2304 | (none) |
| sub-agent loading spinner | subAgentLoading slug → animated dots | none (superseded by sub_agent_activity/run) | yes | ChatView.tsx:396, 1834, 1916; render 2281-2297 | (none) |
| error/stopped terminal status | status='error'\|'stopped' via finalizeLiveCapsule | transform ALWAYS finalizes 'completed' | yes | ChatView.tsx:936-937, 1085; summary 66-68 | chatTransform.ts:320, 646 |
| ordering and grouping (sequencing) | seeded base + live tail; approval Map after loop; stream/spinner last | strict JSONL chronology; boundary finalize; isHistorical retroactive | yes | ChatView.tsx:2025-2225, 2228-2279, 2281-2312, 1418-1473 | chatTransform.ts:348-640, 646, 648-662 |
| WS session routing (strict session_id gating) | every handler drops session_id ≠ sessionIdRef.current | no routing — transform consumes all sessions | yes | ChatView.tsx:1072-1074, 1138-1145, 1188-1193, 1226, 1240-1243, 1261-1285, 1303-1308, 1331-1335 | chatTransform.ts:463-469 |
| running-status derivation (isActivelyRunning) | render-time upgrade completed→running from props | none — transform statically 'completed' | yes | ChatView.tsx:477-480, 2063-2074, 72-81 | chatTransform.ts:642-646 |
| compaction markers (filtered out) | no wire signal (never received) | _compaction=true messages skipped | yes | (none) | chatTransform.ts:351-353 |
| tool_call_id pairing semantics | SYNTHETIC e.id; POSITIONAL pairing | REAL tc.id; by-id backward match | yes | ChatView.tsx:1174, 245-269 | chatTransform.ts:598, 615-630, 395 |
| toolCallToActivity description (+cache) | server e.description one-shot; no args, no cache | toolCallToActivity from name+args+_activity_descriptions+workspace | yes | ChatView.tsx:1173 | chatTransform.ts:152-235, called 593 |
| timestamp provenance | now() for stream-derived; wire ts for tool/sub-agent | real msg.timestamp throughout (tsToMs) | yes | ChatView.tsx:1096/1116/1805, 1176/1255 | chatTransform.ts:310-313 |
| isHistorical flagging | live append omits field | retroactive true before last session_separator | yes | (no live producer) | chatTransform.ts:648-662 |
| seed / reconciliation (persisted→live) | seed effect setItems(historyItems); idle refetch reseed | historyItems memo = transformChatHistory; baseline + overwrite | yes | ChatView.tsx:509-523, 905-937, 650-666, 1823-1832 | ChatView.tsx:486-489 wrapping chatTransform.ts:300-666 |
| new_session reset | clear rawMessages/items → single agent_notify; RESET | none — fresh JSONL yields empty items | yes | ChatView.tsx:947-966 | (none) |
| fetchLatestMessage REST fallback | GET /chat?limit=1 → synthesize closed capsule / agent_message | equivalent data via full transform on refetch | yes | ChatView.tsx:531-629, 925, 1103 | chatTransform.ts:300-666 |
| approval card late-binding render (Map vs embedded) | Map entries rendered unless embedded approval_card exists | embedded approval_card takes precedence | yes | ChatView.tsx:2228-2279, 1195-1206 | chatTransform.ts:432-445, 485-495 |
| cancel affordance (Stop optimistic) | isCancelling bool → spinner; cleared on terminal/timeout | none | yes | ChatView.tsx:426, 2468-2491, 912, 973-980 | (none) |
| inject error message (composer-level) | injectError string → red text; cleared on send | none | yes | ChatView.tsx:401, 1616/1621/1914, 1835/1880; render 2315-2318 | (none) |
| claude.md warning banner | claudemdWarning → banner; dismissable | none | yes | ChatView.tsx:417, 1382-1391, 1993-1998 | (none) |
| slot holder tracking (holderSessionId) | state from /run-status; for isActivelyRunning/cancel/child prop | none | yes | ChatView.tsx:392, 742-760, 1025-1045 | (none) |
| message dedup (nonce+content+optimistic-ts) | nonce/localNoncesRef, content match, optimisticTimestamp | none — transform trusts JSONL verbatim | yes | ChatView.tsx:1275-1279, 587-607, 1823-1832, 1788 | (none) |
| **capsule summary line (tool breakdown + duration)** | capsuleSummaryText over live counts/ended_at | same helper over transform counts/ended_at | **no (control)** | ChatView.tsx:43-70, 2079 (shared) | same ChatView.tsx:43-70 over chatTransform.ts:325-342 |
| **markdown message rendering (bubbles)** | stateless ChatMessage component from DisplayItem | same component over same DisplayItem | **no (control)** | ChatView.tsx:2030-2033 (shared) | same ChatView.tsx:2030-2033 |
| **tool_call_row expand/collapse (renderer)** | ToolCallRow component; ''/null → 'no result content' | same component; real string → truncated | **no (control)** | ChatView.tsx:198-242, ~2127-2133 (shared) | same ChatView.tsx:198-242 |

---

## 3. Phase B — Verdict per concept (information-availability)

### MERGEABLE

- **user_message** — Both hold `{content, timestamp, target?}` at render time (live optimistic/echo wire `UserMessageEvent`; persisted JSONL role='user'). The only delta, `isHistorical`, is a render-time algorithmic decoration (last `session_separator` position) that the single-session live overlay structurally defines as always-false, not as missing data. **Shared producer input:** `{content, timestamp, target?}`; both supply all three.
- **agent_message (visible answer text)** — The body text + `source` are on both wires (live accumulates `StreamDeltaEvent.text` to `is_final`; persisted reads role='assistant' content). Only `timestamp` differs (live synthesizes `now()` because the stream wire carries none); that is a deliberate clock substitute, tracked separately under *timestamp provenance* (IRREDUCIBLE). **Shared producer input:** final text + source (+ a timestamp).
- **reasoning_block** — Reasoning text on both wires (`StreamDeltaEvent.reasoning_content` / `ChatMessage.reasoning_content`). Divergence is only `timestamp`/`turn_id` provenance (a within-capsule key), not content. **Shared producer input:** accumulated reasoning text + a timestamp/turn_id.
- **agent_run capsule (container shell)** — Pure container computed from its children (counts, `has_thinking`, start/end). Given the ordered child list, both paths derive identical container fields. Structural deltas (status, capsule_id scheme, defaultExpanded) are tracked as separate concepts. **Caveat:** the *children themselves* (pending rows, paired results) are individually IRREDUCIBLE; only the shell merges. **Shared producer input:** ordered child list + start/end timestamps.
- **approval_card** — Core fields `{what, tool_name, tool_call_id, tool_args, reasoning?}` present on both (`ApprovalRequestEvent`/REST vs system `_meta.approval_request`/agent chunk). Persisted is actually *weaker* on `recent_activity` (hardcoded `[]`, filled later by the live handler). **Shared producer input:** approval keyed by `tool_call_id` with those fields.
- **approval resolution (approved/denied)** — Both can know the resolution at render time (live `ApprovalResolvedEvent.resolution` + optimistic; persisted `_meta.resolution`), reconciled at render via `approvals.get(id)?.resolved ?? item.resolved`. **Shared producer input:** `tool_call_id` + resolution. *(Caveat: agent-path persisted card has no `resolved`; the live Map supplies it.)*
- **compaction markers** — Outcome is identical on both paths: neither renders anything for a compacted message (persisted reads `_compaction` and skips; live never receives one). The "render nothing" rule is satisfied symmetrically. **Shared producer input:** `_compaction` flag (persisted) / its absence (live) — both lead to "emit nothing."
- **fetchLatestMessage REST fallback** — Fetches the *same* persisted JSONL message via `GET /chat?limit=1` and synthesizes the same DisplayItems the full transform would. It is a single-message shortcut over identical source data, an optimization not an asymmetry. **Shared producer input:** the persisted `ChatMessage` — the transform itself could serve both.
- **approval card late-binding render (Map vs embedded)** — A render-time precedence/dedup rule over the *same* approval identity keyed by `tool_call_id`; a persisted embedded card shadows the live Map entry. Both carry the same identifying fields. **Shared producer input:** approval keyed by `tool_call_id`; divergence is which side holds the entry, not its content.

### IRREDUCIBLE — wire omits data only persistence carries

- **tool_call_row (pending)** — Live has only the `ActivityEvent` (`tool_name`, server description, event id, category, ts). **Missing live:** real `ToolCall.id` and `function.arguments` (+ `_activity_descriptions`). **Available:** only after the assistant JSONL line is persisted+refetched. **Why can't wait:** must render the instant the activity event arrives; the wire omits id and args.
- **tool_call_row (result received/paired)** — Live marks received with EMPTY content, paired POSITIONALLY. **Missing live:** the actual result-content string and the real pairing key. **Available:** only when the role='tool' JSONL line is persisted+refetched and paired by id. **Why can't wait:** the content simply is not on the wire to wait for; live must show "a result arrived" immediately. *Canonical half-paired state.*
- **tool_call_id pairing semantics** — **Missing live:** the real `ToolCall.id` on tool_use events (rides only in `tool_name` for tool_result and is ignored). **Available:** only with persisted `tool_calls[]` + role='tool' lines on refetch. **Why can't wait:** must pair as results stream in; the deterministic id key is off-wire.
- **toolCallToActivity description (+cache)** — **Missing live:** `function.arguments`, the `_activity_descriptions[tc.id]` cache, and workspace context needed to synthesize/restore the arg-based description. **Available:** only after the assistant tool_calls (with args) are refetched. **Why can't wait:** must label the row immediately from the one-shot server `e.description`.
- **timestamp provenance** — **Missing live:** the real JSONL timestamp for stream-derived items (`StreamDeltaEvent` carries none), forcing `now()`. **Available:** only on refetch. **Why can't wait:** must stamp items as they arrive; the render clock ≠ eventual persisted ts until reseed.

### IRREDUCIBLE — whole-turn / cross-session judgements need the full array

- **agent_message (header-only anchor, content-null + machinery)** — **Missing live:** the turn-level fact that a turn has machinery but never emits visible text (`!text && (reasoning||tools) && !currentCapsule`). **Available:** only after the turn completes and is retransformed in one pass. **Why can't wait:** live renders the capsule incrementally and cannot defer the whole turn to learn if text will appear.
- **agent_message (never-vanish guard, empty turn no machinery)** — **Missing live:** any signal at all — a wholly-empty turn fires no delta and no activity. **Available:** only when the empty JSONL line is read by transform. **Why can't wait:** there is no live trigger; the turn produces zero events by definition.
- **agent_message (empty marker INSIDE open capsule)** — **Missing live:** the per-assistant-message boundary signal for an empty-content turn inside an already-open capsule. **Available:** only by walking distinct JSONL lines in order. **Why can't wait:** the live stream collapses multiple turns' machinery into one capsule with no inter-message framing on the wire.
- **session_separator** — **Missing live:** cross-session adjacency, which strict session gating removes from the overlay. **Available:** only in the full multi-session `rawMessages` array. **Why can't wait:** live deliberately scopes to one session and never holds two sessions adjacently.
- **isHistorical flagging** — **Missing live:** the multi-session array and last-`session_separator` index needed to mark prior items historical. **Available:** only in the full transform pass. **Why can't wait:** live is single-session and never sees the boundary defining "historical."
- **sub_agent_run capsule** — **Missing live:** the `sub_agent_tool_rows[]` breakdown (name/duration) and `sub_agent_duration`, computed server-side from the sub-agent transcript post-dispatch. **Available:** only after the /chat endpoint interleaves the synthesized summary into JSONL and it is refetched. **Why can't wait:** live can only show a spinner; the breakdown is not streamed on any wire.
- **sub_agent run header** — **Missing live:** the synthetic source='sub_agent' JSONL line (with handle) that anchors the header. **Available:** only after the server interleaves it and it is refetched. **Why can't wait:** live has only the loading spinner + response bubble; the anchor is a post-dispatch server synthesis.
- **sub_agent_message (response bubble)** — The content is on both wires and the role='agent' sub-case alone would merge, BUT the persisted path also renders the synthetic source='sub_agent' response as a *different type* (`agent_message`, chatTransform.ts:416-425) where live always emits `sub_agent_message`. **Missing live:** the source='sub_agent' framing that decides the type. **Available:** only after server interleave + refetch. **Confidence: medium** (the type divergence is the only barrier; the body itself is symmetric).
- **sub_agent_activity (lifecycle markers)** — **Missing live:** the parsed `[Sub-agent]` markers (action/handle/summary/preview/error), written to JSONL but not delivered as a consumed live event. **Available:** only when the role='system' lines are refetched and regex-parsed. **Why can't wait:** live has only an undifferentiated spinner.

### IRREDUCIBLE — live-only ephemeral runtime (single-path; nothing to merge)

These exist in **only one path**, so there is no shared producer by definition. For most, the *persisted* side is the one lacking data (inverse asymmetry).

- **agent_notify card** — live-only DisplayItem; transform declares the union member but never emits it.
- **refresh_status card** — live-only DisplayItem; transform never emits it.
- **streaming text in-progress indicator** — transient partial buffer; persisted only ever sees final text.
- **thinking indicator (Thinking…)** — driven by live `agentStatus='running'`; transform is status-blind.
- **sub-agent loading spinner** — in-flight dispatch state; persisted shows the completed run instead.
- **error/stopped terminal status** — transform ALWAYS emits 'completed'; only live `agentStatus='error'` can flip it (FE-A1 defers running/error to render time).
- **running-status derivation (isActivelyRunning)** — depends on live `sessionId`/`holderSessionId`/`agentStatus` that the pure transform has no access to.
- **capsule expansion/collapse state** — live `expandedCapsules` Set + render-time running-force-expand; the persisted `defaultExpanded` is effectively always false (locked product decision). The running-expand signal is live runtime; the user-toggle Set is never persisted.
- **WS session routing (strict session_id gating)** — live-only filtering mechanism; the persisted path intentionally does the opposite (retain all + separate).
- **new_session reset** — driven by live `AgentStatusEvent='new_session'`; persisted sees only a fresh empty file with no reset/notice.
- **ordering and grouping (sequencing)** — live composite (seeded base + out-of-band Map approvals + stream/spinner last) vs strict JSONL chronology; several tail elements carry no persisted timestamp. **Confidence: medium** (the container ordering is partly derivable; the overlay positions are not).
- **cancel affordance (Stop optimistic)** — live composer UI state tied to a user action; never persisted.
- **inject error message** — transient composer feedback; never persisted.
- **claude.md warning banner** — live WS-only event; never a persisted DisplayItem.
- **slot holder tracking (holderSessionId)** — live `/run-status` runtime value; no persisted analog.
- **message dedup (nonce+content+optimistic-ts)** — entirely the live overlay-vs-refetch seam; the transform trusts JSONL verbatim and does none.

### UNCERTAIN

- **seed / reconciliation (persisted → live items)** — This is not a DisplayItem producer; it is the *mechanism of the seam itself* (seed effect `setItems(historyItems)` + idle refetch reseed, ChatView.tsx:509-523/905-937). Whether it can be simplified is contingent on first resolving the irreducible inputs it reconciles (the wire-omitted tool data and timestamps that make the live tail differ from the eventual transform output). It cannot be classified MERGEABLE/IRREDUCIBLE in isolation — it is the integration point, not a leaf concept.

---

## 4. Phase C — Convergence prerequisite (feasibility, NOT design)

Before any convergence is *safe*, a **characterization-test harness** must lock the current `DisplayItem` output of both paths so a refactor can prove behavioral equivalence (or deliberately-accepted divergence) rather than silently regress. This section establishes feasibility and fixture availability only — it does **not** design the harness.

**What must be snapshotted** — the full `DisplayItem[]` tree (live overlay state *and* persisted transform output) for representative session shapes:

| fixture scenario | exercises | already covered as assertions? | fixture status |
|---|---|---|---|
| reasoning-only (no answer) | content-null capsule, reasoning_block, FE-A3 header, collapse | yes — chatTransform.test.ts:697-732, 908-933 | **assertion-only; needs full-tree capture** |
| reasoning + answer | answer body + collapsed reasoning capsule | yes — chatTransform.test.ts:934-960 | assertion-only; needs capture |
| multi-tool (parallel + in-flight + paired) | tool_call_row pending/received, by-id vs positional pairing, page-seam | yes — chatTransform.test.ts:219-432; ChatView.test.tsx:1047+ | assertion-only; needs capture |
| sub-agent (peer capsule + lifecycle) | sub_agent_run, sub_agent header, sub_agent_activity, type-divergent bubble | yes (persisted) — chatTransform.test.ts:608-905; **live side only spinner** | **persisted assertions exist; live path has NO sub_agent_run to snapshot — confirms IRREDUCIBLE** |
| error / cancel | error/stopped terminal status, isCancelling | live cancel — ChatView.test.tsx:301-459; **error capsule status not snapshotted** | partial; needs live error-status capture |
| multi-session | session_separator, isHistorical, strict session routing | yes — chatTransform.test.ts:49-67; ChatView.test.tsx:529-720 (leak test) | assertion-only; needs full-tree capture |
| approval (request + resolution) | approval_card, resolution, Map-vs-embedded late-binding | live pending-approval stubbed off (ChatView.test.tsx:53-54) | **not covered; needs capturing both sides** |
| empty/never-vanish + compaction | placeholder guards, _compaction skip | yes — chatTransform.test.ts:961-984, 304-324 | assertion-only; needs capture |

**Fixture-availability finding (verified against `tests/`, `web/src`):**

- **No snapshot or characterization fixtures exist.** `find web -name '*.snap'` returns nothing; `grep toMatchSnapshot` returns nothing. Both `web/src/utils/chatTransform.test.ts` (992 LOC) and `web/src/components/ChatView.test.tsx` (1124 LOC) are **assertion-based** unit tests, not output snapshots.
- **Persisted-side message shapes already exist as inline builders** — `chatTransform.test.ts` constructs `ChatMessage[]` via local helpers `user()`, `asst()`, `tc()`, `tool()` (lines ~26-48). These cover reasoning, capsules, paired/in-flight/parallel results, page seams, sub-agent lifecycle, content-null headers, never-vanish, and session boundaries. They are reusable *inputs* but assert individual fields, not the whole tree.
- **Live-side event scenarios already exist inline** in `ChatView.test.tsx` — stream-delta routing, live activity capsule, multi-session leak, cancel, trailing-spinner running state, expand-by-default, transform-once page seam. Again assertion-based, not tree snapshots.
- `tests/fixtures/` contains only Python ACP/pipe-agent dummies (`dummy_acp_agent.py`, etc.) — **irrelevant** to frontend render-path characterization.

**Plausibly reusable (the message-builder inputs) vs needs capturing (the full-tree golden outputs):** the persisted *inputs* are largely present and can seed golden snapshots cheaply; the **live overlay output trees and the live↔persisted equivalence pairs must be captured fresh**, because no test currently asserts the *entire* `DisplayItem[]` produced by either path, and the approval-card live path is stubbed off entirely. The sub-agent and error/cancel live scenarios will, when captured, *document* irreducibility (the live side has no `sub_agent_run`/error-status to snapshot) rather than reveal mergeability.

---

## 5. DO-NOT-overclaim note

The convenient answer — "both paths just emit `DisplayItem`s, so unify them behind one producer" — is **wrong**, and naming why is the point of this report. Even after merging the ~8 message-body/container concepts, a refactor hits these concrete irreducible walls:

1. **The live wire structurally omits data.** `StreamDeltaEvent` carries no timestamp, no `tool_call_id`, no tool args, no result content; `ActivityEvent` carries a synthetic event id and a one-shot server description. The real `tool_call_id`, `function.arguments`, result-content strings, `_activity_descriptions` cache, and JSONL timestamps **materialize only after persistence + refetch.** No producer unification creates data that is not on the wire. (`tool_call_row` pending+result, `tool_call_id pairing`, `toolCallToActivity description`, `timestamp provenance`.)

2. **Whole-turn and cross-session facts are unknowable incrementally.** Header-only anchors, the never-vanish guard, inner empty-turn delimiters, `session_separator`, and `isHistorical` are negative or boundary judgements that require seeing a *complete* turn or *multiple* sessions in one pass. The strictly single-session, append-as-it-streams live overlay never holds that context.

3. **Sub-agent breakdown is a server-side post-dispatch synthesis.** `sub_agent_run`, its header, lifecycle markers, and the type-divergent response bubble are interleaved into JSONL by the /chat endpoint *after* the run completes. Live only ever has a spinner + a `SubAgentMessageEvent`.

4. **Sixteen concepts are single-path live runtime** (streaming cursor, Thinking…/sub-agent spinners, cancel, inject error, claude.md banner, holder tracking, WS routing, running/error status derivation, new_session reset, dedup, agent_notify, refresh_status, expansion-by-running). They have **no persisted counterpart whatsoever** — there is nothing on the other side to merge with, and forcing them through a shared producer would invent persisted state that does not and should not exist.

5. **One concept (`seed / reconciliation`) is the seam itself**, not a leaf — its simplification is downstream of, and gated by, resolving walls (1)–(2).

A convergence can therefore eliminate **at most the message-body and container shells (the MERGEABLE set)**; everything in the IRREDUCIBLE set must remain hand-synced between the two paths by deliberate, tested intent — and the characterization harness in Phase C is the precondition that makes that intent enforceable rather than accidental.


---

## 6. Verification addendum (independent spot-check by the coordinator)

The synthesis above was produced by sub-agents; the following claims were re-verified directly
against the source before adopting the report:

- **Transform always finalizes `'completed'`** -- every `finalizeCapsule()` call in
  `chatTransform.ts` (364, 433, 453, 464, 472, 484, 518, 543) uses the default, and 646 passes
  `'completed'` explicitly; the function *can* take `'error'|'stopped'` but the transform never
  does. Confirms `error/stopped terminal status` is **live-only / IRREDUCIBLE**.
- **Live tool result is content-empty + positionally paired** -- `markLatestLiveCallResultReceived`
  (`ChatView.tsx:245-269`) reverse-scans for the most-recent `pending` row and sets
  `result_content: ''`. Confirms the canonical half-paired live state and the positional-vs-by-id wall.
- **`agent_notify` / `refresh_status` are declared-but-never-emitted by the transform** -- they
  appear in `chatTransform.ts` only as `DisplayItem` union members (lines 94, 101), with no
  producer in `transformChatHistory`. live-only.
- **Adversarial challengers genuinely engaged** -- the three lenses returned 7 / 4 / 5 refutations
  (tool-data / sub-agent-data / identity-ordering). "0 flipped" is *earned*, not a rubber-stamp,
  with one exception worth flagging (below).

### Corrections applied to the sub-agent draft
- **Headline counts** were wrong in the draft prose (38 / 8 / 26); corrected to match the
  structured tally: **43 concepts = 3 controls + 40 divergent (9 MERGEABLE + 30 IRREDUCIBLE +
  1 UNCERTAIN)**.
- **`DisplayItem` file citation** corrected from `types.ts` to **`chatTransform.ts:15`** (the union
  is defined there; `types.ts` holds `ChatMessage`, `StreamDeltaEvent`, and the WS event types).

### The one genuinely contested verdict
- **`agent_run capsule (container shell)` -- MERGEABLE is contested.** The *identity-ordering*
  challenger argued it should be IRREDUCIBLE because the container's derived fields
  (`tool_call_count_by_name`, `has_thinking`, `ended_at`, deterministic `capsule_id`) are only
  fully knowable once the turn closes, and the live capsule's children are themselves irreducible.
  Only 1 of 3 lenses raised this, so the majority rule kept it MERGEABLE -- but the verdict already
  carries the caveat that **only the shell merges; the children do not.** Treat this as
  "shell-mergeable, contents-irreducible," i.e. the weakest MERGEABLE claim in the set. If a future
  convergence starts here, expect to hit the children-irreducibility wall immediately.

### Tally note
The deterministic verdict pass scored **31 IRREDUCIBLE / 0 UNCERTAIN**; the synthesis reclassified
`seed / reconciliation` to **UNCERTAIN** (30 IRREDUCIBLE / 1 UNCERTAIN) on the grounds that it is
the seam *mechanism*, not a leaf `DisplayItem` producer. That reclassification is sound and is the
only delta between the machine tally and the narrative.

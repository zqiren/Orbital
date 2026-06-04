# INVESTIGATION — Frontend chat rendering defects (Orbital-marketing sessions)

**Date:** 2026-05-27
**Project examined:** Orbital-marketing (`proj_e1164e981494`), workspace `/Users/keanezhou/Desktop/orbital-marketing`
**Sessions dir:** `orbital/sessions/` — 19 `*.jsonl` files
**Method:** read-only. Raw JSONL parse + live daemon read API (`GET /api/v2/agents/{pid}/chat`, the exact path the frontend uses) + static read of `web/src/utils/chatTransform.ts` and `web/src/components/ChatView.tsx`.
**Status:** investigation only — **no code changes made**.

---

## TL;DR

The session JSONLs are **fully intact** (0 corrupted lines; the backend chat endpoint returns 100% of every message — verified `api_returned == disk_lines` for all sampled sessions). Every "truncated / incomplete / stuck" symptom the user observed originates in the **frontend render pipeline** (`chatTransform.ts` + `ChatView.tsx`), with two symptoms rooted partly in upstream **data** (a missing user message persisted by an agent-start path).

Three genuine frontend defects, one by-design-but-confusing behavior, and two upstream data issues:

| ID | Layer | Defect | Affected (of 19) |
|----|-------|--------|------------------|
| **FE-1** | transform + pagination | Per-page transform **drops tool results at page boundaries** and strands boundary tool-calls as "pending" | sessions >50 msgs (6) |
| **FE-2** | transform + render | `content:null` (tool-only) assistant turns render as **bare collapsed capsules**; reasoning hidden → looks like "only a tool call" | all sessions w/ tool-only turns; most visible in 3 |
| **FE-3** | transform + render | **Trailing capsule stuck `running` (perpetual spinner)** on idle/historical sessions | 3 (`f8888c87`, `e141c85a`, `e4bcf4e3`) |
| **FE-4** | pagination (by design) | Sessions >50 msgs open showing only the **last 50**; rest behind a manual "Load earlier" button | 6 |
| DATA-1 | persistence | Originating **user message never written** to JSONL → session opens headless | 3 (`4fc5a5b2`, `36d35287`, `e4bcf4e3`) |
| DATA-2 | session id resolution | 5 files share F1 `session_id:"default"`; `_find_session_uuid_on_disk` is **many-to-one** | 5 (latent) |

---

## Pipeline reference

```
JSONL on disk
  → GET /api/v2/agents/{pid}/chat?limit=50[&offset=N][&session_id=F1]   (agents_v2.py:1449)
      _read_chat_messages_single / _read_chat_messages  — NO role/type filtering; tail pagination by raw line index
  → transformChatHistory(messages, workspace)   (chatTransform.ts:211)  — runs PER PAGE
  → reconcileTrailingRunning(...)               (ChatView.tsx:227)
  → render: user_message / agent_message / agent_run capsule / session_separator   (ChatView.tsx:1735+)
```

---

## FE-1 — Per-page transform drops boundary tool results; strands pending calls

**Symptom:** in long sessions, after "Load earlier messages", the seam between pages is missing tool results and/or shows a tool call that never resolves.

**Root cause:** `transformChatHistory` is invoked **once per page** — `ChatView.tsx:607` (initial 50) and `ChatView.tsx:648` (each older page) — and pages are stitched as already-transformed `DisplayItem[]` (`ChatView.tsx:652`, `[...transformed, ...prev]`), *not* by concatenating raw messages and transforming once. The backend slices by **raw JSONL line index** (`_read_chat_messages_single`, `agents_v2.py:1373`), so a page boundary falls mid-turn. The transform's tool-pairing rule then misfires across the seam:
- A `tool` result whose `assistant` tool-call is in the **adjacent** page has no open capsule / no matching `tool_call_row` → **dropped** (`chatTransform.ts:416-417`, "Orphan tool results … are dropped").
- An `assistant` tool-call at the end of a page whose `tool` result is in the next page → `tool_call_row` stuck at `result_status:'pending'` and its capsule never merges with the next page's items.

**Evidence** (replicated the orphan rule on the real first page of each >50-msg session):

| session | msgs | tool results dropped on first page |
|---|---|---|
| `f8888c87` | 289 | 2 / 34 |
| `9211d2b1` | 189 | 1 / 25 |
| `4b7af7e7` | 124 | 1 / 27 |
| `e141c85a` | 114 | 2 / 24 |
| `7e4b8db4` | 59 | 2 / 31 |
| `943fc93e` | 123 | 0 / 24 |

Modest per seam (1–2), but compounds with each "Load earlier" click.

**Fix direction:** accumulate raw messages across pages and run `transformChatHistory` on the **full concatenated list**, or merge capsules across the seam. Transform-once is the robust cut.

---

## FE-2 — Tool-only (`content:null`) assistant turns render as bare collapsed capsules

**Symptom (user words):** "the frontend shows only the tool call when scroll to the top, which is not a complete turn." Reported on `4fc5a5b2`; user noted "many other sessions share the same problem."

**Root cause:** when the model emits only tool calls + reasoning and no user-visible text, the message is `content:null`. In the transform, `const text = msg.content && msg.content.trim() ? msg.content : null` (`chatTransform.ts:330`) → no `agent_message` bubble; the turn becomes an `agent_run` capsule holding a `reasoning_block` + `tool_call_row`s. `ChatView.tsx:1752-1789` renders that capsule **collapsed by default** (one-line summary + chevron); the `reasoning_content` — which carries the turn's intent — is hidden behind the expand toggle. So a turn whose entire substance is reasoning+tools shows as a bare "tool call" line. This is why **many** sessions exhibit it: *every* tool-only turn renders this way; it is merely most jarring at the top of a session (no preceding context).

**Evidence:** `4fc5a5b2` first message = `assistant, content=null, reasoning_content=1055 chars, tool_calls=[read,read,read]`. Three sessions *start* with such a turn: `4fc5a5b2`, `36d35287`, `e4bcf4e3`.

**Fix direction:** for `content:null` assistant turns, surface a `reasoning_content` preview inline (or render an explicit "thinking" line) instead of collapsing the whole turn into a tool-only capsule.

---

## FE-3 — Trailing capsule stuck `running` (perpetual spinner) on idle sessions

**Symptom (user words):** "the tool call seems to be still running and never return a result. the loading signal is still there at the agent message spinning." Reported on the `f…` session ~34 days old → **`f8888c87`** (Apr 22, 289 msgs).

**Root cause (two frontend pieces):**
1. `transformChatHistory` finalizes the trailing **open** capsule as `status:'running'` unconditionally at end-of-stream (`chatTransform.ts:426`, `finalizeCapsule('running')`), with no knowledge of whether the agent is actually live. A capsule only closes on a *visible-text* `assistant` message or a `user` message; `system` messages (e.g. the ping-pong guard) and `tool` results do **not** close it. So a session whose final turn ends on tool activity leaves a trailing `running` capsule → spinner (`Loader2`, `ChatView.tsx:1785-1787`) + locked-expanded.
2. The reconciliation that would downgrade it to `completed` (`ChatView.tsx:697-710`, on `agentStatus==='idle'` → `finalizeLiveCapsule(prev,'completed')`) is gated behind **`if (!viewing) return`** (`ChatView.tsx:707`), where `viewing` = "viewed session is the live slot holder." A historical/idle session being browsed is **not** the holder → finalize skipped → the `running` capsule persists.

**Evidence:** `f8888c87` ends:
```
286: assistant content=null  tool_calls=[agent_message → claude-code]
287: tool      (result present — "unknown")
288: system    "Repetitive action detected. Save your state and try a different approach."
```
All tool calls returned (`unanswered_tool_calls = 0`) — so it is the **capsule** that is stuck "running", not a literally-missing result. The agent hit the repetition guard and the turn ended without a closing assistant text.

**Blast radius** — sessions whose final turn ends with an open capsule (= spinner on load): **3 / 19**:

| session | ends with | reason capsule stays open |
|---|---|---|
| `f8888c87` | `system` "Repetitive action detected" | ping-pong guard ended turn mid-dispatch |
| `e141c85a` | `tool` result | no closing assistant text after last tool |
| `e4bcf4e3` | `system` sub-agent notice | tool-only turn, never closed |

**Fix direction:** the trailing capsule should be `running` only when the viewed session is **actively running**. Cleanest: pass `isActivelyRunning = viewingHolder && agentStatus==='running'` into `transformChatHistory`; `finalizeCapsule` uses `'running'` only when true, else `'completed'`. (Alternative: move `finalizeLiveCapsule(prev,'completed')` above the `if (!viewing) return` gate so it runs for any viewed idle session — but the load-time flag also covers the first render before any status effect fires.)

---

## FE-4 — 50-message pagination looks like truncation (by design)

**Symptom:** long sessions open showing only their tail.

**Mechanism:** `CHAT_PAGE_SIZE = 50` (`ChatView.tsx:238`); initial fetch is `?limit=50` = most-recent 50. Older history sits behind a **manual** "Load earlier messages (N more)" button (`ChatView.tsx:1717-1723`) — not auto-loaded on scroll, easy to miss. 6 sessions exceed 50 msgs (f8888c87=289, 9211d2b1=189, 4b7af7e7=124, 943fc93e=123, e141c85a=114, 7e4b8db4=59).

**Note:** This is the symptom the user explicitly said was **not** their concern, but it is real and worth resolving alongside FE-1 (auto-load on scroll, or raise the page size). Fixing FE-1 is a prerequisite for paging to be correct.

---

## Upstream data findings (not frontend, but surface as frontend symptoms)

### DATA-1 — Originating user message never persisted (headless sessions)
3 sessions (`4fc5a5b2`, `36d35287`, `e4bcf4e3`) contain **zero `user` messages**, yet the first assistant turn's reasoning says *"The user is asking me to…"*. The lost prompt is **not in any sibling session file** (searched by timestamp) — it was never written to disk. `loop.run()` *does* append `initial_message` and queued messages as `role:"user"` (`agent_os/agent/loop.py:255-263, 303-310`), so these sessions were created by a path that fed the instruction into the model context **without** a session append. All three are the legacy `session_id:"default"` "011 Xiaohongshu/Transformer article" sessions (created May 25–26) — consistent with the now-removed frontend auto-start or a related orchestration/fork path.
**This is why FE-2's symptom is worst on these three** — they open headless *and* with a tool-only first turn.
**Follow-up:** audit every agent-start path (auto-start remnants, triggers, orchestration resume) to guarantee the initiating instruction is appended as a `user` message.

### DATA-2 — `session_id:"default"` collision (latent)
5 files carry internal F1 `session_id:"default"` (`36d35287`, `4fc5a5b2`, `8345e00b`, `9211d2b1`, `e4bcf4e3`). `_find_session_uuid_on_disk` (`agents_v2.py:1410`) is **many-to-one** — `?session_id=default` returns the *first* matching file (confirmed: returns `36d35287`'s 41 msgs). Currently masked because the sidebar addresses idle sessions by **F2 uuid**, but if any "default" session becomes the active holder, `list_sessions` reports its F1 as `"default"` and the chat fetch resolves to the wrong file → would read as severe truncation/session-swap.
**Follow-up:** back-fill F1 to F2 uuid for these files, or make resolution uuid-first.

---

## Recommended priority

1. **FE-3** (stuck spinner) — clear bug, isolated fix, high user confusion. *(load-time `isActivelyRunning` flag)*
2. **FE-1** (transform-once across pages) — correctness; fixes dropped tool results and stranded pending rows; prerequisite for safe paging.
3. **FE-2** (surface reasoning for `content:null` turns) — broad UX improvement; the most widely-felt "only a tool call" symptom.
4. **FE-4** (auto-load / larger page) — pairs with FE-1.
5. **DATA-1 / DATA-2** — backend audits; prevent new headless/colliding sessions (does not repair existing data).

---

## Appendix — session inventory

| session | lines | bytes | first msg | ends_open (spinner) | internal F1 | >50 |
|---|---|---|---|---|---|---|
| 0d089502 | 58 | 279 KB | user | no | uuid | yes |
| 0dbe1fc2 | 6 | 2.7 KB | user | no | uuid | |
| 2599e313 | 13 | 17 KB | user | no | uuid | |
| 272f5780 | 3 | 0.7 KB | user | no | uuid | |
| 36d35287 | 41 | 38 KB | **asst tool-only** | no | **default** | |
| 48abbe19 | 5 | 4.6 KB | user | no | sess_cc00ed20 | |
| 4b7af7e7 | 124 | 157 KB | user | no | uuid | yes |
| 4fc5a5b2 | 24 | 33 KB | **asst tool-only** | no | **default** | |
| 7e4b8db4 | 59 | 62 KB | user | no | uuid | yes |
| 8345e00b | 7 | 12 KB | asst+text | no | **default** | |
| 9211d2b1 | 189 | 224 KB | user | no | **default** | yes |
| 943fc93e | 123 | 72 KB | user | no | uuid | yes |
| 9d3b050e | 11 | 7.5 KB | user | no | uuid | |
| ca6bd59c | 46 | 39 KB | user | no | uuid | |
| e141c85a | 114 | 141 KB | user | **YES** | uuid | yes |
| e4bcf4e3 | 4 | 2.7 KB | **asst tool-only** | **YES** | **default** | |
| ed8c29b0 | 19 | 18 KB | user | no | uuid | |
| f119a576 | 6 | 1.9 KB | user | no | uuid | |
| f8888c87 | 289 | 222 KB | asst+text | **YES** | uuid | yes |

All 19 parse cleanly (0 corrupted lines); backend returns 100% of each. Data integrity is not in question — all defects are render-layer (FE-*) or persistence-path (DATA-*).

<!--
Investigation: "final answer dropped in management render" (rows 4 & 42 of orbital-marketing_6c0e9cfb)
Method: ran the REAL transformChatHistory (HEAD + uncommitted) over the real file, AND rendered the
        real ChatView component (jsdom) with the real 43 rows; asserted on captured DisplayItem[] and DOM.
Scope: read-only. No fixes. All probe files deleted after capture; source unchanged.
Date: 2026-06-03
-->

# Finding — the persisted transform does NOT drop rows 4 and 42 (premise refuted)

## Headline (grounded in captured output, not code-reading)
Running the actual code over the actual file disproves the task's central hypothesis. Rows 4 and 42
**are** emitted as visible, top-level answer bubbles — by the transform, and by the full ChatView
DOM render. The drop the user observed is **not in the persisted transform/render pipeline**. This is
the **third** layer this bug has been (mis)attributed to: splitter → transform; both refuted by data.

## Phase A — captured `DisplayItem[]` output (the disproof)
Fed the exact 43 rows of `orbital-marketing_6c0e9cfb.jsonl` into `transformChatHistory(...)` and
captured the output. Result: **20 DisplayItems**, with the two answers present as non-header-only
`agent_message` bubbles:

```
[ 0] user_message      "what is the state of our current content bank?"   (query 1)
[ 1] agent_message     "\n[STATUS: Checking content bank state]"          (row 1 status text)
[ 2] agent_run         children=[reasoning_block, tool:read, tool:shell]  (row 1 tools)
[ 3] agent_message     "\n# Content Bank Status — 2026-06-03 ..."         <-- ROW 4 ANSWER, VISIBLE
[ 4] user_message      "005 and 002 are already sent. please update it"   (query 2)
 ...                   (status bubbles + capsules for rows 6..41)
[18] agent_run         children=[... tool:checkpoint_state]               (checkpoint capsule)
[19] agent_message     "\nState synced. Summary of what just changed: ..."<-- ROW 42 ANSWER, VISIBLE
```

Explicit assertions on the output:
- row 4 answer ("# Content Bank Status") visible top-level bubble? **true**
- row 42 answer ("State synced. Summary") visible top-level bubble? **true**
- either answer nested/absorbed inside a capsule child? **false** (both are standalone bubbles)

**Responsible line:** none — the answer turns take the intended path. Row 4 / row 42 are
`role:'assistant'`, non-empty `content`, no `reasoning_content`, no `tool_calls`, so they hit
`chatTransform.ts:516` `if (text) { finalizeCapsule(); items.push({type:'agent_message', ...}) }`.
The capsule-absorption / branch-miss / boundary-timing hypotheses are all **refuted** by the output:
the answer is emitted as a standalone top-level bubble, not absorbed, not header-only, not dropped.

### End-to-end DOM render (stronger disproof)
Mounted the real `ChatView` (jsdom) with the real 43 rows as the `/chat` response (idle session,
fresh load) and read the rendered DOM:
- "Content Bank Status" (row 4 answer) in DOM? **true**
- "State synced" (row 42 answer) in DOM? **true**
- both user queries + a `[STATUS:…]` turn + **8** `agent_run` capsules in DOM? **true**

So the whole persisted path `/chat` (passthrough — `_read_chat_messages_single` is pagination-only;
this session has no `[Sub-agent]` markers to interleave) → `transformChatHistory` → `ChatMessage` →
`MarkdownContent` (`react-markdown` + `remark-gfm`) renders **both** answers. There is no
content-based filtering anywhere (`grep '[STATUS'` across `web/src` → no matches).

## Phase B — management-source path audit
The transform branches on `source` in exactly **one** place: `chatTransform.ts:363`
`if (msg.source === 'sub_agent')`. There is **no** `source === 'management'` branch — management rows
fall through to the generic role handlers (`role:'assistant'` → the `if (text)` answer path, same as
any normal assistant). Verdict: the bug is **not management-specific**, and is **not present in the
transform at all** for either source. A pure-content terminal answer turn renders identically whether
`source` is `management` or `assistant`.

## Phase C — reasoning-seam fix: CLEAN (re: this bug)
Ran the probe against **both** `HEAD` (committed) and the **uncommitted** `chatTransform.ts`:
- HEAD: row 4 visible? **true**; row 42 visible? **true**.
- Uncommitted: row 4 visible? **true**; row 42 visible? **true**.

Identical. The reasoning-seam changes do not touch the answer path:
- never-vanish guard (`chatTransform.ts:538-551`) fires **only** for empty `content` AND no reasoning
  AND no tool_calls — rows 4/42 have full `content`, so it never runs for them.
- collapse change (`openCapsuleAt(ts, false)`, `chatTransform.ts:572-581`) only affects a capsule's
  expand state — it never gates whether a top-level `agent_message` answer is emitted.

Verdict: the reasoning-seam fix is **CLEAN** with respect to dropping rows 4/42. It is **not** a
contributing cause.

## Root cause (revised) — the drop is NOT in the persisted render
Since the persisted pipeline provably renders both answers, the drop the user saw must come from one
of (read-only investigation cannot yet distinguish — needs the running app):
1. **Live render-state in `ChatView` `items`** during/after an active turn — i.e. the live overlay or
   the running→idle reseed transiently producing an `items` array missing the answers (the related
   thread: `INVESTIGATION-agent-response-vanishes-after-completion-FINDINGS.md` — note its
   empty-content/reasoning theory is **also refuted** here, since rows 4/42 carry real content).
2. **A stale build / screenshot** predating current code.

The static history (this exact file) is NOT a reproduction of the bug — it renders correctly. So a
static-history regression test over this session would **pass already** and cannot capture the bug.

## Fix prep / next-round TASK direction
- **Do NOT touch `chatTransform.ts`** (proven innocent for this bug). Do not touch the splitter
  (already refuted last round). Do not change the reasoning-capsule collapse decision.
- The next TASK must target **live render-state**: capture `ChatView`'s `items` array in the RUNNING
  app at the moment an answer vanishes (temporary `setItems` instrumentation or React DevTools),
  reproducing the live WS-event sequence (stream deltas + activity + `is_final` + running→idle
  reseed). The WS delta stream is NOT in the JSONL, so the bug cannot be reproduced from disk alone —
  this is why a captured live `items` trace is the required artifact.
- **Regression fixture:** must be a **live-event sequence** (mock WS deltas through ChatView's
  handlers, then idle), NOT a static-history transform fixture — the static transform already renders
  rows 4 and 42 (verified). The acceptance bar: after a streamed management turn that ends in a
  pure-content answer, the answer remains a visible bubble through the running→idle reseed.

## Methodology note (per the skepticism mandate)
Every claim above is grounded in captured output for THIS file: (1) `transformChatHistory` run over
the real 43 rows on HEAD and uncommitted; (2) the real `ChatView` rendered to DOM with the real rows.
"The answer renders fine" was **proven** (not inferred): it appears in both the `DisplayItem[]` and
the DOM. The probe files were throwaway and have been deleted; no source was modified.

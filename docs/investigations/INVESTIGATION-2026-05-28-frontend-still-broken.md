# Frontend Investigation — 2026-05-28 (post-fix retest, still broken)

Scope: F1–F4 in `docs/investigations/USER-BUG-REPORT-2026-05-28-still-broken.md`.
Read-only investigation. No code edits.

---

## F1 — "The instant I send the message, the message is invisible on the chat interface."

### What `handleSend` actually does (optimistic append works correctly)

`web/src/components/ChatView.tsx:1547-1601` — the optimistic append fires
synchronously *before* the network round-trip, and uses `setItems(...)`
directly on the viewed conversation:

```tsx
// ChatView.tsx:1583-1600
setInputText('');
if (textareaRef.current) {
  textareaRef.current.style.height = 'auto';
}
clearAttachments();

setItems((prev) => {
  const afterCapsule = finalizeLiveCapsule(prev, 'completed');
  return [
    ...afterCapsule,
    {
      type: 'user_message',
      content: wireContent,
      timestamp: new Date().toISOString(),
      ...(target && { target }),
    },
  ];
});
```

This pre-network append SHOULD make the bubble visible immediately. So the
"invisible the instant I send" symptom can only come from `items` being
overwritten or filtered out after this synchronous setState.

### The actual overwrite path — the seed effect

`ChatView.tsx:435-453`:

```tsx
const historyItems = useMemo(
  () => transformChatHistory(rawMessages, project.workspace, isActivelyRunning),
  [rawMessages, project.workspace, isActivelyRunning],
);

// Seed effect — stomps `items` whenever historyItems changes
useEffect(() => {
  if (historyItems.length === 0) return;
  setItems(historyItems);
}, [historyItems]);
```

`isActivelyRunning` is `viewingHolder && agentStatus === 'running'`
(line 429). When the user sends a message, `agentStatus` flips
`idle → running` (status-override CustomEvent from the 5s poll, or WS
status broadcast from the loop starting). That re-evaluates the memo,
which re-seeds `items` from `rawMessages` — and `rawMessages` at this
moment does NOT yet contain the just-typed user message (it was only
appended to `items` directly, not to `rawMessages`). The optimistic
bubble is therefore wiped on the very next status flip.

**Repro path (without DevTools):**

1. User types, hits Enter.
2. `handleSend` appends user_message to `items` (visible).
3. `injectMessage` POST returns; loop transitions `idle → running`.
4. `agentStatus` updates, `isActivelyRunning` flips `false → true`.
5. `historyItems` memo re-runs against the unchanged `rawMessages` (the
   user message is not in raw history yet — JSONL writes are async on
   the backend, and even after they land the FE has not refetched).
6. Seed effect runs `setItems(historyItems)` and the optimistic bubble
   disappears.

### Why F1 has not been visible to the maintainer before this fix

Until commit `7d08d1b` introduced the `isActivelyRunning` flag as an
*input* to the transform-once memo, status transitions did not
invalidate the `historyItems` memo — only `rawMessages` did. With
`isActivelyRunning` now in the dep array, every status flip re-seeds
`items` from raw history and loses the optimistic tail. F1 is a direct
side-effect of the FE-1/FE-3 transform-once refactor.

### What needs to change (one line)

The seed effect at `ChatView.tsx:450-453` needs to either:

- Append-merge rather than overwrite (preserve any live-tail items beyond the historical baseline), OR
- Push the optimistic user_message into `rawMessages` (so the memo re-run reproduces it), OR
- Remove `isActivelyRunning` from the memo's dep list and recompute the trailing-capsule status separately.

Any of the three resolves F1. Push-into-rawMessages is the smallest diff
and aligns the optimistic state with the same source the memo reads.

### Shared root cause with backend SessionKey? — NO, but adjacent

The `session_id` filter on `/api/v2/agents/{pid}/chat` is implemented
correctly (agents_v2.py:1498-1559). When the user POSTs `/inject` from
a freshly-minted session (one created via `/new-session` whose id is
held only on the route), the backend lazy-mints a handle under
`(project_id, route.sessionId)` — see `inject_message` in
`agent_os/daemon_v2/agent_manager.py:969-1024`. So the persisted
message ends up under the SAME session_id the frontend filters by; the
chat-list scope is consistent. F1 is purely the FE seed effect
clobbering the optimistic tail — it would still reproduce on an
existing, established session.

---

## F2 — `›` glyph between attachment button and text input

`web/src/components/ChatView.tsx:2178`:

```tsx
<button ...>
  <Plus size={18} />
</button>
<span className="shrink-0 font-mono text-secondary select-none" aria-hidden>›</span>
<textarea ...>
```

The literal `›` (U+203A SINGLE RIGHT-POINTING ANGLE QUOTATION MARK) is
inserted as a separator between the `+` attachment button and the
textarea. Delete the entire `<span>` element to fix.

---

## F3 — `⌘↩` command-enter glyph on right of input bars (both chat AND queue)

### Chat composer

`web/src/components/ChatView.tsx:2259`:

```tsx
<kbd className="shrink-0 px-1 py-0.5 border border-border rounded-[3px] text-[9.5px] font-mono bg-sidebar text-secondary select-none max-md:hidden" aria-hidden>⌘↩</kbd>
```

### Queue composer

`web/src/components/QueueComposer.tsx:61-63`:

```tsx
<kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded border border-border bg-sidebar text-[10px] font-mono text-secondary shrink-0">
  ⌘↩
</kbd>
```

Both `<kbd>` elements need to be removed.

Note: the `⌘↩` label is also misleading — the keydown handlers at
`ChatView.tsx:1726-1729` and `QueueComposer.tsx:40-45` send on plain
`Enter` (without Shift), not Cmd+Enter. The label was always inaccurate
in addition to being unwanted.

---

## F4 — SlotHeldNotice style mismatch

`web/src/components/SlotHeldNotice.tsx:60-107` (the whole component) uses
a visual language that does not match the rest of the app:

```tsx
// SlotHeldNotice.tsx:61-66 — container
<div
  data-testid="slot-held-notice"
  className="mb-3 rounded-lg border border-secondary/30 bg-secondary/5 px-4 py-3"
  role="status"
  ...
>
```

```tsx
// SlotHeldNotice.tsx:86-105 — buttons
<button
  ...
  className="rounded-md bg-accent/20 px-3 py-1.5 text-sm font-medium text-primary hover:bg-accent/30 disabled:opacity-50"
>
  Wait
</button>
<button
  ...
  className="rounded-md border border-secondary/40 px-3 py-1.5 text-sm font-medium text-secondary hover:text-primary hover:border-secondary/60 disabled:opacity-50"
>
  {busy ? 'Cancelling…' : 'Cancel running session and send'}
</button>
```

### Specific mismatches against the rest of the app

Comparing to peer inline-prompt components:

- **`ComposerDisabledPrompt.tsx:36`** (the visually-nearest inline prompt,
  in the same composer region): uses
  `bg-background border border-border rounded-[6px] shadow-lg`.
  SlotHeldNotice uses `rounded-lg` (different radius), `border-secondary/30`
  (wrong border color — the app palette uses `border-border` for opaque
  card edges), `bg-secondary/5` (washed-out tint vs. solid `bg-background`),
  and no `shadow-lg`.

- **`ClaudemdWarningBanner.tsx:48`** (the other inline notice): uses
  `rounded-md`, semantic tint (`border-amber-500/40 bg-amber-500/10`), and
  a leading `lucide-react` icon. SlotHeldNotice has no icon and uses a
  neutral gray tint that reads as "disabled" rather than "attention
  required".

- **Primary button style** in the app (e.g. `ChatView.tsx:2232-2236`
  Queue/Send button, `ComposerDisabledPrompt.tsx:50` Pause-queue button,
  `FolderPickerModal.tsx:299`): all use
  `bg-accent text-white hover:bg-accent/85` with
  `text-xs font-semibold tracking-wide`. SlotHeldNotice's "Wait" button
  uses `bg-accent/20 text-primary` (a watered-down ghost variant)
  and `text-sm font-medium` (wrong size + weight).

- **Secondary button style** in the app (e.g. `ApprovalCard`) uses
  `border-border` not `border-secondary/40`, and a smaller `text-xs`.

### What needs to change

Restyle `SlotHeldNotice` to the app's standard inline-prompt shell
(match `ComposerDisabledPrompt`'s container) plus the standard
primary/secondary button styles. No icon is strictly required, but
adding a warning icon (per `ClaudemdWarningBanner`) would align it with
the "needs user decision" semantic.

---

## Shared root cause section

F1 — caused by the FE-1/FE-3 transform-once refactor that put
`isActivelyRunning` into the `historyItems` memo deps; status flips
now stomp the optimistic tail. Not coupled to the backend SessionKey
work — would reproduce on any session, fresh or established.

F2, F3, F4 — independent cosmetic/component issues with no shared
underlying cause. F3 (chat kbd) and F3 (queue kbd) are the same widget
copy-pasted into two files; otherwise unrelated.

None of F1–F4 is coupled to the unresolved backend SessionKey issue.
F1's session-id story is consistent on both ends — the bug is purely
the seed-effect overwrite.

---

## Files referenced

- `/Users/keanezhou/Desktop/orbital-test/web/src/components/ChatView.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/QueueComposer.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/SlotHeldNotice.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/ComposerDisabledPrompt.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/ClaudemdWarningBanner.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/components/ChatTab.tsx`
- `/Users/keanezhou/Desktop/orbital-test/web/src/hooks/useAgent.ts`
- `/Users/keanezhou/Desktop/orbital-test/agent_os/api/routes/agents_v2.py` (`/chat` endpoint and `/inject` endpoint)
- `/Users/keanezhou/Desktop/orbital-test/agent_os/daemon_v2/agent_manager.py` (`inject_message`, lazy session minting)

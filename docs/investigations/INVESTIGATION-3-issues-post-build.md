# INVESTIGATION — 3 issues reported after post-FE-A1 build

**Date:** 2026-05-28
**Branch:** `fix/rotation-by-session-id` (post-FE-A1/A2/A3/A4 ship)
**Build under test:** `dist/Orbital-0.5.2-macOS.dmg`
**Scope:** read-only. No code edits.

---

## Issue 1 — Approval card "no Deny button"

### Verdict
**The Deny button IS rendered in the DOM.** I verified this end-to-end in the smoke test earlier: my JS evaluate against the real approval card returned `clicked: 1` for `buttons.filter(b => b.innerText.trim() === 'Deny')`. So this is a **rendering / visibility / occlusion** problem, not a missing-element bug. Three plausible causes ranked by likelihood:

### Cause A (most likely) — Deny is rendered but visually subtle
The three buttons use intentionally-different prominence (`ApprovalCard.tsx:240-270`):

```tsx
// Deny — ghost style
"border border-border bg-background text-secondary rounded-[6px]
 text-[11px] px-2.5 py-1 font-medium hover:bg-card-hover"

// Auto-approve 10 min — ghost style (same as Deny)
"border border-border bg-background text-secondary rounded-[6px]
 text-[11px] px-2.5 py-1 font-medium hover:bg-card-hover"

// Approve — solid primary
"bg-primary text-white rounded-[6px] text-[12px] font-medium
 px-3.5 py-1.5 hover:opacity-90"
```

Approve is solid white-on-primary and noticeably larger (`text-[12px] px-3.5 py-1.5`). Deny and Auto-approve are `text-[11px] px-2.5 py-1` ghost buttons in `bg-background` + `text-secondary` + `border-border`. The card's own background is `#FFFBF0` (cream-yellow, `ApprovalCard.tsx:147`). The ghost buttons' `bg-background` is close enough in tone that on a quick glance they read as "card decoration" rather than as clickable buttons. The user likely sees one obvious button (Approve) and misses the two ghost ones to its left.

**File:line:** `web/src/components/ApprovalCard.tsx:247-249` (Deny class), `:257` (Auto-approve class), `:266` (Approve class).
**Fix direction (when you choose to act):** give Deny a clearly-destructive style — e.g. `border-error/40 text-error hover:bg-error/5` (the same pattern the "Confirm Deny" button already uses at `:232`) — so its meaning matches its visual weight.

### Cause B (also live; surfaced in my smoke run) — sticky-bottom buttons get occluded by adjacent chat content
The Playwright click on Deny earlier failed three times with the same error: another DOM element intercepted pointer events. The intercepting element was the next chat row (`<p>Claude Code 子代理已完成但未生成输出文件…</p>`). Root cause is the layout choice at `ApprovalCard.tsx:241`:

```tsx
<div className="flex flex-col md:flex-row gap-2 md:justify-end
                sticky bottom-0 md:static
                bg-[#FFFBF0] pt-2 md:pt-0
                -mx-4 px-4 md:mx-0 md:px-0 pb-1 md:pb-0">
```

`position: sticky` plus `bottom-0` on the buttons row, inside an outer card with `overflow-hidden` (`:147`). On mobile, the buttons are supposed to stick to the bottom of the chat scroll container, but the `overflow-hidden` ancestor changes that behavior — and because the buttons row carries its own `bg-[#FFFBF0]` background and bleeds with `-mx-4 px-4`, when the layout doesn't behave as expected the row visually sits "underneath" the next chat row but ABOVE it in z-order at certain scroll positions, so clicks aimed at the buttons hit the adjacent text. Whether this manifests as "Deny is invisible" or "Deny doesn't respond to taps" depends on viewport and scroll position.

**Fix direction:** drop `sticky bottom-0` (the chat container already scrolls), OR move the row outside the `overflow-hidden` wrapper, OR use a flex layout in the chat list that bottom-pins approvals without relying on `sticky`.

### Cause C (rule out only with a screenshot) — wrong card variant
The chat render path conditionally renders `CredentialCard` instead of `ApprovalCard` when `tool_name === 'request_credential'` (`ChatView.tsx:1943-1985`). `CredentialCard` HAS a Deny button (`CredentialCard.tsx:122-131`), so this only matters if you can confirm via DevTools that `[data-testid="agent_run"]` is showing a credential card vs an approval card. Worth a quick check.

### What to confirm before fixing
Open DevTools on the affected card and inspect: (a) is the `<button>Deny</button>` present in the DOM tree? (b) what's its computed `bg-background` actually rendering as — solid white or transparent against the cream card? (c) does the card you're looking at have `border-warning/50 bg-[#FFFBF0]` (approval) or `border-border` (credential)? Those three answers split A/B/C cleanly.

---

## Issue 2 — Project Settings "duplicate headers"

### Verdict
The two views never render together; the App-level router is a strict ternary (`App.tsx:488-528`). What looks duplicate is the header band inside `SettingsModalPage` itself — it stacks **four** title-class elements on top of each other:

```tsx
// SettingsModalPage.tsx:39-57
<div className="flex flex-col gap-1 px-6 pt-5 pb-4 border-b border-border">
  <button>← Back to {project.name}</button>          // line 40-47
  <p className="text-xs font-mono uppercase ...">
    Project · this project only
  </p>                                                // line 48-50  ← chip
  <h1 className="text-lg font-semibold ...">
    Project settings — {project.name}                 // line 51-53  ← h1
  </h1>
  <p className="text-sm text-secondary">
    Configure agent behaviour, LLM provider, ...
  </p>                                                // line 54-56  ← subtitle
</div>
```

For a project named "Quick Tasks", you see in vertical sequence:

1. `← Back to Quick Tasks`
2. `PROJECT · THIS PROJECT ONLY`
3. `Project settings — Quick Tasks`
4. `Configure agent behaviour, LLM provider, autonomy, and more for this project.`

Three of those four lines say "Project" and two of them name the project. The chip (#2) and the h1 (#3) are the redundant pair the user is reading as a "duplicate header" — both are uppercase-vs-large variants of the same statement. The back button (#1) also re-states the project name.

`SettingsView.tsx` does NOT add its own h1 — it starts straight at the `Agent Name` form field (`SettingsView.tsx:267+`). So the duplication is purely inside the modal page's own band, not a cross-component clash.

**File:line:** `web/src/components/SettingsModalPage.tsx:39-57`.
**Fix direction:** collapse to one title + one subtitle. The chip "Project · this project only" was added to disambiguate from a Global Settings view of the same field, but the h1 already says "Project settings — {name}", which carries the same information. Drop the chip OR drop the redundant project name in the h1.

### Sanity-check on the "two cards both rendering" hypothesis
Just to be explicit: I traced `App.tsx:488-528` — `{route.settings ? <SettingsModalPage…/> : <ProjectDetail…>{tabs}</ProjectDetail>}` is a ternary, not a fragment. ProjectDetail's own header (`Quick Tasks` + status badge + `Assistant · $0.00` + Settings icon) is NOT in the DOM when `route.settings === true`. Verified by reading the JSX, no logic branch produces both.

---

## Issue 3 — "Why does the autonomy preset trigger an approval for a sub-agent request?"

### Verdict
This is **by design, not a bug** — but the design is confusing UX and I think you're asking the right question. Here's what the code does today:

### The chain
1. **Project carries an autonomy** (`hands_off` / `check_in` / `supervised`). For Quick Tasks today: `check_in`.
2. **Sub-agent inherits the project's autonomy at start time** — `sub_agent_manager.py:460-467`:
   ```python
   autonomy = None
   if project:
       autonomy_str = project.get("autonomy", "check_in")
       try:
           autonomy = Autonomy(autonomy_str)
       except ValueError:
           autonomy = Autonomy.CHECK_IN
   ```
   Then `:552-555` passes it into the sub-agent's transport:
   ```python
   transport = self._resolve_transport(
       manifest, config_dict, autonomy=autonomy, system_prompt=system_prompt,
   )
   ```
3. **SDKTransport applies the SAME `should_auto_approve` policy as the management agent** — `sdk_transport.py:316-319`:
   ```python
   if self._autonomy is not None and should_auto_approve(tool_name, self._autonomy):
       return PermissionResultAllow()
   ```
4. **The policy in `tool_risk.py:60-73`:**
   - `HANDS_OFF` → auto-approve everything.
   - `CHECK_IN` → auto-approve only `READ` category tools (Read / Glob / Grep / LS / Search / Explore / TaskGet / TaskList / TaskOutput / WebSearch / WebFetch / AskUser).
   - `SUPERVISED` → auto-approve nothing.
5. So when claude-code (sub-agent) under a `check_in` project tries `Edit` / `Write` / `Bash` / `TodoWrite` / `Agent`, `should_auto_approve` returns False → `_handle_permission` queues a permission_request → frontend renders an `ApprovalCard` → user approves/denies.

### Why this is "the autonomy preset is working"
Exactly — it's working as written. The project-level autonomy preset is the single source of truth for any tool decision happening inside that project, regardless of whether the requester is the management loop or a sub-agent.

### Why your intuition (sub-agents shouldn't prompt) is also defensible
The user's mental model when typing "@claude-code do X" is "I've already approved Claude Code to do whatever X requires." Requiring a second approval for every Bash/Edit inside that delegation feels like a re-prompt for consent already given. There are three reasonable design options:

| Option | Behavior | Implementation |
|---|---|---|
| **A. Status quo** | Sub-agents respect project autonomy; user is asked per-tool. | Already shipped. |
| **B. Sub-agents always permissive** | Once dispatched, the sub-agent's tools auto-approve. | Pass `autonomy=Autonomy.HANDS_OFF` (or `None`) to `_resolve_transport` at `sub_agent_manager.py:554`. |
| **C. Separate `sub_agent_autonomy` preset** | The user picks autonomy per-tier — management vs. sub. | Add a project field, surface a second autonomy switch in `SettingsView`, plumb it into the sub-agent start path. |

The current code chose A. Your question reads as a preference for B (or maybe C). It's a design call, not a bug — if you want B I can wire it up in one commit; the chain above is the only path that needs to change.

### Note on the management agent's `agent_message` dispatch tool
The MANAGEMENT agent's `agent_message` tool (the one that says "dispatch to @claude-code") goes through the management agent's own autonomy gate, not the sub-agent's. `agent_message` is not in `_TOOL_CATEGORY` (`tool_risk.py:22-49`), so `classify_tool("agent_message") == REQUIRES_APPROVAL`. Under `CHECK_IN`, that would normally prompt — but the management agent in this codebase uses a different transport (not SDKTransport — agent_message is implemented as an Orbital tool defined in the prompt/loop), so `_handle_permission` doesn't fire for it. Dispatches are silent at the management level. The approval the user sees is from the SUB-agent's transport, not the dispatch.

---

## Summary table

| # | Issue | Severity | Code state | Fix complexity |
|---|---|---|---|---|
| 1a | Deny button visually too subtle | P1 (usability) | Real CSS choice in `ApprovalCard.tsx:247-249` | One class change |
| 1b | Sticky-bottom buttons collide with adjacent chat rows | P1 (mobile usability) | `ApprovalCard.tsx:241` `sticky bottom-0` inside `overflow-hidden` ancestor | Drop `sticky` |
| 2 | 4-line header band in Project Settings has redundant title text | P2 (cosmetic) | `SettingsModalPage.tsx:39-57` | Trim one of chip/h1 |
| 3 | Sub-agent tool calls trigger project-level approvals | Design question, not a bug | `sub_agent_manager.py:460-554` + `sdk_transport.py:316-319` + `tool_risk.py:60-73` | One-line `autonomy=Autonomy.HANDS_OFF` override if Option B is chosen |

No code was modified — this is investigation only. Recommend confirming #1 with a screenshot (which sub-cause is the real one) before picking a fix, and confirming #3 is a design preference (Option B vs A) before changing it.

# Raw user bug report — 2026-05-28 round 3 (post bf3c97c)

Captured verbatim. Two new bugs surfaced during retest of the latest .app build.

---

## User report (verbatim)

> a few things. the single session slot wasnt enforced during sub agent dispatch.
> please investigate again. and also, i notice that the approval request wasnt
> rendered. please dispatch sub agents for the investigation again. this time
> give me three potential root causes. no code change. still the two latest
> session in quick tasks

## Bugs

(R3-1) The single-session slot was NOT enforced during sub-agent dispatch.
Expected: when session A has a sub-agent running, attempting to start work in
session B in the same project should be blocked (or surface the SlotHeldNotice
rotation prompt). Observed: session B starts freely.

(R3-2) The approval request was NOT rendered.
When a sub-agent (claude-code, which spawns with `--permission-prompt-tool stdio`)
requests permission for a tool call, the user-facing approval modal/card does
not appear. The dispatch presumably stalls or proceeds without user consent.

## Investigator instructions

- Use sub-agents for both investigations (coordinator context may be biased).
- No code change.
- Look at the latest two sessions in Quick Tasks for raw evidence.
- For EACH bug, surface THREE potential root causes ranked by likelihood with
  evidence supporting and ruling-out each.

---

## Environment snapshot at report time

- Running daemon: `/Applications/Orbital.app` PID 39197, started Thu May 28 13:53:23 2026.
- `/Applications/Orbital.app/Contents/MacOS/Orbital` binary mtime: 13:51:11,
  matches `dist/Orbital.app/Contents/MacOS/Orbital` bytes-equal.
- Fix commit `bf3c97c` (SDK-completion routing + chat UI regressions) at 13:50.
- Build complete: 13:51:11. .dmg: 13:51:23. App installed + started 13:53:23.
- Previous fix `7d08d1b` (SessionKey routing for tool / observer / manager) also loaded.
- Quick Tasks project ID from prior round: `proj_6be4f16fb272`.
- Workspace: `/Users/keanezhou/Library/Application Support/Orbital/scratch`.
- Daemon log: `/Users/keanezhou/Library/Application Support/Orbital/logs/daemon.log`.
- Session JSONLs: `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sessions/`.

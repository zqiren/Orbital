# Raw user bug report — 2026-05-28, post-fix testing

Captured verbatim. User reports the previously-shipped fix
(`7d08d1b fix(daemon): route sub-agent lifecycle under management SessionKey`,
committed 11:54, .dmg rebuilt 12:06, `/Applications/Orbital.app` installed
12:10) did NOT resolve the symptoms. Raw material for investigators — do not
edit or summarize.

---

## Frontend bugs

1. The instant I send the message, the message is invisible on the chat interface.
2. There is a `>` sign between attachment and text input area.
3. There is a command-enter symbol on the right of the text input area and the
   queue text input area. Please remove them.
4. The single-slot reminder ("stop the running session and run the current
   session") component is not the same style as the rest of the app.

## Backend bugs (Quick Tasks, latest two sessions)

1. The fix didn't land — when dispatched to sub-agent, the session still
   appears idle on the front end.
2. When one session dispatched a sub-agent, other sessions are still able
   to run.
3. The slot rotation — "stop the running session and run the current session"
   — didn't work.

## Investigator instructions from the user

- Please dispatch sub-agents to investigate these.
- No code change.
- Put the original report somewhere so we have raw materials to refer to.
- Please also check if the currently running daemon contains the fix.
  This bug has been here many times.
- Use sub-agent to investigate, your (coordinator's) context may be
  contaminated by the previous approaches already. ultrathink.

---

## Environment snapshot at report time (12:38 PDT)

- Running daemon process: `/Applications/Orbital.app` PID 47514, started
  Thu May 28 12:10:18 2026.
- `/Applications/Orbital.app/Contents/MacOS/Orbital` mtime: 12:05:53.
- Fix commit `7d08d1b` time: 11:54:42.
- Rebuilt `.dmg`: `dist/Orbital-0.5.2-macOS.dmg` at 12:06:06.
- Inner binary bytes-equal between `/Applications/Orbital.app` and
  `dist/Orbital.app` (size 22908336, mtime 12:05:53) — i.e. the bootloader
  was copied across, so the .app was reinstalled from the fresh build.
- An active claude sub-agent process was visible at PID 73864, suggesting
  at least one dispatch had occurred under the .app during testing.
- Project workspace path observed: `/Users/keanezhou/Library/Application Support/Orbital/scratch`.

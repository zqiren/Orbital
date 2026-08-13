# dsh evaluation artifacts (2026-08-13)

Preserved from the live evaluation session that established the Verified Facts
in `docs/superpowers/plans/2026-08-13-deepseek-harness-subagent.md`. These are
smoke-test seeds, not shipping assets — the shipping composition lives at
`agent_os/agents/assets/dsh/` (Task 1 of the plan).

- `cordis.yml` — the composition that booted `dsh-acp-demo` end-to-end against
  a real DeepSeek key (9 plugins; note `dsh-sandbox-policy` and
  `dsh-user-approval` resolved transitively, which is why the shipping
  package.json must pin all nine).
- `package.json` / `package-lock.json` — the exact install that was verified
  (`^0.1.0-rc.6` ranges; the lockfile records the working resolution and
  integrity hashes; the shipping manifest pins exactly and installs via
  `npm ci`).
- `probe_real.py` — full ACP turn (initialize → session/new → prompt with real
  bash tool use). Requires `DEEPSEEK_API_KEY` in the environment.
- `probe_multiturn.py` — two prompts on one live session ("remember 8317" →
  recall) proving in-process multi-turn memory.

Run from a directory containing an installed composition (npm ci against
package.json, cordis.yml present):

    DEEPSEEK_API_KEY=... python probe_real.py
    DEEPSEEK_API_KEY=... python probe_multiturn.py

These are manual/live scripts — they hit the DeepSeek API and are not part of
the pytest suite.

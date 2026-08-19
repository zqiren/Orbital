# dsh evaluation artifacts (2026-08-13)

Preserved from the live evaluation session that established the dsh sub-agent's
verified facts. These are smoke-test seeds, not shipping assets — the shipping
composition lives at `agent_os/agents/assets/dsh/`.

- `cordis.yml` — the composition that booted `dsh-acp-demo` end-to-end against
  a real DeepSeek key (9 plugins; note `dsh-sandbox-policy` and
  `dsh-user-approval` resolved transitively, which is why the shipping
  package.json must pin all nine).
- `package.json` / `package-lock.json` — the exact install that was verified
  (`^0.1.0-rc.6` ranges; the lockfile records the working resolution and
  integrity hashes; the shipping manifest pins exactly and installs via
  `npm ci`).

The live ACP probe scripts that accompanied this evaluation were machine-local
and are not preserved here; they are recoverable from git history if the
end-to-end turn ever needs re-running.

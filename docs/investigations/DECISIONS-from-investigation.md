# DECISIONS from sub-agent context investigation

**Status:** PROPOSED. Awaiting Qiren review per the spec ("DECISIONS-from-investigation.md exists and is reviewed by Qiren before any implementation spec is written").

**Inputs:** [FINDINGS-sub-agent-context-and-persistence.md](FINDINGS-sub-agent-context-and-persistence.md), [PROMPT-TEMPLATE-sub-agent-system.md](PROMPT-TEMPLATE-sub-agent-system.md).

---

## D1. Should orbital build a parallel session log for sub-agents?

**Recommendation: NO. Lean on Claude Code's persisted session JSONL.**

Q1 and Q5 confirmed that `claude --resume <uuid>` restores the full message history (user/assistant turns AND tool_use/tool_result blocks) across process boundaries. The session lives at `~/.claude/projects/<sanitized-cwd>/<uuid>.jsonl` and survives orbital daemon restart. Building a parallel log on orbital's side would duplicate state that already lives on disk and add a divergence risk (orbital's view of "what the sub-agent has seen" vs. what claude actually has in context).

**What orbital DOES need to track:**
1. The `session_id` per sub-agent (already present in `SDKTransport`).
2. The **`workspace_cwd`** at session creation (Q6 finding — resume from a different cwd fails). Orbital should refuse to resume if `current_workspace_cwd != session_workspace_cwd`, and treat that as session-loss + restart.

**What orbital does NOT need to build:**
- Tool-call history mirror.
- Message log mirror.
- Manual replay of past messages on resume.

This applies to **claude-code only** (SDK transport). For other transports see D5.

---

## D2. Should orbital inject project-context file *contents*, or rely on the agent to read them?

**Recommendation: rely on the agent to read them, via the system-prompt template (PROMPT-TEMPLATE doc). DO NOT auto-inject file contents into the prompt.**

The Tier 4 trial got 100% adherence on the very first iteration: the model reliably called `Read` on the relevant project files before responding. There is no measured ceiling that would force orbital to fall back to content injection.

Reasons to prefer agent-read over content-injection:
- **Token efficiency:** files are read into context only when the agent decides they are relevant. Injecting all four files into every system prompt would waste cache and bloat invariant prompt size.
- **Freshness:** if a file is edited mid-session, the next `Read` picks up the new content. Injection would freeze the contents at session start.
- **Cwd-portability:** injected paths would be rendered into a static prompt; agent-read paths track the live cwd.

**Caveats / open questions:**
- The trial used Haiku 4.5. If the default sub-agent model changes, re-run the Tier 4 trial before keeping this decision.
- Adversarial prompts ("ignore the rules above") were NOT tested. If the agent ignores its system prompt under user pressure, content-injection becomes a stronger fallback. Flag for a separate prompt-injection investigation.
- The template currently does NOT explicitly say "if these files do not exist, proceed without reading them." When orbital first dispatches into a fresh project, the files may be absent and the agent will issue Read-not-found tool calls. Either:
  - **(a)** ensure orbital writes placeholder content to the four files at project init, OR
  - **(b)** amend the template with a "files may not exist; proceed if absent" clause and re-run Tier 4.

  I recommend **(a)** — placeholder files are cheap, deterministic, and reduce cognitive load on the model.

---

## D3. Should orbital use `--system-prompt`, `--append-system-prompt`, or CLAUDE.md as the injection point?

**Recommendation: use `--append-system-prompt-file <path>` for orbital-managed instructions. Reserve CLAUDE.md for project-author-controlled content.**

Q3 established the priority order on conflict: **CLAUDE.md > `--system-prompt` ≈ `--append-system-prompt`**. CLAUDE.md wins even when `--system-prompt` claims to fully replace the default Claude Code prompt. So:

- **`--append-system-prompt-file`** — orbital's own per-dispatch instruction layer (the PROMPT-TEMPLATE). Append-mode preserves Claude Code's default behavior. **File variant is mandatory** (Q7 finding: inline argv hits the OS limit at ~32 KB on Windows; the `-file` flag scales to 50 KB+).
- **`--system-prompt-file`** — only when orbital explicitly wants to replace Claude Code's default system prompt. Risky; not needed for the inheritance use case.
- **`CLAUDE.md`** — leave to project authors. Orbital should NOT auto-write to a project's `CLAUDE.md`. Doing so would (a) be invisible to project authors who don't expect orbital to mutate their canonical config file, and (b) collide with project-authored content that legitimately wins on conflict.

**Implication for the worker-context-inheritance design:** the inheritance mechanism is `--append-system-prompt-file <orbital-rendered-template>`. Project authors retain control over CLAUDE.md, which orbital reads but does not write.

---

## D4. What about the ancestor-walk leak (`~/.claude/CLAUDE.md`)?

**Recommendation: document the leak. Do NOT try to suppress it for v1.**

Q2 found that claude-code walks the cwd ancestor chain all the way to `~/.claude/CLAUDE.md` and concatenates everything. So a sub-agent orbital dispatches inherits the user's personal Claude Code config (memories, behavioral preferences, etc.) by default. There is **no clean suppression** other than `--bare`, which disables OAuth entirely (forcing API-key auth).

This is mostly fine — user-level CLAUDE.md tends to be benign behavioral preferences. But:
- If the user has memory entries that contradict orbital's project-level guidance, **the user's content wins** by virtue of being added later in the concatenation chain (assuming the documented "loads in order, concatenates" semantic).
- Orbital cannot, today, give users a knob like "isolate this sub-agent from my personal CLAUDE.md."

**Defer the suppression mechanism to a follow-up.** When users have a clear scenario where personal CLAUDE.md interferes with project work, orbital can revisit `--bare`-plus-API-key as an opt-in mode. For v1 inheritance, document the leak in user-facing docs ("your sub-agents inherit your personal Claude Code memories by default") and move on.

---

## D5. What is the degraded path for non-SDK transports (PTY, Pipe) and ACP?

**Recommendation: per-transport, ranked from best to worst case.**

| Transport | Inheritance status | Recommended path |
|---|---|---|
| **SDKTransport** (claude-agent-sdk → claude.exe) | Works fully. | Append system prompt via `ClaudeAgentOptions(system_prompt=...)` (the SDK exposes this). All Tier 1 findings apply directly. |
| **PipeTransport** (`claude -p --output-format stream-json`) | Works fully. | Pass `--append-system-prompt-file <path>` in args. Same persistence guarantees as SDK (Q1/Q5). |
| **PTYTransport** (interactive TTY) | **Degraded.** No clean per-session system-prompt injection on a TTY-launched session. | Inject context via the FIRST user turn ("Before we begin, here's the project state: ..."). Pre-write context to `<workspace>/CLAUDE.md` and rely on Q2-confirmed auto-discovery. Both paths leave the inheritance instruction inside the message log rather than the system prompt — robustness is lower. |
| **ACPTransport** (JSON-RPC over stdio) | **Dormant against claude-code.** Q4: claude-code 2.1.138 has no `acp` subcommand. Per the README, ACP targets gemini-cli. | Two follow-ups: (a) tag ACP transport in the registry as gemini-only and stop attempting it on claude agents; (b) once a gemini-cli probe machine is available, re-run Q1/Q2/Q3/Q5 against `gemini --acp`-equivalent. Tier 3 docs suggest gemini's session-prompt knob is `GEMINI_SYSTEM_MD` env var, not a CLI flag — so ACP per-session injection may need to be set in the JSON-RPC `session/new` params, not via process args. Confirm empirically. |

**Concrete code-level follow-ups (NOT for this PR — investigation only):**
1. `agent_os/agent/transports/acp_transport.py`: tag as gemini-only or remove from claude-code registry entries.
2. `agent_os/agent/transports/sdk_transport.py:69-86`: add an optional `system_prompt: str | None` argument that gets forwarded into `ClaudeAgentOptions`. Today the transport hardcodes only `cwd / permission_mode / can_use_tool / cli_path / env`.
3. `agent_os/agent/transports/pipe_transport.py`: same — wire up `--append-system-prompt-file` injection.
4. The PTY path needs a "first turn injection" wrapper. Out of scope for v1; flag as a known gap.

---

## D6. Should orbital store `sessionId` durably across daemon restart?

**Recommendation: yes. Store `(sub_agent_id, session_id, workspace_cwd, created_at)` tuples in orbital's own state.**

Q6 confirmed:
- Sessions persist on disk indefinitely under `~/.claude/projects/<cwd>/<uuid>.jsonl`.
- A fresh process can resume them as long as cwd matches.
- Resume from a different cwd returns *"No conversation found with session ID"*.

Orbital today already tracks `session_id` in `SDKTransport._session_id` but only in process memory. After a daemon restart that field is None. Orbital should:
1. Persist sessionId per sub-agent in its existing state store (wherever sub-agent metadata lives — likely `~/.orbital/<project>/state/`).
2. Include `workspace_cwd` alongside, and validate match before resume.
3. On mismatch (or missing JSONL on disk), fall back to `--session-id <new-uuid>` rather than erroring out.

This is small, well-scoped, and unlocks transparent sub-agent continuity across orbital restart.

---

## D7. What about the Specialist concept v1?

The spec lists Specialist v1 as a downstream gate. Findings relevant to it:

1. **A specialist is a sub-agent with an opinionated `--append-system-prompt`.** No new transport mechanism is needed — the same injection path that worker-context-inheritance uses also delivers specialist-level instructions. Stack them: `[orbital base template] + [specialist persona] + [project state pointers]`.
2. **Size is not a concern.** Q7 showed 50 KB system prompts work fine via the `-file` variant. A specialist persona of a few KB plus project pointers fits comfortably.
3. **Specialist personae must be designed not to conflict with project CLAUDE.md.** CLAUDE.md wins on conflict (Q3). If a specialist persona tells the sub-agent "always respond in JSON" but project CLAUDE.md says "always respond in markdown", the project wins. This is probably the right default — specialists are augments, not overrides — but document it so specialist authors don't fight the priority order.
4. **Specialist invocation: same `--append-system-prompt-file` mechanism.** No new code path. Just a different file rendered into it.

---

## Summary

The empirical floor under the worker-context-inheritance design is solid:

- Session continuity works (Q1/Q5/Q6) → no parallel log needed.
- Prompt-driven file consultation works at 100% on the first try (Tier 4) → no content-injection needed.
- Append-mode prompt injection works up to 50 KB via the `-file` variant (Q7) → space is not a constraint.
- CLAUDE.md trumps `--system-prompt` (Q3) → orbital injects via append-mode and leaves CLAUDE.md to project authors.

The remaining unknowns — ACP behavior against gemini-cli, PTY degraded path, prompt-injection robustness against adversarial users — are flagged as separate follow-ups and do not block the inheritance design from being written.

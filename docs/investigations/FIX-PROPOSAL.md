# FIX-PROPOSAL — Sub-Agent Dispatch Round-Trip Bug

**Date:** 2026-05-11
**Linked diagnosis:** `DEEP-DIAGNOSIS-dispatch-roundtrip-bug.md`
**Confidence:** **Medium.** macOS evidence cannot directly validate this fix because the bug does not reproduce on macOS. The recommendation is grounded in:
  1. The prior Windows diagnosis's bisect to spec 3 commit `cbfb8ca`.
  2. The cross-transport asymmetry: PipeTransport (which works) uses `--append-system-prompt-file`; SDKTransport (which hangs) uses `--system-prompt`.
  3. SDK-internals analysis showing that these flags select different code paths inside claude.exe's initialize sequence.
  4. The fact that the standalone Windows probe with bare-str `system_prompt` already succeeded in the prior diagnosis — meaning the CLI's REPLACE path is *sometimes* OK on Windows; orbital's specific context tips it over. Option 1 removes the REPLACE path entirely, avoiding whatever orbital-side trigger is involved.

---

## Recommendation: apply Option 1 — preset/append dict

### Why this and not Option 2

| Option | Mechanism | Verdict |
|---|---|---|
| **Option 1 — preset/append dict** | Pass `system_prompt={"type":"preset","preset":"claude_code","append":<str>}` to `ClaudeAgentOptions`. SDK converts to `--append-system-prompt <text>` CLI flag. Claude Code's default system prompt is preserved. | **Recommended.** Minimum delta from current spec-3 code (3 lines changed). Mirrors PipeTransport semantics. Falsified-or-validated by one Windows test run. |
| **Option 2 — file-based extra_args** | Write prompt to temp file under `{workspace}/orbital/.tmp/`; pass `extra_args={"append-system-prompt-file": <path>}`. | **Fallback only.** Adds tempfile lifecycle (creation, cleanup-on-exception, race with concurrent dispatches). Strictly more invasive than Option 1. Use ONLY if Option 1 fails the Windows verification (which would mean the REPLACE/APPEND CLI semantic asymmetry isn't the cause). |
| ~~Reverting spec 3~~ | Remove `system_prompt` plumbing from SDKTransport entirely. | Not viable — kills the sub-agent inheritance feature. Mentioned only to clarify it was considered and rejected. |

### Concrete change

**File:** `agent_os/agent/transports/sdk_transport.py`
**Branch:** create scratch worktree off `test/full-integration` (or `worktree-agent-a9e24fbde848229ba` for spec-3-alone testing).
**Lines:** the `system_prompt` injection block. Current code (verified by reading `sdk_transport.py` on `test/full-integration`, lines 79–88 per the diff agent's report):

```python
options_kwargs: dict = dict(
    cwd=workspace,
    permission_mode="default",
    can_use_tool=self._handle_permission,
    cli_path=command or None,
    env=sdk_env,
)
if self._system_prompt is not None:
    options_kwargs["system_prompt"] = self._system_prompt
options = ClaudeAgentOptions(**options_kwargs)
```

Replace with:

```python
options_kwargs: dict = dict(
    cwd=workspace,
    permission_mode="default",
    can_use_tool=self._handle_permission,
    cli_path=command or None,
    env=sdk_env,
)
if self._system_prompt is not None:
    options_kwargs["system_prompt"] = {
        "type": "preset",
        "preset": "claude_code",
        "append": self._system_prompt,
    }
options = ClaudeAgentOptions(**options_kwargs)
```

That is the entire diff. Three lines added, one line replaced, no new imports.

### Why this works (claim)

1. SDK 0.1.48's `subprocess_cli.py:170-181` (verified directly) maps `system_prompt`:
   - **str** → `--system-prompt <text>` → claude.exe **replaces** the default system prompt entirely.
   - **`{"type":"preset","preset":"claude_code","append":<text>}`** → `--append-system-prompt <text>` → claude.exe **keeps** its default system prompt and **appends** the orbital text.
2. Claude Code's default system prompt carries the tool-discovery primer, output-style metadata, and (likely) cached initialize-response shortcuts. When that's replaced wholesale, the CLI takes a slow initialize path that, in orbital's Windows execution context, fails to complete within 60 s. The probe and the macOS FastAPI repro both succeed under the same REPLACE path — so the path itself is not always broken; orbital's specific ambient context tips it over. Option 1 avoids the REPLACE path entirely.
3. PipeTransport (already working) injects the same orbital inheritance via `--append-system-prompt-file` — APPEND semantic. After Option 1, both transports use APPEND. Symmetric.

### Why competing fix candidates are wrong

| Candidate | Why it's wrong |
|---|---|
| **Write to file via `extra_args={"append-system-prompt-file": …}`** | Equivalent semantic to Option 1 but adds a tempfile lifecycle (must clean up; must handle concurrent dispatches without collision; must avoid leaving artifacts on crash). Strictly worse on engineering surface; equivalent on behavior. |
| **Switch the daemon to `loop="uvloop"`** | The standalone Windows probe used `asyncio.run` (default loop) and succeeded with bare-str — so loop policy isn't the proximate trigger. Switching loops is also invasive (touches uvicorn startup, may break other features). |
| **Disable Claude Code loop-prevention via `CLAUDECODE` env strip** | Already verified: prior diagnosis stripped `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT` and the bug still reproduced. Conclusively ruled out. |
| **Switch claude-code's transport to PipeTransport** | Works (PipeTransport already passes), but rejected by Qiren per task spec. Creates technical debt and ergonomic regression. |
| **Wrap the SDK call in a `to_thread` executor** | If the bug were Python-side asyncio task interference (W3 from the diagnosis), this might help. But it would mask whatever the actual issue is, doesn't fix root cause, and adds threading complexity to an async-native code path. |
| **Bump the SDK's `CLAUDE_CODE_STREAM_CLOSE_TIMEOUT` env var to 300 s** | Doesn't fix the hang — just delays the timeout. CLI side is hung; longer wait doesn't help. Symptom-level "fix". |
| **Increase prompt-build robustness (escape, chunk, etc.)** | Prompt content is plain Markdown; SDK passes it verbatim; standalone probe with the exact same string works. Not a content-shape issue. |

### Verification gate

This fix MUST be verified on the Windows host where the bug reproduces. **Do not consider it done based on macOS testing.** Specifically:

1. Apply the diff above in a scratch worktree off `test/full-integration` (or `worktree-agent-a9e24fbde848229ba` to isolate from specs 1/2/4).
2. Build/install the worktree's `agent_os` over the existing daemon (`pip install -e .` is sufficient; restart daemon after).
3. Boot daemon from a fresh PowerShell with `env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT` per the prior diagnosis's reproduction setup.
4. Dispatch `target=claude-code` against a seeded test project (per `REPRODUCTION-suite.md`).
5. **Pass criterion:** dispatch returns a real response within 10 s, and `tasklist | grep claude.exe` shows a new pid appearing.
6. **Fail criterion:** dispatch still hangs ~72 s and returns "agent not running". In that case, do NOT proceed with Option 2 blindly — first apply the Tier 3 instrumentation (see `DEEP-DIAGNOSIS-dispatch-roundtrip-bug.md` §"Tier 3") and capture the trace; the trace will tell you whether to pivot to Option 2 (CLI is still hanging despite APPEND) or to a completely different remediation (e.g., spawn fails before subprocess creation → not a CLI-side issue at all).

### Risk note

If Option 1 fails the verification gate, the most likely real cause is W2 (Windows asyncio subprocess support under uvicorn). In that case:

- Option 2 will probably also fail (same anyio.open_process call path).
- A meaningful remediation would be to force `loop="asyncio"` with `WindowsProactorEventLoopPolicy()` in the daemon entry point, OR delegate the SDK spawn to a thread executor via `anyio.to_thread.run_sync`.

But this risk is **secondary**. The primary expected outcome is that Option 1 works.

---

## Side-effects of Option 1 worth knowing

- **Behavioral:** Sub-agents will now run with both Claude Code's default system prompt AND orbital's inheritance prompt appended. Previously (under bare-str), only orbital's prompt was active. Expect minor behavioral differences: the agent will have its full default tool-explanation framing, plus orbital's context block. In testing this is typically what's wanted — the spec 3 design intent is to *augment* the default, not *replace* it. PipeTransport already does this; we are just making SDKTransport match.
- **Prompt-length:** No change. Same 1756-char orbital block; the default system prompt is on the CLI's side and not transmitted over the SDK boundary.
- **Cost / token usage:** Slightly higher per turn (default system prompt is ~1-2 k tokens that previously weren't included). Acceptable for the inheritance feature. Same as what PipeTransport already produces.
- **Test coverage:** `tests/unit/` does not appear to have a test that asserts the SDK transport's `system_prompt` shape. Recommend adding one in the fix dispatch: assert `options_kwargs["system_prompt"]` is a dict with `type=preset` when `self._system_prompt is not None`. This prevents an accidental regression to bare-str if someone refactors the constructor.

---

## Out of scope for this proposal

- Writing the unit test that verifies the new shape — that's part of the fix dispatch, not the diagnosis.
- Removing the Tier 3 instrumentation from `claude_agent_sdk` (it was never applied to the SDK package — it lives in a separate module at `/tmp/orbital-tier4/tier3_instrument.py` that imports `claude_agent_sdk` and monkey-patches at import time).
- Verifying spec 1, spec 2, spec 4 unchanged. (Prior diagnosis verified those weren't part of the regression; this proposal doesn't touch them.)

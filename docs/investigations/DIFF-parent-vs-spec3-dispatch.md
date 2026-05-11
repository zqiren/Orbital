# DIFF — Parent vs Spec 3 Dispatch

**Investigation:** `TASK-diff-parent-vs-spec3-dispatch.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-agent-sdk 0.1.48, claude-code 2.1.138.
**Branches compared:**
- Parent: `feature/render-chat-variant-a` @ `7bc9ef4` (the published parent; dispatch code identical to its commit `63140eb` where the original bisect happened)
- Spec 3 alone: `worktree-agent-a9e24fbde848229ba` @ `cbfb8ca`
**Status:** Decisive. Classification: **Argv composition regression** (row 2 of the spec's Phase 3 pattern table). No fix proposed.

---

## 1. Phase 1 — static comparison

Comparison of the four dispatch-path artifacts named in the spec:

| Artifact | Parent | Spec 3 | Diff |
|---|---|---|---|
| `agent_os/agents/manifests/claude_code.yaml` → `transport:` field | `sdk` | `sdk` | **same** (rules out "transport switch" classification) |
| `agent_os/agents/manifests/claude_code.yaml` → `mode:` field | `pipe` | `pipe` | same |
| `_resolve_transport()` for claude+sdk | constructs `SDKTransport(autonomy=autonomy)` | constructs `SDKTransport(autonomy=autonomy, system_prompt=system_prompt)` | spec 3 adds `system_prompt` kwarg (default `None`, identical behavior when `None`) |
| `SDKTransport.__init__` signature | `(autonomy=None)` | `(autonomy=None, system_prompt=None)` | spec 3 adds optional `system_prompt` field |
| `SDKTransport.start()` `ClaudeAgentOptions(...)` kwargs | `cwd, permission_mode, can_use_tool, cli_path, env` | same + conditional `system_prompt` IFF `self._system_prompt is not None` | **identical when `system_prompt is None`** |
| `PipeTransport.__init__` signature | `(config=None)` | `(config=None, system_prompt=None, agent_slug=None)` | spec 3 adds two optional kwargs (defaults preserve parent behavior) |
| `_start_from_registry()` pre-flight | (no sub-agent inheritance block) | NEW ~50-line block: `setup_engine.check_all()` → peer-slug list → `ensure_memory_md()` → `render_sub_agent_prompt()` → `_maybe_emit_claudemd_warning()` | spec 3 always renders a non-empty `system_prompt` when `workspace` is set |

### Phase 1 decision-gate verdict (preliminary)

Spec 3 adds optional `system_prompt` plumbing throughout the dispatch chain, but the only **runtime-observable** consequence is whether `_start_from_registry()` calls `render_sub_agent_prompt(...)` and produces a non-`None` string. When it does, that string flows down to `ClaudeAgentOptions(system_prompt=...)`. When it doesn't (parent path), `ClaudeAgentOptions` doesn't receive a `system_prompt` kwarg.

This **rules out classification #1** (transport switch) — both branches resolve to `SDKTransport`. The remaining question requires runtime trace: when parent omits `system_prompt`, what argv does claude-agent-sdk actually emit, and does the resulting claude.exe invocation hang or work? The static diff can't answer that — Phase 2 was needed.

---

## 2. Phase 2 — Tier 3 trace on parent branch

Same Tier 3 harness (`/tmp/orbital-tier4/tier3_instrument.py` — extended Phase 1 version that logs full argv, stdio, env, returncode). Same env hygiene (`CLAUDECODE=''`, `CLAUDE_CODE_*` stripped via shell). Same workspace `D:\repro-smoke` seeded with `orbital/PROJECT_STATE.md` / `DECISIONS.md` / `LESSONS.md`.

### Parent trace (working — `Message sent to claude-code` in 5s elapsed)

```
[T3 20.463 TRANSPORT.connect] ENTER
[T3 20.463 SUBPROC.spawn] PRE argv_len=2   ← version-probe spawn
[T3 20.463 SUBPROC.argv] [0] 'C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD'
[T3 20.463 SUBPROC.argv] [1] '-v'
[T3 20.477 SUBPROC.spawn] POST pid=10472 returncode=None
[T3 20.598 SUBPROC.spawn] PRE argv_len=14  ← SDK session spawn
[T3 20.598 SUBPROC.argv] [0] 'C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD'
[T3 20.598 SUBPROC.argv] [1] '--output-format'
[T3 20.598 SUBPROC.argv] [2] 'stream-json'
[T3 20.598 SUBPROC.argv] [3] '--verbose'
[T3 20.598 SUBPROC.argv] [4] '--system-prompt'        ← REPLACE flag, but value is empty
[T3 20.598 SUBPROC.argv] [5] ''                        ← EMPTY STRING
[T3 20.598 SUBPROC.argv] [6] '--permission-prompt-tool'
[T3 20.598 SUBPROC.argv] [7] 'stdio'
[T3 20.598 SUBPROC.argv] [8] '--permission-mode'
[T3 20.598 SUBPROC.argv] [9] 'default'
[T3 20.598 SUBPROC.argv] [10] '--setting-sources'
[T3 20.598 SUBPROC.argv] [11] ''
[T3 20.598 SUBPROC.argv] [12] '--input-format'
[T3 20.598 SUBPROC.argv] [13] 'stream-json'
[T3 20.598 SUBPROC.spawn] cwd='D:\\repro-smoke'
[T3 20.598 SUBPROC.spawn] stdin=-1 stdout=-1 stderr=None
[T3 20.598 SUBPROC.spawn] env keys count=62   ← same set as spec 3
[T3 20.598 SUBPROC.env] CLAUDECODE=''
[T3 20.598 SUBPROC.env] CLAUDE_CODE_ENTRYPOINT='sdk-py'
... (same CLAUDE_* keys as spec 3)
[T3 20.606 SUBPROC.spawn] POST pid=9728 returncode=None
[T3 20.606 TRANSPORT.connect] EXIT pid=9728
[T3 20.606 QUERY.start] ENTER
[T3 20.606 QUERY.start] EXIT
[T3 20.606 QUERY.initialize] ENTER
[T3 20.606 QUERY.ctrl_req] ENTER subtype=initialize timeout=60.0
[T3 20.606 TRANSPORT.write] ENTER size=113
[T3 20.606 TRANSPORT.write] POST
[T3 25.624 QUERY.ctrl_req] EXIT subtype=initialize  ← SUCCESS at ~5s
[T3 25.624 QUERY.initialize] EXIT
[T3 25.626 TRANSPORT.write] ENTER size=168          ← user prompt sent next
[T3 25.627 TRANSPORT.write] POST
```

### Spec 3 trace (failing — captured previously in `PHASE2-DIAGNOSIS-claude-no-response.md`)

```
[T3  4.030 TRANSPORT.connect] ENTER
[T3  4.031 SUBPROC.spawn] PRE argv_len=2   ← version-probe spawn (same shape as parent)
[T3  4.031 SUBPROC.argv] [0] 'C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD'
[T3  4.031 SUBPROC.argv] [1] '-v'
[T3  4.038 SUBPROC.spawn] POST pid=5804 returncode=None
[T3  4.159 SUBPROC.spawn] PRE argv_len=14  ← SDK session spawn
[T3  4.159 SUBPROC.argv] [0] 'C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD'
[T3  4.159 SUBPROC.argv] [1] '--output-format'
[T3  4.159 SUBPROC.argv] [2] 'stream-json'
[T3  4.159 SUBPROC.argv] [3] '--verbose'
[T3  4.159 SUBPROC.argv] [4] '--append-system-prompt'   ← APPEND flag (Option 1)
[T3  4.159 SUBPROC.argv] [5] '<2080-char orbital prompt>'  ← FULL TEXT
[T3  4.159 SUBPROC.argv] [6] '--permission-prompt-tool'
[T3  4.159 SUBPROC.argv] [7] 'stdio'
[T3  4.159 SUBPROC.argv] [8] '--permission-mode'
[T3  4.159 SUBPROC.argv] [9] 'default'
[T3  4.159 SUBPROC.argv] [10] '--setting-sources'
[T3  4.159 SUBPROC.argv] [11] ''
[T3  4.159 SUBPROC.argv] [12] '--input-format'
[T3  4.159 SUBPROC.argv] [13] 'stream-json'
[T3  4.159 SUBPROC.spawn] cwd='D:\\repro-smoke'
[T3  4.159 SUBPROC.spawn] stdin=-1 stdout=-1 stderr=None
[T3  4.159 SUBPROC.spawn] env keys count=62
[T3  4.168 SUBPROC.spawn] POST pid=19676 returncode=None
[T3  4.168 TRANSPORT.connect] EXIT pid=19676
[T3  4.168 QUERY.start] ENTER / EXIT
[T3  4.168 QUERY.initialize] ENTER
[T3  4.168 QUERY.ctrl_req] ENTER subtype=initialize timeout=60.0
[T3  4.168 TRANSPORT.write] ENTER size=113
[T3  4.168 TRANSPORT.write] POST
[T3 64.170 QUERY.ctrl_req] EXC Exception: Exception('Control request timeout: initialize')  ← TIMEOUT at 60.013s
[T3 64.170 QUERY.initialize] EXC
```

### Side-by-side comparison

| Tier 3 event | Parent (working) | Spec 3 (failing) | Diff |
|---|---|---|---|
| `TRANSPORT.connect ENTER` | t=20.463 | t=4.030 | same shape; just different daemon-boot offset |
| version-probe `SUBPROC.spawn PRE argv_len=2` | `[claude.CMD, -v]` | `[claude.CMD, -v]` | **byte-identical** |
| version-probe `POST pid/returncode` | pid=10472, rc=None | pid=5804, rc=None | same shape (pid varies per run) |
| SDK-session `SUBPROC.spawn PRE argv_len=14` | see table below | see table below | **differs in argv[4..5] only** |
| `cwd` | `'D:\\repro-smoke'` | `'D:\\repro-smoke'` | **byte-identical** |
| `stdin / stdout / stderr` | `-1 / -1 / None` | `-1 / -1 / None` | **byte-identical** |
| `env keys count` | 62 | 62 | **byte-identical** |
| `CLAUDECODE` env | `''` | `''` | same |
| `CLAUDE_CODE_*` inherited env vars | `CLAUDE_CODE_ENTRYPOINT='sdk-py'` + the same `CLAUDE_CODE_EXECPATH`, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, `CLAUDE_CODE_SESSION_ID`, `CLAUDE_AGENT_SDK_VERSION`, `CLAUDE_EFFORT`, `AI_AGENT` | identical set | **byte-identical** |
| creationflags / startupinfo | not in kwargs (anyio defaults) | not in kwargs | same |
| `SDK-session SUBPROC.spawn POST pid/returncode` | pid=9728, rc=None (stayed alive) | pid=19676, rc=None (stayed alive) | same shape |
| `TRANSPORT.connect EXIT` | t+0 of POST | t+0 of POST | same |
| `QUERY.start ENTER/EXIT` | instant | instant | same |
| `QUERY.initialize ENTER` | t=20.606 | t=4.168 | same shape |
| `QUERY.ctrl_req ENTER subtype=initialize timeout=60.0` | yes | yes | same |
| `TRANSPORT.write ENTER size=113` (initialize payload) | yes | yes | **byte-identical** |
| `TRANSPORT.write POST` (initialize stdin written) | yes | yes | same |
| **stdout response from claude.exe** | **arrives** in 5.018s (parent) | **never arrives** (60.013s silence) | **THE DIFFERENCE IS HERE** |
| `QUERY.ctrl_req EXIT` (success) | t=25.624 (5.018s after write) | never | divergence |
| `QUERY.ctrl_req EXC TimeoutError` | never | t=64.170 (60.002s after write) | divergence |
| Subsequent `TRANSPORT.write ENTER size=168` (user prompt) | t=25.626 | never | parent proceeds; spec 3 dies |

### The decisive delta — argv[4] and argv[5]

| | Parent | Spec 3 |
|---|---|---|
| **argv[4]** | `'--system-prompt'` | `'--append-system-prompt'` |
| **argv[5]** | `''` (empty string, 0 chars) | `'<2080-char orbital inheritance prompt>'` |

These are the ONLY two argv elements that differ. Everything else — argv[0-3], argv[6-13], cwd, stdio, env, creationflags — is byte-identical between the working parent run and the failing spec 3 run.

The reason for the difference is in claude-agent-sdk's argv builder at `_internal/transport/subprocess_cli.py:170-180`:

```python
if self._options.system_prompt is None:
    cmd.extend(["--system-prompt", ""])                              # ← PARENT path
elif isinstance(self._options.system_prompt, str):
    cmd.extend(["--system-prompt", self._options.system_prompt])    # ← spec 3 pre-Option-1
elif (system_prompt.get("type") == "preset"
      and "append" in self._options.system_prompt):
    cmd.extend(["--append-system-prompt",
                self._options.system_prompt["append"]])              # ← spec 3 post-Option-1 (this run)
```

Parent passes `system_prompt=None` to `ClaudeAgentOptions` (because spec 3's `_start_from_registry` inheritance block doesn't exist on parent). The SDK then emits the first branch — `--system-prompt ""` — and claude.exe responds. Spec 3 passes the rendered template either as a plain string (pre-Option-1) or as the preset/append dict (post-Option-1). Both forms result in argv with a non-empty value after `--system-prompt` or `--append-system-prompt`. Both forms hang.

---

## 3. Classification

**Row 2 of the spec's Phase 3 pattern table fires: "Argv composition regression."**

Quoted evidence from the comparison:

- Parent argv[4-5]: `'--system-prompt', ''` → `QUERY.ctrl_req EXIT` at t=25.624 (5.018s after TRANSPORT.write POST)
- Spec 3 argv[4-5]: `'--append-system-prompt', '<2080-char text>'` → `QUERY.ctrl_req EXC Control request timeout: initialize` at t=64.170 (60.002s after TRANSPORT.write POST)

Every other Tier 3 event (cwd, stdio handles, env keys, creationflags, the initial 113-byte stdin write, the version-probe spawn) is byte-identical between the two runs. The argv[4-5] delta is the **only** code-path-visible difference; the dispatch-time-visible difference is whether claude.exe emits a `control_response` for the `initialize` request.

The spec's implication for row 2: *"Fix is to remove or modify that arg in SDKTransport's invocation when system_prompt is set."* — surfaced for joint decision per spec § DO NOT (no fix proposed in this doc).

### What this classification means in plain terms

When orbital does not render a sub-agent inheritance prompt (parent branch behavior), the claude-agent-sdk's default `--system-prompt ""` argv is used, and claude.exe responds to the SDK's initialize control_request within ~5 seconds, dispatch succeeds end-to-end.

When orbital renders any non-empty prompt and routes it through `ClaudeAgentOptions.system_prompt` (spec 3 behavior, with or without Option 1 applied), the SDK emits an argv containing a non-empty `--system-prompt <text>` or `--append-system-prompt <text>` element, and claude.exe on Windows enters the silent-stdout state documented in `PHASE2-DIAGNOSIS-claude-no-response.md`. It does not emit a `control_response` for the SDK's initialize request, and the 60-second SDK timeout fires.

### Reconciling with `PHASE2-DIAGNOSIS` Phase 2.7

`PHASE2-DIAGNOSIS-claude-no-response.md` concluded the hang's trigger is "stdin open + `--output-format stream-json`" based on Phase 2.7 C7 (explicit stdin close) and C8 (file-redirect EOF) both working with claude.CMD invoked under `--output-format stream-json --verbose` plus a stdin-side EOF. That conclusion holds in isolation — those two probes did NOT include a non-empty `--system-prompt` or `--append-system-prompt` value. They invoked claude.CMD with the minimum stream-json args ONLY.

The diff investigation finds a finer trigger: **a non-empty `--system-prompt` or `--append-system-prompt` value in argv** is sufficient to make claude.exe hang silently even when other args (output-format, verbose, stream-json input/output, permission tooling) are present, AND parent's identical-other-argv path with `--system-prompt ""` works.

These two observations are consistent — they describe the same Windows-side behavior from different argv vantage points. The diff investigation more directly identifies the specific argv element that distinguishes parent's working invocation from spec 3's hanging one, which is what the spec asked for.

---

## 4. Concrete git diff (dispatch-path files between branches)

```
$ git diff feature/render-chat-variant-a..worktree-agent-a9e24fbde848229ba \
    -- agent_os/agent/transports/ \
       agent_os/daemon_v2/sub_agent_manager.py \
       agent_os/agents/manifests/claude_code.yaml
```

### `agent_os/agents/manifests/claude_code.yaml` — no diff

### `agent_os/daemon_v2/sub_agent_manager.py` — adds `system_prompt` plumbing + `_start_from_registry` inheritance block

```diff
@@ class SubAgentManager:
     def __init__(self, process_manager, adapter_configs: dict | None = None,
                  platform_provider=None, registry=None, setup_engine=None,
-                 project_store=None, lifecycle_observer=None):
+                 project_store=None, lifecycle_observer=None,
+                 ws_manager=None):
+        # CLAUDE.md interference state (banner suppression) — see DECISIONS-from-followup.md
+        self._claudemd_warning_state: dict[tuple[str, str], str] = {}
+        self._ws_manager = ws_manager
         ...

-    def _resolve_transport(self, manifest, config_dict, autonomy=None):
-        """Resolve the appropriate transport for a manifest."""
+    def _resolve_transport(self, manifest, config_dict, autonomy=None, system_prompt: str | None = None):
+        """Resolve the appropriate transport for a manifest. system_prompt forwarded to SDK/Pipe."""
         transport_hint = getattr(manifest.runtime, 'transport', 'auto')
         mode = manifest.runtime.mode
         ...
         if transport_type == "sdk":
             try:
                 from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
                 if HAS_SDK:
-                    return SDKTransport(autonomy=autonomy)
+                    return SDKTransport(autonomy=autonomy, system_prompt=system_prompt)
             except ImportError:
                 pass
             from agent_os.agent.transports.pipe_transport import PipeTransport
-            return PipeTransport(config=self._get_pipe_config(manifest.slug))
+            return PipeTransport(
+                config=self._get_pipe_config(manifest.slug),
+                system_prompt=system_prompt, agent_slug=manifest.slug,
+            )
         elif transport_type == "pipe":
             from agent_os.agent.transports.pipe_transport import PipeTransport
-            return PipeTransport(config=self._get_pipe_config(manifest.slug))
+            return PipeTransport(
+                config=self._get_pipe_config(manifest.slug),
+                system_prompt=system_prompt, agent_slug=manifest.slug,
+            )

+    def _maybe_emit_claudemd_warning(self, project_id: str, workspace: str) -> None:
+        """detect+warn passive surface for workspace CLAUDE.md interference"""
+        ...

+    def dismiss_claudemd_warning(self, project_id: str, content_hash: str) -> None:
+        ...

     async def _start_from_registry(self, project_id: str, handle: str, depth: int = 0) -> str:
         ...
+        # NEW: render inheritance prompt + ensure MEMORY.md + emit CLAUDE.md banner
+        system_prompt: str | None = None
+        if workspace:
+            try:
+                from agent_os.agent.sub_agent_prompt import (
+                    ensure_memory_md, render_sub_agent_prompt,
+                )
+                enabled_sub_agents = project.get("enabled_sub_agents") or [...]
+                ensure_memory_md(workspace, handle)
+                system_prompt = render_sub_agent_prompt(
+                    workspace=workspace, namespace=None,
+                    agent_slug=handle, enabled_sub_agents=enabled_sub_agents,
+                )
+            except Exception:
+                logger.exception(...)
+            self._maybe_emit_claudemd_warning(project_id, workspace)
-        transport = self._resolve_transport(manifest, config_dict, autonomy=autonomy)
+        transport = self._resolve_transport(
+            manifest, config_dict, autonomy=autonomy, system_prompt=system_prompt,
+        )
         ...
```

### `agent_os/agent/transports/sdk_transport.py` — adds `system_prompt` plumbing into `ClaudeAgentOptions`

```diff
-    def __init__(self, autonomy: "Autonomy | None" = None):
+    def __init__(self, autonomy: "Autonomy | None" = None, system_prompt: str | None = None):
         ...
+        self._system_prompt: str | None = system_prompt

     async def start(self, command, args, workspace, env=None):
         ...
-        options = ClaudeAgentOptions(
+        options_kwargs: dict = dict(
             cwd=workspace, permission_mode="default",
             can_use_tool=self._handle_permission, cli_path=command or None, env=sdk_env,
         )
+        if self._system_prompt is not None:
+            options_kwargs["system_prompt"] = self._system_prompt
+        options = ClaudeAgentOptions(**options_kwargs)
         self._client = ClaudeSDKClient(options=options)
         await self._client.connect()
```

### `agent_os/agent/transports/pipe_transport.py` — adds tempfile-based `--append-system-prompt-file` injection

```diff
+import logging
+import uuid
+logger = logging.getLogger(__name__)

 class PipeTransport(AgentTransport):
-    def __init__(self, config: PipeTransportConfig | None = None):
+    def __init__(self, config=None, system_prompt=None, agent_slug=None):
         ...
+        self._system_prompt: str | None = system_prompt
+        self._agent_slug: str = agent_slug or "agent"

+    def _write_system_prompt_tempfile(self) -> str | None:
+        """Write self._system_prompt to {workspace}/orbital/.tmp/<unique>.txt"""
+        ...

     async def send(self, message: str) -> str | None:
         args = list(self._args)
         ...
+        system_prompt_path: str | None = self._write_system_prompt_tempfile()
+        if system_prompt_path is not None:
+            args.extend(["--append-system-prompt-file", system_prompt_path])
         ...
         try:
             ... subprocess.run ...
         finally:
+            if system_prompt_path is not None:
+                try: os.remove(system_prompt_path)
+                except OSError: pass
```

The PipeTransport diff is informational — PipeTransport is not on the failing dispatch path here.

---

## 5. Fix path implied (NOT a fix proposal)

Row 2 of the spec's classification table — "Argv composition regression" — implies:

> *Fix is to remove or modify that arg in SDKTransport's invocation when system_prompt is set.*

Surfacing the implication only. The "arg in question" is the non-empty `--system-prompt <text>` or `--append-system-prompt <text>` element that the claude-agent-sdk emits into argv whenever `ClaudeAgentOptions.system_prompt` is non-None. Parent avoids the bug because `system_prompt=None` causes the SDK to emit `--system-prompt ""` (empty), which claude.exe on Windows tolerates.

Any concrete fix is your decision; this doc does not pick among the options. Possible directions consistent with the trace evidence (listed for completeness, not recommendation):

- A path that gets the inheritance text in front of claude WITHOUT routing it through `ClaudeAgentOptions.system_prompt` (e.g., injecting the prompt content via a different SDK mechanism, or via a one-time message after init).
- A claude-agent-sdk-level change that avoids emitting `--system-prompt <non-empty-text>` on Windows when streaming mode is enabled (likely an upstream report).
- A claude.exe-level fix to flush stdout regardless of whether a non-empty system-prompt was specified (also upstream).
- Falling back to parent's behavior (no `ClaudeAgentOptions.system_prompt`) and accepting that the sub-agent inheritance prompt won't be injected via SDKTransport on Windows.

Surfaced for joint decision. No recommendation.

---

## Cleanup status

- Tier 3 import in `agent_os/api/app.py` (parent worktree): to be reverted before the doc is final.
- `tier3_instrument.py` at parent worktree root: to be removed before the doc is final.
- Canonical harness at `/tmp/orbital-tier4/tier3_instrument.py`: preserved for re-use.
- Daemon: to be stopped before report.
- No code committed during this investigation.

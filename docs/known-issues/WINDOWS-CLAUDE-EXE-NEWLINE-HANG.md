# Windows claude.exe argv-newline hang

**Affects:** claude-code 2.1.138 on Windows under streaming SDK mode.
**Workaround in:** `agent_os/agent/sub_agent_prompt.py`
(`render_sub_agent_prompt()` flattens `\n` -> `; ` in the rendered output)
and `agent_os/agent/transports/sdk_transport.py` (sends
`system_prompt` as `{type:preset, preset:claude_code, append:<text>}`).

## The bug

When `claude.exe` is invoked with
`--output-format stream-json --input-format stream-json --verbose
--permission-prompt-tool stdio --permission-mode default
--setting-sources ''`, and the value passed for
`--system-prompt` (or `--append-system-prompt`) contains one or more
newline characters (`\n`, LF, U+000A), the process emits zero bytes
of stdout for at least 60 seconds (the SDK timeout) and the
`Query.initialize` control request never receives a response. With
newlines removed from the same argv value, the process responds in
~5 seconds. The trigger is the LF character only; backslashes,
em-dashes, Windows paths, and orbital-specific word content are
all ruled out by a 7-cell content bisect.

In orbital this surfaced as `Error: SDK query failed: Control request
timeout: initialize` whenever a sub-agent was dispatched into a project
with the inheritance template active.

## The workaround

Two changes in tandem:

1. **Newline flatten.** `render_sub_agent_prompt()` replaces every `\n`
   in the rendered template with `; ` (semicolon space) before
   returning. Section breaks are preserved as visible separators.
2. **APPEND semantic.** `SDKTransport` forwards `system_prompt` as a
   preset/append dict so the rendered text is sent to claude.exe as
   `--append-system-prompt <text>` instead of `--system-prompt
   <text>` (REPLACE), preserving claude-code's default system prompt.

Either change alone is insufficient: flatten alone reverts to REPLACE
semantics and loses claude-code's default prompt; APPEND alone still
ships newlines in argv and triggers the hang.

## When to remove

Remove the flatten when claude-code addresses the argv-newline bug
upstream and Orbital pins to that version or higher. The APPEND-dict
shape is the preferred shape regardless of the hang and should stay.

## References

- `docs/investigations/TRIGGER-orbital-prompt-content.md` — LF-isolated bisect
- `docs/investigations/BOUNDARY-claude-exe-hang.md` — length ruled out
- `docs/investigations/DIFF-parent-vs-spec3-dispatch.md` — argv-composition diff
- `docs/investigations/PHASE2-DIAGNOSIS-claude-no-response.md`
- `docs/investigations/PHASE3-output-format-probe.md`
- `docs/investigations/TRACE-windows-dispatch-bug.md` — original W1 trace
- Implementation: `agent_os/agent/sub_agent_prompt.py`,
  `agent_os/agent/transports/sdk_transport.py`
- Regression test:
  `tests/unit/test_sub_agent_inheritance.py::TestRenderSubAgentPromptPaths::test_render_sub_agent_prompt_contains_no_newlines`

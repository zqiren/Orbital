# Keyless x Workbench: Mac results, ACP agent (dsh)

**D3 answer:** yes, an ACP model called `dispatch`: dsh listed the orbital MCP tool and called it, and the child's result (Claude Code PONG, or Codex's usage-limit error) came back as its next turn.

- Commit under test: 5e69b77 (batch/1109). Date: 2026-11-09. Companion to note-keyless-results-mac.md (Claude Code and Codex).
- dsh: 0.2.1-rc.3 at `/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh`. Route: native dsh mirrored route, provider opencode-go (api anthropic-messages, baseURL https://opencode.ai/zen/go, header x-opencode-session: orbital), model deepseek-v4.1-flash, key in env `OPENCODE_API_KEY` (redacted).
- Daemon command (keyless, port 8590, same data dir as the first pass):
  `DSH_HOME=.../run/1109-keyless/dsh-home OPENCODE_API_KEY=<redacted> PYTHONPATH=.../memkeyring PYTHON_KEYRING_BACKEND=memkeyring.MemoryKeyring AGENT_OS_PORT=8590 .venv/bin/python run/dev-daemon.py --worktree .../1109-batch --port 8590 --data-dir .../run/1109-keyless/data`

## Cases

Added after the lead supplied a key. Daemon restarted once with `DSH_HOME=.../run/1109-keyless/dsh-home` (desktop profile `cordis.patch.yml` with the opencode-go provider row and default model deepseek-v4.1-flash) and `OPENCODE_API_KEY` from dsh.env (never printed). Same data dir and port. dsh 0.2.1-rc.3 reported installed and ready; first pinned send answered PONG. ACP = dsh for every row below. Evidence: ws-p, ws-q, ws-r under the run dir, out-dsh1.txt, out-D3-dsh.txt, out-D3b-dsh.txt, out-C4dsh.txt.

- A3-dsh PARTIAL. `start-onboarding` 201 `runner: "dsh"`; the interview reply stops with no files. After my answer (not after a confirm) dsh wrote project_goals.md and INDEX.md at once, and on the bare "ok" it created `groceries.md` with five invented items instead of waiting for a go. It does not follow the confirm-first flow Claude Code and Codex follow.
- A4-dsh PASS. `cold-start-scan` 201 `runner: "dsh"`; read the folder, drafted goals, asked three questions, "Say ok and I'll write the three files"; nothing written.
- A5-dsh BLOCKED. The import-scan agent is chosen by the daemon (Claude Code); dsh cannot be selected.
- B3-dsh PASS (UI send on dsh-proj, `target: dsh`, session `pinned_target: dsh`, answered PONG). B6-dsh PASS (upload, new-session, inject with attachments; answered "BANANA-42 is the first line"). B7-dsh PASS (Quick Tasks, Agent chip DeepSeek Harness; transcript under data/scratch/orbital/sub_agents/dsh).
- C1-dsh FAIL. With the same prompt as the others, and then once more with a plainer request, dsh did not write `orbital/ASKS.md`. It showed it knew the grammar ("no existing ask line to close with `- done <id>`") but reasoned that nothing had been decided and asked me which colour I chose instead. No ask card appeared.
- C4-dsh PASS. SIGKILL on the dsh child while `sleep 80` ran: `dsh stopped with error: Error: Connection closed`. Re-pin to Claude Code and a new send continued with the recap ("earlier you asked me to run sleep 80"). The card label was not captured before it was overwritten by the continuation.
- C5-dsh PASS. Memory-record turns ran after every dsh run and wrote PROJECT_STATE.md (and a LESSONS.md note after the failed dispatch); no bubble. C2/C6/C7 not repeated for dsh.
- D3 PASS (W2 resolved: an ACP model DID call `dispatch`). dsh listed the tool (log: `ListToolsRequest` at each spawn) and called it (`CallToolRequest`, 20:09:51 and 20:11:22). Dispatch to Claude Code: child ran, `worker_result` row, dsh's next turn said "Claude Code came back exactly as requested". Dispatch to Codex: the child failed with "You've hit your usage limit ... try again at 9:29 PM" (the owner's ChatGPT plan is now at its limit after this run); the `worker_result` carried the error and dsh relayed it correctly and recorded a lesson.
- Real ~/.dsh: `~/.dsh` mtime Oct 26 11:54:56 (unchanged). `~/.dsh/profiles/acp/cordis.yml` mtime is Nov 9 19:54:56, later than the 19:04 the lead recorded, and it was already 19:54:56 when I first stat'ed it, before the DSH_HOME daemon started; it did not change during the dsh run. The only dsh activity before that was my earlier daemon's agent probes (`/agents/available`, Settings), which may rewrite the acp profile; I cannot tell from here whether that or the owner's own app did. The scratch `dsh-home/` holds `profiles/acp` and `storages` created by the dsh spawns, so the spawns did use the scratch home.
- Update to the lists above: D3 is no longer blocked; the ACP rows for A3, A4, B3, B6, B7, C4, C5 are now covered by dsh; C1-dsh is a FAIL; A5-dsh stays BLOCKED. Kimi, MiniMax and Cursor remain refusing turns.
- HOME files after the dsh pass: same three mtimes as before (Nov 9 16:13:58, 16:13:27, Nov 7 12:35:33).

## HOME and ~/.dsh check (end of run)
```
Oct 26 11:54:56 2026 /Users/keanezhou/.dsh
Nov  9 19:54:56 2026 /Users/keanezhou/.dsh/profiles/acp/cordis.yml
Nov  9 16:13:58 2026 /Users/keanezhou/.orbital/sub_agent_config.json
Nov  9 16:13:27 2026 /Users/keanezhou/orbital/user_memory.md
Nov  7 12:35:33 2026 /Users/keanezhou/orbital/user_preferences.md
```
The three Orbital HOME files match home-mtimes-before.txt. The acp cordis.yml mtime was already 19:54:56 before the DSH_HOME daemon started and did not change during the dsh run.

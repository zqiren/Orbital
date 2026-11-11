# Fresh-install test

Replays a brand-new user's first minutes on a machine that has never seen
Orbital, against the **CI-built installer** (never a local build: local builds
are ad-hoc signed without the hardened runtime, and that gap has shipped
regressions). One scenario, `scenario.mjs`, runs on macOS and Windows.

## What it checks

| step | what a new user does | asserted on |
|---|---|---|
| 1 precondition | — | no Orbital app, data dir, `~/orbital`, logs; no Orbital process |
| 0 rig | had Codex + dsh installed already | Codex `config.toml` → OpenCode Go (Responses, `gpt-6-luna`), `codex login --with-api-key`; a localhost header forwarder for dsh |
| 2 install | macOS: mount DMG, `ditto` to /Applications. Windows: Inno `/VERYSILENT` | app present; version; codesign/spctl facts |
| 3 launch | opens the app normally (not `--smoke-test`) | daemon answers on the port in `~/orbital/daemon.pid.meta`; SPA served; OS window list + desktop screenshot |
| 4 keyless | wizard → skip key → New project → pin Codex → "PONG" | Codex reply row with PONG in the session; Codex bubble in the chat |
| 5a card | Settings → Add provider → OpenCode Go, key, `deepseek-v4-pro`, Test, Save | test result text; `/settings/cards` default card has a key |
| 5b workers | Settings → Sub-agents → dsh key, base URL, model | `/settings/sub-agents` codex + dsh `ready` |
| 5c manager | new project, message Orbital | a `management` reply row, rendered |
| 5d dispatch | ask the manager to send "PONG" to codex and dsh | each worker's result row with PONG |
| 6 browser | ask the manager to open example.com with its browser | a `browser` tool call whose result shows "Example Domain" |
| 7 upgrade | (separate machine) previous release, a project + Codex turn, quit, install new over it, relaunch | no wizard; same project id; session rows intact; old reply renders |
| 8 evidence | — | screenshots per step, daemon/app logs, session JSONL + chat dumps, **scrubbed of the key** (`scrub-report.json`); any surviving copy fails the run |

Results: `results.md` / `results.json` in the evidence dir, every step
`passed`, `failed` or `not_verified` with the reason.

The UI is driven with Playwright in a browser pointed at the installed app's
own daemon (WebKit on macOS, the engine of the app's WKWebView; Chromium on
Windows, the engine of WebView2). The native window is opened too (the app is
launched normally) and recorded from the OS side, but Playwright cannot
attach to an embedded WKWebView/WebView2.

### Things a Go user has to do by hand (and the scenario does)

- **Codex model.** Orbital picks Codex's catalog default (`isDefault` from
  `model/list`, today `gpt-6-astra`) whenever no model is chosen in Orbital,
  which overrides `model` in `~/.codex/config.toml`. OpenCode Go answers that
  id with "Model is unavailable", so the scenario sets `gpt-6-luna` in
  Settings → Sub-agents → Codex → Model (`--codex-model ''` skips it).
- **dsh session header.** Go rejects any request without
  `x-opencode-session`; dsh's `DEEPSEEK_BASE_URL` route cannot add headers, so
  `rig/go-header-proxy.mjs` (test rig only, never product code) adds it and
  the base URL saved in Settings points at `http://127.0.0.1:18787`.
- **dsh model.** dsh's stock default `deepseek-flash` is not a Go model, and
  v0.21.1's Settings → Sub-agents → DeepSeek Harness → Model dropdown offers
  only `deepseek-official/deepseek-v4-flash` (not in dsh 0.2.2's catalog
  either). So the rig sets the model where a DSH Desktop user would: the
  `agent-default-model` row of `~/.dsh/profiles/desktop/cordis.patch.yml`
  (`deepseek-official/deepseek-v4-pro`), which Orbital mirrors when no model
  is chosen in Orbital (`--dsh-model` sets it in Orbital's UI instead).
- **Telemetry.** An app opened through LaunchServices does not inherit the
  shell's env, so on macOS the scenario passes `AGENT_OS_TELEMETRY_DISABLED`
  to the GUI session with `launchctl setenv` before `open` (the Tart rig and
  CI both set it); the daemon log then says "telemetry sender not started".
- MiniMax Code (`mcode`) has no scriptable credential (interactive
  `mcode login` only), so dsh is the ACP agent.
- No Claude Code anywhere (owner's rule).

## Run it locally (Tart macOS VM)

Never on the host Mac: the owner's real Orbital lives there. Each run clones a
golden VM, runs inside it, copies scrubbed evidence out and deletes the clone.

```bash
# once: golden image (Node 24, Codex, dsh, Playwright WebKit; no Orbital)
bash tests/fresh_install/tart/prepare-golden.sh

# each run (key on stdin into the VM, never on a command line)
set -a; source <file with ORBITAL_TEST_OPENCODE_KEY>; set +a
bash tests/fresh_install/tart/run.sh --tag v0.21.1 --out <evidence root>
bash tests/fresh_install/tart/run.sh --tag v0.21.1 --mode upgrade --out <evidence root>   # v0.21.0 -> v0.21.1
bash tests/fresh_install/tart/run.sh --run-id <CI run id> --out <evidence root>          # a CI artifact
```

Tart 2.x on Apple Silicon; the base image is
`ghcr.io/cirruslabs/macos-sequoia-base:latest` (admin/admin, SSH on; the rig
authorizes its own key from `FRESH_STATE_DIR`, default
`~/.cache/orbital-fresh-install`). `--keep` leaves a failed clone running for
debugging. A run costs roughly $0.10–0.30 of the Go plan's cap; don't loop it.

## What CI runs

`.github/workflows/ci.yml` jobs `fresh-install-macos` and
`fresh-install-windows`, each a matrix of `fresh` and `upgrade`, after
`build-macos` / `build-windows`, on the **same run's** installer artifact.
Only on `v*` tags and `workflow_dispatch` (real key, ~15 min, and secrets must
never reach pull requests). The key is the repo secret `OPENCODE_GO_TEST_KEY`.
Evidence uploads as `fresh-install-evidence-<os>-<scenario>` only when the
scrub check passes. On macOS the job creates an unlocked default keychain for
the runner user (a logged-in user's keychain is unlocked; the runner's login
keychain password is unknown); the app's keyring backend is not touched.

## Adding the Windows host (local VirtualBox VM, later)

The scenario is already OS-agnostic (`lib/platform.mjs` has the Windows
install/launch/quit/paths). A Windows host script needs to mirror
`tart/run.sh`: restore a clean snapshot (Node 24 with npm prefix
`%APPDATA%\npm`, `npm i -g @openai/codex @deepseek-ai/dsh@0.2.2-rc.1`,
`npx playwright install chromium`, no Orbital), copy the `.exe` and this
directory in over SSH (Tailscale), pipe the key on stdin, run
`node scenario.mjs --installer <exe> --evidence <dir> --require-key`, copy the
evidence out, run `node scrub.mjs <dir> --require-key` on the host, and revert
the snapshot.

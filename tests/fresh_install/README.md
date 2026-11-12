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

## Run it locally (VirtualBox Windows VM)

The Windows twin of the Tart rig, in `vbox/`. Never on the host PC itself:
Orbital only ever runs inside VM `orbital-fresh-win` (Windows 11 x64 from the
**Simplified Chinese** ISO, Home edition (image 1, `Core`), unactivated; EFI + TPM 2.0,
4 vCPU, 8 GB, 64 GB dynamic disk), which is rolled back to snapshot `clean`
before and after every run. Chinese Home is the point: it matches the users
and catches locale bugs (code page 936, translated group names) that
GitHub's English Server image misses.

The guest has one local admin `tester` with auto-logon to the desktop, no
sleep/lock/screen saver, Windows Update paused (Defender on), Win32-OpenSSH
with **key-only** login (the rig key goes in
`C:\ProgramData\ssh\administrators_authorized_keys`, ACL set by SID), Node 24
with npm prefix `%APPDATA%\npm` (and `NPM_CONFIG_PREFIX`, as ci.yml does),
`@openai/codex`, `@deepseek-ai/dsh@0.2.2-rc.1` and Playwright 1.60.0's
Chromium; no Orbital, no `~\.codex` / `~\.dsh`. NAT with one port forward,
`127.0.0.1:2222 -> 22`, on the host's loopback only.

```powershell
# once, as the Windows user that will own the VM and the runner (one UAC prompt
# for VirtualBox, then about an hour unattended):
tests\fresh_install\vbox\bootstrap.cmd D:\orbital-winvm-cache
#   = install-host.ps1 (elevated: VirtualBox 7.2 + Extension Pack, checksums verified)
#   + prepare-vm.ps1   (VBoxManage unattended install with autounattend-template.xml,
#                       guest-firstlogon.ps1, guest-provision.ps1 over SSH, snapshot clean)

# each run (Windows PowerShell 5.1; key in the env, never on a command line)
.\tests\fresh_install\vbox\run.ps1 -Installer <new .exe> -Mode fresh -Out <evidence dir>
.\tests\fresh_install\vbox\run.ps1 -Installer <new .exe> -Mode upgrade -OldInstaller <old .exe> -Out <dir>
.\tests\fresh_install\vbox\run.ps1 -RunId <CI run id> -Out <dir>      # or -Tag vX.Y.Z; downloads with gh
```

**Why the desktop shell starts the scenario.** Anything started over SSH on
Windows runs outside the logged-in desktop, so the Orbital window would never
appear and the scenario's desktop screenshot / window check
(`desktopScreenshot`, `nativeWindowFacts`) would come back empty. A process
that Task Scheduler starts is in the desktop but has a logon session of its
own, where Credential Manager writes fail (CredWrite 1312): Orbital started
that way can't save a provider card, and the installer's `--setup-sandbox`
can't store the sandbox password. So `run.ps1` pipes the key over SSH stdin
(no BOM; the guest decodes raw bytes as UTF-8, since `[Console]::In` would
read them as GBK) into `guest-start.ps1`. It writes the key to a file only
`tester` (and SYSTEM) can read, and registers a one-shot task that only
asks tester's desktop explorer to open `guest-run.cmd`. Explorer starts it
with the shell's own unelevated token, as a double-click would. The Inno
installer elevates itself as it does for a user; the guest's admin consent
is set to "elevate without prompting", standing in for the UAC "Yes". Both
sides log a key fingerprint (length + 8 hex of sha256, never the key).
`guest-run.ps1` reads the key into its env, deletes the file,
runs `npm ci` and `node scenario.mjs --require-key` with
`AGENT_OS_TELEMETRY_DISABLED=1`, and writes `console.log`, `exitcode` and
`done` into `C:\fi\state`; the host polls and streams `console.log`. The
evidence (scrubbed inside by the scenario) is copied to `-Out`, scrubbed
again on the host (`scrub.mjs --require-key`; a surviving key deletes the
copy, exit 4), and the VM is powered off and restored to `clean` whatever
happened (`-Keep` leaves a failed run's VM up for debugging).

Exit codes: the scenario's (0 pass, 1 a step failed), 3 `npm ci` in the
guest failed, 4 key found after scrub, 5 guest wrapper error, 6 timeout
(`-TimeoutMin`, default 75), 7 rig error (VM, SSH, copy).

State lives in `%LOCALAPPDATA%\orbital-fresh-install` (rig key, `tester`'s
password, logs, installer cache), never in the repo.

**CI.** The private repo `zqiren/orbital-vm-ci` runs `run.ps1` from its
`fresh-install-winvm.yml` on a self-hosted runner on the PC (label
`orbital-winvm`, running as the VM's owner, started at logon;
`vbox/setup-runner.ps1`), once per mode, for every `v*` tag and manual
build. Private so that pull requests to this public repo can never reach the
runner. Report only: a release does not wait on it.

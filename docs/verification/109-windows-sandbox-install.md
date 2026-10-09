# Windows verification — spec 109 (install stall, hidden AgentOS-Worker account), issue #55

Branch `batch/1009-specs-102-110`. Use the **CI-built** installer (`gh workflow run ci.yml --ref batch/1009-specs-102-110`,
then download `Orbital-Setup-0.16.0.exe` from that run) — never a local `build-desktop.sh` artifact.
Run from an elevated PowerShell unless a step says otherwise. Record every number; the spec's claims that still
need a Windows box are V1 and V5.

## V1 — an inheritable ACE reaches existing descendants WITHOUT `/T` (the whole fix rests on this)

The grant now runs in the daemon at start (`refresh_sandbox_grants`), not in the installer. On a machine with a
populated `%LOCALAPPDATA%\Programs` (or `%USERPROFILE%\.cargo`, `%APPDATA%\npm`):

```powershell
# before: make sure the account has no entry on a deep file
icacls "$env:LOCALAPPDATA\Programs\<some app>\<deep>\<file>"
# time one root grant exactly as the daemon issues it (no /T)
Measure-Command { icacls "$env:LOCALAPPDATA\Programs" /grant "AgentOS-Worker:(OI)(CI)RX" /Q }
# after: the deep file must now show an AgentOS-Worker row flagged (I) = inherited
icacls "$env:LOCALAPPDATA\Programs\<some app>\<deep>\<file>"
# for comparison, the old form on the same root (expect minutes on a big tree)
Measure-Command { icacls "$env:LOCALAPPDATA\Programs" /grant "AgentOS-Worker:(OI)(CI)RX" /T /Q }
```
Pass: the deep file shows `AgentOS-Worker:(I)(RX)` after the `/T`-less grant, and that grant takes seconds.
If the leaf does NOT get the row, say so loudly: spec 109 D1 is then wrong and `/T` must come back (off the
install path only). Record both timings in `agent_os/platform/windows/permissions.py` `grant_toolchain_roots`
docstring (it has `<fill in>` placeholders).

## V2 — install wall-clock

Fresh VM, or first `"%ProgramFiles%\Orbital\bin\Orbital.exe" --teardown-sandbox` on an existing install.
Run the installer. Pass: the page after Welcome is the bilingual "About the AgentOS-Worker account" page (English
then Chinese, no mojibake); the step labelled "Creating the AgentOS-Worker sandbox account..." completes in
seconds, no CPU spike. First launch: `%LOCALAPPDATA%\Orbital\logs\daemon.log` (or the daemon's log location on
your build) contains one line `refresh_sandbox_grants(): N toolchain root(s) granted, M failed, K workspace(s)
checked in X ms` — record X.

## V3 — sign-in screen, description, and the agent still runs

```powershell
reg query "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList" /v AgentOS-Worker
net user AgentOS-Worker
```
Pass: the reg value exists with data `0x0`; `net user` shows Comment `Orbital agent sandbox account. Agents run
as this low-privilege user. Removed by the Orbital uninstaller.`; after sign-out (or reboot) the sign-in screen
lists only human users; in a project, ask the agent to run `dir` — the shell tool works (CreateProcessWithLogonW
is unaffected by the hide).

## V4 — upgrade path

Install the public v0.15.0 first, then this build over it. Pass: after the upgrade the UserList value exists
(the `[Registry]` section ran even though `--setup-sandbox` short-circuited), the account still works (V3 shell
check). Note: the description is NOT expected on this path (D4 accepted gap) — only on a fresh install.

## V5 — uninstall of a fresh install of this build

Pass: finishes in seconds with the progress bar moving; `net user AgentOS-Worker` → "The user name could not be
found"; the reg value is gone; `icacls` on a toolchain root AND on a sampled deep child shows no AgentOS-Worker
row and no unresolvable-SID (`S-1-5-21-…`) row — this confirms a root-only `/remove` propagates; a workspace
`.git\hooks` shows no unresolvable-SID row either.

## V7 — uninstall of a pre-109 install that was upgraded to this build

Install v0.15.0 → upgrade to this build → uninstall. Pass: finishes in seconds. A deep child under
`%LOCALAPPDATA%\Programs` MAY still show an unresolvable-SID row (explicit per-file ACEs written by v0.15.0's
`/T`; accepted residue, Windows ignores ACEs for SIDs that no longer exist) — record whether it does. A fresh
install afterwards: the agent runs a shell command.

## V6 — timeout insurance

Optional: with Process Monitor or a deliberately slow DACL you cannot easily produce; skip unless cheap. The unit
tests pin the 120 s bound and the "timed out → continue" behaviour.

## Report back

For each of V1–V7: pass/fail, the timings, and the exact `icacls` output lines for the deep-file checks. If V1
fails, stop and report before anything else.

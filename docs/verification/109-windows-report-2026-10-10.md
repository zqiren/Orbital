# Windows verification report — v0.16.0, spec 109 (#55) + general smoke

Branch `batch/1009-specs-102-110`. Run unattended overnight on 2026-10-10 on the dev laptop
(Lenovo 82JW, Windows 10 Home 22H2 zh-CN, user `qiren`, an existing v0.12.0 install at `C:\Orbital`).

## The one constraint that shaped everything

The session ran **non-elevated**, and UAC is `ConsentPromptBehaviorAdmin=5` on the secure desktop.
`Orbital-Setup-0.16.0.exe` is `PrivilegesRequired=admin` (manifest `asInvoker`; Inno elevates itself at
launch), and so is the uninstaller. A UAC consent prompt on the secure desktop cannot be answered by
software, by design. Nobody was at the keyboard. There is no VM tooling on this machine (Windows 10 Home:
no Hyper-V or Windows Sandbox, no VirtualBox or VMware) and no second human account. I did not try to get
around UAC.

**Consequence:** nothing that runs the installer or uninstaller could be executed. That covers V2, V3's
post-install checks, V4, V5, V7 and all of Part B. Everything that does *not* need elevation was done for
real: V1 on the real tree, live daemon runs from source for the code I changed, and a render check of the
pre-install page. Each blocked item below lists exactly what to run.

## Part A — spec 109

### V1: PASS, inheritance works. Timings are in `grant_toolchain_roots`

Tree: `%LOCALAPPDATA%\Programs`, **104,018 entries** (Python 3.13, VS Code, Inno Setup, …), NTFS SSD.

| Operation | Time |
|---|---|
| `icacls <root> /grant AgentOS-Worker:(OI)(CI)RX /Q` (no `/T`, as the daemon issues it) | **23.5 s** (19.2–19.3 s warm) |
| same, **repeated** while the ACE is already present | **19.2 s**: it re-walks every time |
| `icacls <root> /grant … /T /Q` (old form) | **182.3 s** (104,018 files processed) |
| `icacls <root> /remove AgentOS-Worker /Q` (root-only) | **19.6 s** |
| `icacls <root> /remove AgentOS-Worker /T /Q` (old form) | **182.0 s** |

Deep file, before → after the `/T`-less grant:
```
...\Programs\Python\Python313\Lib\encodings\utf_8.py NT AUTHORITY\SYSTEM:(I)(F)          (no AgentOS-Worker row)
...\Programs\Python\Python313\Lib\encodings\utf_8.py LAPTOP-DOSVQIJ9\AgentOS-Worker:(I)(RX)
```
After the root-only `/remove`, the row is gone from the same file. That also confirms V5's propagation
claim at the ACL level.

**Caveat (recorded in the docstring):** a subtree whose DACL is *protected* (inheritance disabled) does not
receive the ACE. Here that was `Microsoft VS Code`. A deep file under it had no row after the root-only grant;
`/T` writes an explicit ACE on the protected node. That subtree grants `BUILTIN\Users:(RX)` and
`Authenticated Users:(RX)` anyway, so the worker (a member of `Users`) reads it regardless. Decision: D1
stands, and `/T` stays off.

**Follow-up finding, fixed:** every `/grant` re-propagates even when the identical ACE is already there, and
the daemon re-runs the grant at every start. So every launch burned ~20 s of background icacls per populated
root. `grant_toolchain_roots` now does one single-object query and skips a root that already carries the
explicit `(OI)(CI)` RX entry. This is what `refresh_sandbox_grants`' docstring already promised. Live, from
source (`create_app` against a throwaway data dir, 5 roots: npm, Programs, pnpm, uv, `.local\bin`):
```
start 1: refresh_sandbox_grants(): 5 toolchain root(s) granted, 0 failed, 1 workspace(s) checked in 58073 ms
start 2: refresh_sandbox_grants(): 5 toolchain root(s) granted, 0 failed, 1 workspace(s) checked in 364 ms
```

### V2 install wall-clock: BLOCKED (UAC). Pre-install page: PASS on rendering

* The bilingual `before-install.txt` (UTF-8 + BOM, CRLF) was rendered by a throwaway non-elevated Inno 6
  script that uses the same file as `InfoBeforeFile`. UI Automation read the memo back: the Chinese is intact,
  with no mojibake (e.g. `关于 AgentOS-Worker 账户`, `如果你在「计算机管理 → 本地用户和组 → 用户」里看到`).
  The English half is visible in a screenshot. **The Chinese section stays.** Caveat: this machine is zh-CN;
  an en-US machine was not available.
* To run: install the CI build, time the "Creating the AgentOS-Worker sandbox account..." step, then on first
  launch grep the daemon log for `refresh_sandbox_grants()`. Expect up to ~1 min of background icacls on a
  dev machine (58–78 s here for 5 roots) on the first start only, and well under a second afterwards.

### V3 sign-in screen, description, shell: BLOCKED (UAC)

Pre-upgrade baseline on this machine (account created by v0.12.0):
```
reg query "HKLM\...\Winlogon\SpecialAccounts\UserList" /v AgentOS-Worker
  ERROR: The system was unable to find the specified registry key or value.
net user AgentOS-Worker
  Comment
  Local Group Memberships      *Users
```
After installing 0.16.0 over it, expect the `UserList` value `0x0` and an **empty** Comment (D4 accepted gap
on the exists path; see upgrade notes below).

### V4 upgrade from v0.15.0, V5 uninstall, V7 pre-109 uninstall: BLOCKED (UAC)

Two findings from V1 numbers change what V5 and V7 will show:

* **Uninstall does not "finish in seconds" on a dev machine.** `--teardown-sandbox` revokes each toolchain
  root inside one hidden `[UninstallRun]` step, and the bar does not move. A root-only `/remove` still walks
  the subtree, **and does so even when the account has no entry on the root**. Measured per root here: npm
  34.8 s, Programs 19.3 s, pnpm 4.9 s, uv and `.local\bin` ~0.01 s, **59 s total with or without the ACE**.
  Fixed (`revoke_toolchain_roots`): a root with no explicit entry is skipped after one query (**0.04 s** vs 59
  s), and the rest are revoked concurrently (**49.7 s** vs 58.8 s with the ACE on all five roots; the work is
  disk-bound). Verified live from source against the real roots: afterwards the roots and the deep file are
  clean. Expect V5's teardown step to take roughly as long as the slowest populated root (~35–50 s here),
  not "seconds".
* Each icacls call in teardown is bounded at 120 s. At this machine's rate (~5k entries/s) a root above
  ~500k entries would time out mid-propagation. Partial propagation can leave inherited rows naming the
  deleted SID, the same class of residue V7 already accepts.

### V6 timeout insurance: skipped (optional per the checklist; unit tests pin it)

## Part B — general smoke of the installed app: BLOCKED (UAC)

No step could run: install, onboarding, project, shell/grep, worker pin, workspace panel, new-session
button, unread dot, file card, browser automation, tray close/reopen/quit, port-8000 relaunch, upgrade,
uninstall. Static checks done instead:

* ripgrep: `agentos.spec` bundles `agent_os/vendor/rg/rg.exe` to `agent_os/vendor/rg`, which is the first
  path `grep_tool` searches under `_MEIPASS`. OK.
* Stray console windows: every module under `agent_os/` that spawns a subprocess references a no-window
  creation flag. No offender found.
* Port 8000 held by a foreign server: `is_already_running` adopts 8000 only if `/api/v2/settings` answers
  200. Otherwise `boot_daemon_with_retry` probes both binds and retries on a random port (bug #75). The logic
  looks right; it was not run.
* Duplicate tray: `tests/regression/test_single_instance_tray.py` had one red,
  `test_guard_precedes_tray_and_daemon_in_source`. That was a stale source-text check (ae89cba moved
  `start_daemon` into `boot_daemon_with_retry`), not a behaviour change. Fixed; 23/23 green.
* The installer is **unsigned** (`Get-AuthenticodeSignature`: NotSigned), so expect SmartScreen.

## HIGH severity, pre-existing since v0.1.0: the sandbox account can replace `Orbital.exe`

`DefaultDirName=C:\Orbital`, not Program Files, so `{app}` inherits the drive root's
`Authenticated Users:(OI)(CI)(IO)(M)`. On this machine's install:
```
C:\Orbital\bin\Orbital.exe NT AUTHORITY\Authenticated Users:(I)(M)
```
`AgentOS-Worker` is an authenticated user. A sandboxed agent can therefore overwrite the binary the human
launches and that the **elevated** uninstaller runs (`[UninstallRun] {app}\bin\Orbital.exe
--teardown-sandbox`). That is a sandbox escape plus a route to elevation.

**Not changed on the release branch**, because an installer behaviour change could not be install-tested
here. The fix is on **`proposal/windows-install-dir-acl`** (commit `8f08c47`, on top of `340527f`; its own CI installer: run `37970617880`). It adds a
root-only `icacls "{app}" /inheritance:r /grant:r SYSTEM/Administrators F, Users/Authenticated Users RX`
`[Run]` entry before the first `Orbital.exe` entry, and it also runs on upgrades. Nothing writes to `{app}`
at runtime: logs, browsers and data are all under the per-user data dir, and pywebview's WebView2 folder is a
temp dir in private mode. I validated the exact command non-elevated on a scratch `C:\OrbitalAclProbe` with
the same inherited DACL: a nested `bin\Orbital.exe` went from `Authenticated Users:(I)(M)` to `(I)(RX)` in
17 ms. To verify after merging it: install, then run `icacls C:\Orbital\bin\Orbital.exe` and check that no
`(M)` row is left.

## Other findings

* **The unit suite was mutating the developer's real ACLs.** On a Windows box where `AgentOS-Worker` exists,
  any `with TestClient(create_app(...))` test ran the real startup refresh, i.e. `icacls /grant` on the real
  toolchain roots. That is how V1's cleaned root got re-granted mid-session. Fixed with an autouse fixture in
  `tests/conftest.py`; the five lifespan-heavy files went from 126 s to 63 s.
* The local `tests/unit` run on this machine has **22** environment-only failures: symlink privilege,
  SDK-transport and quota tests, browser profile lock, CLI-login idle timeout. The set is identical with and
  without my changes. CI windows-latest and macos-14 show only the expected
  `test_browser_live.py::test_route_is_mounted_on_the_app`.
* An enabled leftover local account `AgentOS-Test-User` ("Agent OS Sandbox User - DO NOT DELETE", 2026-02-04)
  exists on this machine and is not hidden from the sign-in screen. It is not created by current code, as far
  as I can tell. Delete it by hand if it's unneeded.
* `docs/verification/109-windows-sandbox-install.md` V2 says
  `"%ProgramFiles%\Orbital\bin\Orbital.exe" --teardown-sandbox`. The real default is
  `C:\Orbital\bin\Orbital.exe`.

## What a v0.15.0 user will notice

**On upgrade:** a new "About the AgentOS-Worker account" page (EN + ZH) after Welcome. The sandbox step
finishes almost at once, because the account exists and `--setup-sandbox` short-circuits; there is no more
toolchain `/T` walk. The account disappears from the sign-in screen (the `[Registry]` entry runs on every
install). Its description stays empty (D4 gap). First launch: v0.15.0's `/T` grant already left the explicit
root ACE, so the new skip makes the background refresh cost milliseconds. Roots v0.15.0 never granted (a
toolchain installed later) get one background walk of tens of seconds on that first start only.

**On uninstall (after upgrading):** one hidden step that takes about as long as the slowest populated root
(~35–50 s on this machine, versus the minutes the `/T` walk took in v0.15.0). Then the account and the
`UserList` value are deleted. Per-file explicit ACEs that v0.15.0's `/T` wrote stay behind as
unresolvable-SID rows on deep files (accepted V7 residue). Workspace and user data are kept.

## Files changed on `batch/1009-specs-102-110`

| Commit | File | Why |
|---|---|---|
| `8eb4319` | `agent_os/platform/windows/permissions.py` | V1 timings + protected-DACL caveat in `grant_toolchain_roots`; skip a root that already has the ACE (`_has_inheritable_rx_grant`) |
| `8eb4319` | `tests/unit/test_windows_permissions.py` | V1 recorded in the T1 docstring; skip/no-skip/query-failure tests; failing-root test updated for the query |
| `8bc5ada` | `tests/conftest.py` | autouse fixture: the startup refresh never runs real icacls in tests |
| `8bc5ada` | `tests/unit/test_startup_sandbox_grant_refresh.py` | guard test for that fixture |
| `8bc5ada` | `tests/regression/test_single_instance_tray.py` | stale `start_daemon` source check → `boot_daemon_with_retry` |
| `340527f` | `agent_os/platform/windows/permissions.py` | `revoke_toolchain_roots`: skip roots without an explicit entry; revoke concurrently (`_has_explicit_entry`) |
| `340527f` | `tests/unit/test_windows_permissions.py`, `tests/unit/test_windows_sandbox_install.py` | skip, any-explicit-entry, query-failure, concurrency (barrier) and no-roots tests; teardown test feeds a granted root |
| (this file) | `docs/verification/109-windows-report-2026-10-10.md` | this report |

`installer/before-install.txt` is unchanged (rendering passed).

## CI

* Started from run `37961602031` (`6bf8c84`, same code as `cd948fc`): installer
  `Orbital-Setup-0.16.0.exe`, 404,753,212 bytes, SHA-256 `3A89C90D…ACF67D703`, unsigned. Downloaded but
  could not be run (UAC).
* `37966948821` (`8eb4319`): green except the expected `test_browser_live` red.
* **Last installer: run `37970451700` (`340527f`, the code head of this branch).** Backend tests green on
  windows-latest (4446 passed) and macos-14 (4556 passed), apart from the expected `test_browser_live` red.
  Frontend, .dmg and .exe builds succeeded. `Orbital-Setup-0.16.0.exe`: 404,750,768 bytes, SHA-256
  `458E2E5766008C0B859C716B5BFDA844DE90411BC26DEBE85C4CC8E6D52A4169`, ProductVersion 0.16.0, NotSigned.
  Downloaded and fingerprinted, **not installed** (UAC). This commit adds only this report, so the installer
  matches the branch code.
* Proposal branch `proposal/windows-install-dir-acl` (`8f08c47`): run `37970617880`, same result (only the
  expected red), with its own installer artifact for testing the ACL fix.

## To finish verification (needs someone at the keyboard to approve UAC)

1. `gh run download 37970451700 -n Orbital-Windows-installer`. Uninstall v0.12.0 first, or install v0.15.0
   (for V4/V7), then run the 0.16.0 installer.
2. V2–V5 and V7 per `109-windows-sandbox-install.md`. Expect the teardown step at roughly the slowest
   populated toolchain root (~35–50 s on this laptop), not "seconds".
3. Part B walk-through as in the original task list.
4. Decide on `proposal/windows-install-dir-acl`. If taken, add `icacls C:\Orbitalin\Orbital.exe` (no
   `(M)` row) to V3.

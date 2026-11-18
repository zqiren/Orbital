# In-app upgrade: Windows manual test

The Windows path of the one-click upgrade (`agent_os/update_windows.py` plus the
`[Code]` handshake in `installer/agentos-setup.iss`) has unit tests with the
platform mocked. Nothing has run it on Windows yet. This runbook is that run.
Use the **CI-built installers** (never a local `build-desktop.sh` build), on a
clean Windows 10 or 11 VM with a snapshot taken before step 1.

## What you need

- **OLD**: a CI `Orbital-Setup-X.Y.Z.exe` built from a commit that already
  contains the in-app upgrade. The first version that ships it cannot be
  upgraded *to* in place; it can only be upgraded *from*.
- **NEW**: a CI installer with a higher version, published as a GitHub release
  (draft releases are refused) with the stable alias `Orbital-Setup.exe` and its
  sha256 digest, and/or mirrored to OSS (`latest.json`).
- To test before a real release, point the app at a feed you control. Set
  these user environment variables before launching OLD, then sign out and
  back in so Explorer passes them on:
  - `AGENT_OS_UPDATE_URL=http://<host>:<port>/latest`, a JSON
    `{"version": "<NEW>", "url": "..."}` (the notify check).
  - The installer feed URLs are constants in `agent_os/updater.py`. For a
    pre-release run, use a real published release, or a CI build of a branch
    with those constants patched. Never ship a patched build.
- Sysinternals Process Explorer (optional, for step 9).

## Steps

1. **Install OLD** by double-clicking it (normal interactive install, default
   folder `C:\Orbital`). Launch Orbital, finish onboarding, create a project,
   send one chat message. Note the version under Settings, then About.
2. **The pill appears.** Within a minute of launch (startup check), the
   bottom-left pill reads **Update to vNEW** (Chinese UI: **更新到 vNEW**). If it
   reads "vNEW available →" (a link), the app decided it cannot update itself.
   Collect `%APPDATA%\Orbital\logs\` and stop: `install_layout` did not
   recognise `C:\Orbital\bin\Orbital.exe` plus `C:\Orbital\unins000.exe`.
3. **Click once.** The pill shows **Downloading n%**, rising, with a thin bar.
   The app stays usable: open another project and type. Check that the file
   grows in `%LOCALAPPDATA%\Orbital\updates\Orbital-NEW.exe.part`.
4. **Resume.** While it downloads, disconnect the network for 30 s. The pill
   shows **Update failed** with **Retry** and **Download**. Reconnect and click
   **Retry**. The percentage should continue from where it stopped, not from 0.
5. **Verifying, then UAC.** At 100 % the pill shows **Verifying…**, then
   **Restarting…**, and the UAC prompt appears. It should name the installer
   with an *Unknown publisher* (unsigned). Record whether a **SmartScreen**
   ("Windows protected your PC") screen appears *before* UAC. The expected
   answer is no: a file downloaded by Orbital's own process has no
   Mark-of-the-Web. Run
   `Get-Item %LOCALAPPDATA%\Orbital\updates\Orbital-NEW.exe -Stream *` and
   note whether a `Zone.Identifier` stream exists.
6. **Decline UAC (No).** Orbital must stay open on OLD, and the pill must
   read **Restart to update**. Nothing in `C:\Orbital` may change (check the
   modified dates).
7. **Busy prompt.** Start a long agent turn (e.g. "write a 1500-word essay").
   While it streams, click **Restart to update**. Expect the existing quit
   dialog, now reading "Agents are still working. Quitting stops them." Click
   **Cancel**: the turn continues and the pill stays on **Restart to update**.
8. **Accept.** Let the turn finish, click **Restart to update**, accept UAC.
   Expect, in order: the Orbital window and tray icon disappear within about
   2 s, leaving no ghost tray icon. Setup's small progress window appears
   (`/SILENT`), with no wizard pages and no "close applications" prompt. The
   window closes and Orbital reopens by itself, **not elevated** (Task
   Manager → Details → Elevated column = No), on NEW (Settings → About).
9. **Nothing lingered.** Before step 8, open Process Explorer and note any
   `node.exe` / `rg.exe` running from `C:\Orbital\bin`. After step 8 none of
   the old ones remain, and `%LOCALAPPDATA%\Orbital\updates\setup-NEW.log`
   contains no "file in use" / "RestartManager" failures. If the log shows
   Restart Manager closing a process, note which one.
10. **Data survived.** The project and its chat history from step 1 are
    there. The sandbox account still works: run an agent command in the
    project. Apps & features lists exactly **one** Orbital, at NEW (an explicit
    `AppId=Orbital` now sits in the .iss, equal to the old implicit value).
11. **Cleanup.** After NEW starts, `%LOCALAPPDATA%\Orbital\updates\` no longer
    contains `Orbital-NEW.exe` or `pending.json`.
12. **Defender.** Note any Microsoft Defender notification during steps 3 to 8
    (Protection history: Settings → Privacy & security → Windows Security →
    Virus & threat protection → Protection history). A detection on the
    downloaded installer, or on Setup started by Orbital, is a release
    blocker for this feature.

## Failure cases (restore the VM snapshot to OLD before each)

13. **Hash mismatch.** Serve a NEW whose bytes differ from the feed's sha256
    (e.g. a feed you control pointing at another file). Expected: the pill
    shows **Update failed** after the download. The `.exe` is gone from
    `updates\`. **Download** opens the browser download. No UAC prompt.
14. **Setup aborts mid-way.** Start the upgrade, then while Setup's progress
    window shows, open `C:\Orbital\bin\Orbital.exe` with an exclusive lock
    from another tool (or kill `Setup.tmp` in Task Manager). Expected: Setup
    rolls back, Orbital reopens on **OLD**, and the pill shows
    **Update failed** (the startup reconcile reads `pending.json`).
15. **App will not quit.** Simulate a hung shell (Process Explorer → Suspend
    `Orbital.exe` right after you accept UAC). Expected: Setup waits up to
    60 s, then aborts without touching files (AppMutex check +
    `/SUPPRESSMSGBOXES`). Resume Orbital: it is still OLD and usable.
16. **Standard user.** Log in as a non-admin user and repeat steps 3 to 8 with
    an admin's credentials in UAC. Expected: Orbital reopens for the
    *standard* user (ExecAsOriginalUser), on NEW.
17. **Portable copy.** Copy `C:\Orbital\bin` elsewhere and run it from there.
    The pill must show the link ("vNEW available →"), never the in-place
    button.

Record per step: pass/fail, a screenshot of the pill, and the
`%LOCALAPPDATA%\Orbital\updates\setup-*.log` plus `%APPDATA%\Orbital\logs\`
for any failure.

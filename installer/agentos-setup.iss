; installer/agentos-setup.iss
; Inno Setup script for Orbital

[Setup]
AppName=Orbital
AppVersion=0.16.0
AppPublisher=Orbital
DefaultDirName=C:\Orbital
DisableDirPage=no
DefaultGroupName=Orbital
OutputBaseFilename=Orbital-Setup-0.16.0
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\bin\Orbital.exe
PrivilegesRequired=admin
; Spec 109 D5: shown as the page after Welcome, BEFORE any file is written.
; Names the AgentOS-Worker account and why it exists (English, then Simplified
; Chinese in the same file; UTF-8 with BOM so the Chinese renders — a plain
; ANSI .txt would mojibake).
InfoBeforeFile=before-install.txt
; Must match SINGLE_INSTANCE_MUTEX_NAME in agent_os/desktop/main.py. Makes
; Setup/Uninstall ask the user to close a running Orbital instead of writing
; files under a live process and then launching a second copy (line 53).
AppMutex=OrbitalDesktopShell

[Files]
; Application binaries (PyInstaller output)
Source: "..\dist\Orbital\*"; DestDir: "{app}\bin"; Flags: recursesubdirs ignoreversion

; React SPA
Source: "..\web\dist\*"; DestDir: "{app}\web"; Flags: recursesubdirs ignoreversion

; Icon assets
Source: "..\assets\icon.png"; DestDir: "{app}\assets"
Source: "..\assets\icon.ico"; DestDir: "{app}\assets"

; WebView2 Evergreen bootstrapper (~2 MB) — downloaded into installer/ by
; scripts/build-desktop.sh (and CI) before iscc runs. Without the runtime,
; pywebview silently degrades to the IE11 engine and Orbital shows a blank
; window.
Source: "MicrosoftEdgeWebView2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
; Desktop shortcut
Name: "{autodesktop}\Orbital"; Filename: "{app}\bin\Orbital.exe"; IconFilename: "{app}\assets\icon.ico"
; Start Menu
Name: "{group}\Orbital"; Filename: "{app}\bin\Orbital.exe"; IconFilename: "{app}\assets\icon.ico"
Name: "{group}\Uninstall Orbital"; Filename: "{uninstallexe}"

[Registry]
; Spec 109 D3: keep the AgentOS-Worker sandbox account off the Windows sign-in
; screen. Winlogon lists every enabled local account unless it is named here
; with DWORD 0. It lives in the installer, not in Python, because (a) the
; installer is already elevated, (b) this section runs on EVERY install
; including upgrades — where `--setup-sandbox` short-circuits as soon as the
; account already exists, so a Python-side write would never reach the
; machines that installed before this entry existed — and (c) the uninstaller
; reverses it (uninsdeletevalue). Python mirrors the write best-effort for
; source installs (SandboxAccountManager._hide_from_sign_in).
; Winlogon reads the 64-bit registry view; this script does not set
; ArchitecturesInstallIn64BitMode, so a plain HKLM write from the 32-bit
; Setup process could land in WOW6432Node. HKLM64 forces the 64-bit view on
; x64; the plain HKLM entry is for a 32-bit Windows (HKLM64 is invalid there).
Root: HKLM64; Subkey: "SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList"; \
    ValueType: dword; ValueName: "AgentOS-Worker"; ValueData: 0; \
    Flags: uninsdeletevalue; Check: IsWin64
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList"; \
    ValueType: dword; ValueName: "AgentOS-Worker"; ValueData: 0; \
    Flags: uninsdeletevalue; Check: not IsWin64

[Run]
; Lock the install dir down BEFORE anything runs from it. DefaultDirName is
; C:\Orbital, not Program Files, so {app} inherits the drive root's
; "Authenticated Users:(OI)(CI)(IO)(M)" — every authenticated account,
; AgentOS-Worker included, could replace bin\Orbital.exe: a sandboxed agent
; could rewrite the binary the human launches and the elevated uninstaller
; (UninstallRun section) executes. Replace the inherited DACL with the Program Files
; shape: SYSTEM + Administrators full, Users + Authenticated Users read/execute.
; Runs on upgrades too (UsePreviousAppDir keeps old installs at C:\Orbital).
; Root-only: the bundle's files carry only inherited ACEs, which Windows
; recomputes from the new root DACL (spec 109 V1). Nothing writes to {app} at
; runtime — logs, browsers and data live under the per-user data dir.
Filename: "{sys}\icacls.exe"; \
    Parameters: """{app}"" /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F *S-1-5-32-545:(OI)(CI)RX *S-1-5-11:(OI)(CI)RX /Q"; \
    StatusMsg: "Securing the Orbital install folder..."; \
    Flags: runhidden waituntilterminated
; Install the WebView2 runtime FIRST when it's missing — must precede any
; Orbital launch (the app cannot render without it).
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; \
    StatusMsg: "Installing Microsoft Edge WebView2 Runtime..."; \
    Check: NeedsWebView2; Flags: waituntilterminated
; Run sandbox setup with admin privileges (installer is elevated).
; Account + workspace + worker home only; the toolchain grants run in the
; daemon at startup (spec 109). Before 109 this step walked every per-user
; toolchain root with `icacls /T` — minutes of pinned CPU behind a frozen
; progress bar on a dev machine (issue #55). Every icacls call inside is now
; bounded to 120 s, so this step can no longer wedge the installer.
Filename: "{app}\bin\Orbital.exe"; Parameters: "--setup-sandbox"; \
    StatusMsg: "Creating the AgentOS-Worker sandbox account..."; \
    Flags: runhidden waituntilterminated
; Launch after install
Filename: "{app}\bin\Orbital.exe"; Description: "Launch Orbital"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\bin\Orbital.exe"; Parameters: "--teardown-sandbox"; \
    Flags: runhidden waituntilterminated; RunOnceId: "SandboxTeardown"

[UninstallDelete]
; Clean up logs on uninstall (but NOT %APPDATA%\Orbital — preserve user data)
Type: filesandordirs; Name: "{app}\logs"

[Code]
// Probes the real Evergreen Runtime client GUID — the previous probe used
// a GUID that does not exist in any WebView2 install, so it always
// reported "missing". Per-user installs register under HKCU, machine
// installs under HKLM (+WOW6432Node).
function NeedsWebView2(): Boolean;
begin
  Result := not (
    RegKeyExists(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}') or
    RegKeyExists(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}') or
    RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'));
end;

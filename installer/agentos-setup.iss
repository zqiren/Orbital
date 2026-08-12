; installer/agentos-setup.iss
; Inno Setup script for Orbital

[Setup]
AppName=Orbital
AppVersion=0.8.7
AppPublisher=Orbital
DefaultDirName=C:\Orbital
DisableDirPage=no
DefaultGroupName=Orbital
OutputBaseFilename=Orbital-Setup-0.8.7
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\bin\Orbital.exe
PrivilegesRequired=admin
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

[Run]
; Install the WebView2 runtime FIRST when it's missing — must precede any
; Orbital launch (the app cannot render without it).
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; \
    StatusMsg: "Installing Microsoft Edge WebView2 Runtime..."; \
    Check: NeedsWebView2; Flags: waituntilterminated
; Run sandbox setup with admin privileges (installer is elevated)
Filename: "{app}\bin\Orbital.exe"; Parameters: "--setup-sandbox"; \
    StatusMsg: "Configuring agent sandbox..."; \
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

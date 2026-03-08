; installer/agentos-setup.iss
; Inno Setup script for Orbital

[Setup]
AppName=Orbital
AppVersion=1.0.0
AppPublisher=Orbital
DefaultDirName=C:\Orbital
DisableDirPage=no
DefaultGroupName=Orbital
OutputBaseFilename=Orbital-Setup-1.0.0
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\bin\Orbital.exe
PrivilegesRequired=admin

[Files]
; Application binaries (PyInstaller output)
Source: "..\dist\Orbital\*"; DestDir: "{app}\bin"; Flags: recursesubdirs ignoreversion

; React SPA
Source: "..\web\dist\*"; DestDir: "{app}\web"; Flags: recursesubdirs ignoreversion

; Icon assets
Source: "..\assets\icon.png"; DestDir: "{app}\assets"
Source: "..\assets\icon.ico"; DestDir: "{app}\assets"

[Icons]
; Desktop shortcut
Name: "{autodesktop}\Orbital"; Filename: "{app}\bin\Orbital.exe"; IconFilename: "{app}\assets\icon.ico"
; Start Menu
Name: "{group}\Orbital"; Filename: "{app}\bin\Orbital.exe"; IconFilename: "{app}\assets\icon.ico"
Name: "{group}\Uninstall Orbital"; Filename: "{uninstallexe}"

[Run]
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
function NeedsWebView2(): Boolean;
var
  RegKey: String;
begin
  RegKey := 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BEE-13A6279B0638}';
  Result := not RegKeyExists(HKLM, RegKey);
  if Result then
  begin
    RegKey := 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BEE-13A6279B0638}';
    Result := not RegKeyExists(HKLM, RegKey);
  end;
end;

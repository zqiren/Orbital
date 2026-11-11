@echo off
rem One-time setup of the Windows fresh-install VM on this PC. Double-click it
rem (or run it from a normal, NOT elevated, console) as the user that owns the
rem runner. It asks for ONE UAC prompt (VirtualBox install), then builds the
rem VM unelevated: unattended Windows 11 zh-CN install, provisioning, snapshot
rem `clean`. Takes about an hour. Log: %LOCALAPPDATA%\orbital-fresh-install\logs
rem
rem   bootstrap.cmd [cache dir, default D:\orbital-winvm-cache]
setlocal
set "HERE=%~dp0"
set "CACHE=%~1"
if "%CACHE%"=="" set "CACHE=D:\orbital-winvm-cache"

if not exist "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" (
  echo Installing VirtualBox: approve the UAC prompt.
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','\"%HERE%install-host.ps1\"','-Cache','\"%CACHE%\"'; exit $p.ExitCode"
  if errorlevel 1 ( echo install-host failed: see %CACHE%\install-host.log & exit /b 1 )
)
if not exist "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" ( echo VirtualBox is still not installed & exit /b 1 )

for %%F in ("%CACHE%\iso\*.iso") do set "ISO=%%~fF"
for %%F in ("%CACHE%\vbox\VBoxGuestAdditions_*.iso") do set "GA=%%~fF"
for %%F in ("%CACHE%\node\node-v24*-x64.msi") do set "NODEMSI=%%~fF"
for %%F in ("%CACHE%\openssh\OpenSSH-Win64-*.msi") do set "SSHMSI=%%~fF"

powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%prepare-vm.ps1" -Iso "%ISO%" -AdditionsIso "%GA%" -NodeMsi "%NODEMSI%" -OpenSshMsi "%SSHMSI%" %2 %3 %4 %5
if errorlevel 1 ( echo prepare-vm failed: see %LOCALAPPDATA%\orbital-fresh-install\logs & exit /b 1 )
echo VM orbital-fresh-win with snapshot clean is ready.

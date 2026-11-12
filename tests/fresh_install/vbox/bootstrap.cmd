@echo off
rem One-time setup of the Windows fresh-install VM on this PC. Double-click it
rem (or run it from a normal, NOT elevated, console) as the user that owns the
rem runner. It asks for ONE UAC prompt (VirtualBox install), then builds the
rem VM unelevated: unattended Windows 11 zh-CN install, provisioning, snapshot
rem `clean`. Takes about an hour. The window stays open at the end (pass/fail).
rem Logs: <cache>\bootstrap.log, <cache>\install-host.log,
rem       %LOCALAPPDATA%\orbital-fresh-install\logs\prepare-vm-*.log
rem
rem   bootstrap.cmd [cache dir, default D:\orbital-winvm-cache] [extra prepare-vm args]
rem Image 1 = "Windows 11 Home" (EditionID Core) on Microsoft's zh-cn consumer
rem ISO, build 26300 (it has no Home China image); a later -ImageIndex wins.
setlocal
set "HERE=%~dp0"
set "CACHE=%~1"
if "%CACHE%"=="" set "CACHE=D:\orbital-winvm-cache"
set "BLOG=%CACHE%\bootstrap.log"
echo %DATE% %TIME% bootstrap started as %USERNAME% >> "%BLOG%"

if not exist "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" (
  echo Installing VirtualBox: a UAC prompt is coming, click Yes.
  echo %DATE% %TIME% asking for UAC to run install-host.ps1 >> "%BLOG%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $p = Start-Process powershell -Verb RunAs -Wait -PassThru -ErrorAction Stop -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','\"%HERE%install-host.ps1\"','-Cache','\"%CACHE%\"'; exit $p.ExitCode } catch { Write-Host ('UAC / elevation failed: ' + $_.Exception.Message); exit 9 }" >> "%BLOG%" 2>&1
  if errorlevel 1 goto install_failed
)
if not exist "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" goto install_failed
echo %DATE% %TIME% VirtualBox present >> "%BLOG%"

for %%F in ("%CACHE%\iso\*.iso") do set "ISO=%%~fF"
for %%F in ("%CACHE%\vbox\VBoxGuestAdditions_*.iso") do set "GA=%%~fF"
for %%F in ("%CACHE%\node\node-v24*-x64.msi") do set "NODEMSI=%%~fF"
for %%F in ("%CACHE%\openssh\OpenSSH-Win64-*.msi") do set "SSHMSI=%%~fF"

echo Building the VM (about an hour; you can leave this window alone).
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%prepare-vm.ps1" -Iso "%ISO%" -AdditionsIso "%GA%" -NodeMsi "%NODEMSI%" -OpenSshMsi "%SSHMSI%" -ImageIndex 1 %2 %3 %4 %5
if errorlevel 1 (
  echo %DATE% %TIME% prepare-vm failed >> "%BLOG%"
  echo prepare-vm FAILED: see %LOCALAPPDATA%\orbital-fresh-install\logs
  pause
  exit /b 1
)
echo %DATE% %TIME% VM ready >> "%BLOG%"
echo VM orbital-fresh-win with snapshot clean is ready.
pause
exit /b 0

:install_failed
echo %DATE% %TIME% VirtualBox install failed or UAC declined >> "%BLOG%"
echo.
echo VirtualBox was NOT installed (UAC declined or timed out, or install-host failed).
echo See %BLOG% and %CACHE%\install-host.log. Run bootstrap.cmd again and click Yes.
pause
exit /b 1

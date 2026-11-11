# The one step of the Windows rig that needs administrator rights: install
# VirtualBox 7.x (its kernel driver) and the Extension Pack on the host.
# Minimal feature set: the application + VBoxSup driver only (no bridged /
# host-only network drivers, no USB, no Python bindings; the rig uses NAT).
# Run elevated (bootstrap.cmd asks for the one UAC prompt):
#   powershell -ExecutionPolicy Bypass -File install-host.ps1 -Cache D:\orbital-winvm-cache
# Expects in <Cache>\vbox: VirtualBox-<ver>-<build>-Win.exe,
# Oracle_VirtualBox_Extension_Pack-<ver>.vbox-extpack and SHA256SUMS from
# download.virtualbox.org; verifies both against SHA256SUMS first.
# Windows PowerShell 5.1; ASCII only.
param([string]$Cache = 'D:\orbital-winvm-cache')
$ErrorActionPreference = 'Stop'
$log = Join-Path $Cache 'install-host.log'
Start-Transcript -Path $log -Append | Out-Null
try {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  if (-not ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'not elevated: run this from an elevated PowerShell (or bootstrap.cmd)'
  }
  $dir = Join-Path $Cache 'vbox'
  $sums = Get-Content (Join-Path $dir 'SHA256SUMS')
  function Verify($f) {
    $name = Split-Path -Leaf $f
    $want = ($sums | Where-Object { $_ -match [regex]::Escape("*$name") + '$' } | Select-Object -First 1) -replace '\s.*$', ''
    $got = (Get-FileHash -Algorithm SHA256 $f).Hash
    if (-not $want -or $got -ne $want.ToUpper()) { throw "checksum mismatch for $name" }
    Write-Host "sha256 ok: $name"
  }
  $exe = Get-ChildItem $dir -Filter 'VirtualBox-*-Win.exe' | Select-Object -First 1
  $ext = Get-ChildItem $dir -Filter 'Oracle_VirtualBox_Extension_Pack-*.vbox-extpack' | Where-Object { $_.Name -notmatch '-\d{6}\.vbox-extpack$' } | Select-Object -First 1
  Verify $exe.FullName
  Verify $ext.FullName

  $vbm = 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
  if (-not (Test-Path $vbm)) {
    Write-Host "installing $($exe.Name)"
    # --logging writes the MSI log under %TEMP%\VirtualBox.
    $p = Start-Process $exe.FullName -Wait -PassThru -ArgumentList @('--silent', '--ignore-reboot', '--logging',
      '--msiparams', 'ADDLOCAL=VBoxApplication VBOX_INSTALLDESKTOPSHORTCUT=0 VBOX_INSTALLQUICKLAUNCHSHORTCUT=0 VBOX_START=0')
    Write-Host "VirtualBox installer exit $($p.ExitCode)"
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { throw "VirtualBox install failed ($($p.ExitCode)); MSI log under $env:TEMP\VirtualBox" }
  }
  & $vbm --version
  Write-Host "installing $($ext.Name)"
  'y' | & $vbm extpack install --replace $ext.FullName
  if ($LASTEXITCODE -ne 0) { throw "extpack install failed ($LASTEXITCODE)" }
  & $vbm list extpacks
  Write-Host 'install-host OK'
} finally {
  Stop-Transcript | Out-Null
}

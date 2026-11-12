# Orbital fresh-install test: shared helpers for the VirtualBox (local
# Windows VM) host. Dot-sourced by prepare-vm.ps1 and run.ps1.
#
# Windows PowerShell 5.1 compatible, ASCII only (5.1 reads a BOM-less script
# in the ANSI code page, which is GBK on a Chinese PC). Never installs or runs
# Orbital on the host: everything Orbital happens inside the VM, which is
# rolled back to snapshot `clean` before and after every run.

$ErrorActionPreference = 'Stop'

$FreshVm       = if ($env:FRESH_VM) { $env:FRESH_VM } else { 'orbital-fresh-win' }
$FreshSnapshot = if ($env:FRESH_SNAPSHOT) { $env:FRESH_SNAPSHOT } else { 'clean' }
$FreshSshPort  = if ($env:FRESH_SSH_PORT) { [int]$env:FRESH_SSH_PORT } else { 2222 }
$FreshVmUser   = if ($env:FRESH_VM_USER) { $env:FRESH_VM_USER } else { 'tester' }
# Rig SSH key, tester password, logs. Never in the repo.
$FreshStateDir = if ($env:FRESH_STATE_DIR) { $env:FRESH_STATE_DIR } else { Join-Path $env:LOCALAPPDATA 'orbital-fresh-install' }
$FreshSshKey   = Join-Path $FreshStateDir 'id_ed25519'
$FreshGuestDir = 'C:\fi'              # everything the rig puts in the guest lives here

function Log([string]$m) { Write-Host ('{0} [vbox] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m) }

function Find-VBoxManage {
  $cands = @()
  if ($env:VBOX_MSI_INSTALL_PATH) { $cands += (Join-Path $env:VBOX_MSI_INSTALL_PATH 'VBoxManage.exe') }
  $cands += 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
  foreach ($c in $cands) { if (Test-Path $c) { return $c } }
  throw 'VBoxManage.exe not found: install VirtualBox 7.x (tests/fresh_install/vbox/install-host.ps1)'
}
$VBoxManage = Find-VBoxManage

# VBox <args...>: run VBoxManage, throw on a non-zero exit, return its output lines.
function VBox {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  $out = & $VBoxManage @args 2>&1
  if ($LASTEXITCODE -ne 0) { throw "VBoxManage $($args -join ' ') exited $LASTEXITCODE`n$($out | Out-String)" }
  return $out
}
# VBoxTry <args...>: same, never throws; returns $true on success.
function VBoxTry {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  $null = & $VBoxManage @args 2>&1
  return ($LASTEXITCODE -eq 0)
}

function Get-VmState {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  $info = & $VBoxManage showvminfo $FreshVm --machinereadable 2>$null
  foreach ($l in $info) { if ($l -match '^VMState="(.*)"') { return $Matches[1] } }
  return 'unknown'
}

# Power the VM off (hard; the snapshot is the source of truth) and wait.
function Stop-Vm {
  $s = Get-VmState
  if ($s -in @('running', 'paused', 'stuck', 'starting', 'restoring', 'saving', 'stopping', 'livesnapshotting')) {
    Log "powering off $FreshVm (state $s)"
    $null = VBoxTry controlvm $FreshVm poweroff
  }
  for ($i = 0; $i -lt 60; $i++) {
    $s = Get-VmState
    if ($s -in @('poweroff', 'aborted', 'saved')) { break }
    Start-Sleep -Seconds 2
  }
  # The VM's session lock is released a moment after the state flips.
  Start-Sleep -Seconds 3
}

function Restore-Clean {
  Stop-Vm
  Log "restoring snapshot $FreshSnapshot"
  for ($i = 1; $i -le 5; $i++) {
    if (VBoxTry snapshot $FreshVm restore $FreshSnapshot) { return }
    Start-Sleep -Seconds 5
  }
  $null = VBox snapshot $FreshVm restore $FreshSnapshot
}

$SshExe = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$ScpExe = Join-Path $env:SystemRoot 'System32\OpenSSH\scp.exe'
$TarExe = Join-Path $env:SystemRoot 'System32\tar.exe'
$SshOpts = @('-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=NUL', '-o', 'LogLevel=ERROR',
             '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
             '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-i', $FreshSshKey)

# The guest's sshd runs commands through cmd.exe (the default shell). Run
# PowerShell either as a file or encoded, so nothing needs cmd quoting.
function GuestCmd([string]$cmd) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  & $SshExe @SshOpts -p $FreshSshPort "$FreshVmUser@127.0.0.1" $cmd
}
function GuestPs([string]$script) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  # No progress records: over SSH they arrive as CLIXML on stderr and garble the output.
  $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("`$ProgressPreference='SilentlyContinue'`n" + $script))
  & $SshExe @SshOpts -p $FreshSshPort "$FreshVmUser@127.0.0.1" "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $b64"
}
function GuestPsFile([string]$file, [string]$argline = '') {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  & $SshExe @SshOpts -p $FreshSshPort "$FreshVmUser@127.0.0.1" "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $file $argline"
}
# Copy to/from the guest. Guest paths are like C:/fi/in/ (forward slashes).
# -O (legacy scp protocol): the guest's Win32-OpenSSH 10.4 sftp subsystem
# exits 1 right away ("Connection closed"), which breaks scp's default SFTP mode.
function CopyTo([string]$src, [string]$dst) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  & $ScpExe -O -q @SshOpts -P $FreshSshPort $src "$($FreshVmUser)@127.0.0.1:$dst"
  if ($LASTEXITCODE -ne 0) { throw "scp $src -> guest $dst failed ($LASTEXITCODE)" }
}
function CopyFrom([string]$src, [string]$dst) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  & $ScpExe -O -q @SshOpts -P $FreshSshPort "$($FreshVmUser)@127.0.0.1:$src" $dst
  return ($LASTEXITCODE -eq 0)
}

function Wait-Ssh([int]$timeoutSec = 600) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  $t0 = Get-Date
  while (((Get-Date) - $t0).TotalSeconds -lt $timeoutSec) {
    # -E NUL: ssh's own "banner exchange" errors while the guest boots go
    # nowhere instead of filling the transcript.
    $null = & $SshExe @SshOpts -E NUL -p $FreshSshPort "$FreshVmUser@127.0.0.1" 'exit 0' 2>$null
    if ($LASTEXITCODE -eq 0) { return }
    if ((Get-VmState) -in @('paused', 'stuck', 'aborted', 'poweroff')) {
      throw "VM $FreshVm is $(Get-VmState) while waiting for SSH (see its Logs\VBox.log: disk full? out of memory?)"
    }
    Start-Sleep -Seconds 10
  }
  throw "ssh to $FreshVmUser@127.0.0.1:$FreshSshPort never came up within $timeoutSec s"
}

# Wait for tester's desktop (auto-logon): explorer.exe runs only in the
# interactive session, never in the SSH one.
function Wait-Desktop([int]$timeoutSec = 300) {
  $ErrorActionPreference = 'Continue'  # 5.1: native stderr must not throw
  $t0 = Get-Date
  while (((Get-Date) - $t0).TotalSeconds -lt $timeoutSec) {
    $null = GuestPs 'if (Get-Process explorer -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }' 2>$null
    if ($LASTEXITCODE -eq 0) { return }
    Start-Sleep -Seconds 5
  }
  throw "tester's desktop (explorer.exe) never appeared within $timeoutSec s: is auto-logon on?"
}

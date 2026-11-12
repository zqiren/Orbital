# One fresh-install run in the throwaway VirtualBox Windows VM (the Windows
# twin of tart/run.sh). Windows PowerShell 5.1; ASCII only.
#
#   $env:ORBITAL_TEST_OPENCODE_KEY = ...   (never on a command line)
#   .\tests\fresh_install\vbox\run.ps1 -Installer <new .exe> -Mode fresh|upgrade -Out <evidence dir> [-OldInstaller <previous .exe>]
#   .\tests\fresh_install\vbox\run.ps1 -RunId <Orbital CI run id> [-Mode upgrade] -Out <dir>   # download it (gh)
#   .\tests\fresh_install\vbox\run.ps1 -Tag v0.21.1 [-Mode upgrade] -Out <dir>                 # a release
#
# Steps: restore snapshot `clean` -> start the VM headless -> wait for SSH on
# 127.0.0.1:2222 and tester's auto-logon desktop -> copy the installer(s) and
# tests/fresh_install (without node_modules) in -> key on stdin into a file
# only tester can read -> guest-start.ps1 starts guest-run.ps1 as a scheduled
# task in tester's desktop session (Orbital's window needs a real desktop),
# which runs `npm ci` + scenario.mjs --require-key -> poll for its exit code
# -> copy the evidence (already scrubbed inside) to -Out -> scrub.mjs again
# here, deleting the copy if the key survived -> power off and restore
# `clean`, whatever happened. Exit 0 only if the scenario passed.
#
# Exit codes: scenario's own (0 pass, 1 a step failed, 2 usage), 3 npm ci in
# the guest failed, 4 key found in the evidence after scrub, 5 guest wrapper
# error, 6 timeout, 7 rig error (VM, SSH, copy).
param(
  [string]$Installer,
  [ValidateSet('fresh', 'upgrade')][string]$Mode = 'fresh',
  [string]$Out = (Join-Path (Get-Location) 'fresh-install-evidence'),
  [string]$OldInstaller,
  [string]$RunId,
  [string]$Tag,
  [string]$OldTag,
  [switch]$Keep,                 # leave the VM running on failure (debugging); restore it yourself
  [string]$ScenarioArgs = '',
  [int]$TimeoutMin = 75
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'common.ps1')
# Continue, not Stop: in 5.1 a native command's stderr would otherwise throw.
# Failures are checked explicitly ($LASTEXITCODE, VBox throws).
$ErrorActionPreference = 'Continue'
$FiDir = Split-Path -Parent $here
$Repo = if ($env:FRESH_REPO) { $env:FRESH_REPO } else { 'zqiren/Orbital' }

if (-not $env:ORBITAL_TEST_OPENCODE_KEY) { Write-Host 'ORBITAL_TEST_OPENCODE_KEY is not set'; exit 2 }
if (-not (Test-Path $FreshSshKey)) { Write-Host "no rig key at ${FreshSshKey}: run prepare-vm.ps1"; exit 2 }
$snaps = & $VBoxManage snapshot $FreshVm list --machinereadable 2>$null
if (-not ($snaps -match "=`"$FreshSnapshot`"")) { Write-Host "VM $FreshVm has no snapshot ${FreshSnapshot}: run prepare-vm.ps1"; exit 2 }

# ---- installers (host side; downloaded and cached, never installed here)
$cache = Join-Path $FreshStateDir 'installers'
function Get-ReleaseExe([string]$t) {
  $d = Join-Path $cache $t
  $f = Get-ChildItem $d -Filter 'Orbital-Setup-*.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $f) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    Log "downloading $t Windows installer"
    & gh release download $t --repo $Repo -p 'Orbital-Setup-*.exe' -D $d --clobber
    if ($LASTEXITCODE -ne 0) { throw "gh release download $t failed" }
    $f = Get-ChildItem $d -Filter 'Orbital-Setup-*.exe' | Select-Object -First 1
  }
  return $f.FullName
}
if (-not $Installer) {
  if ($Tag) { $Installer = Get-ReleaseExe $Tag }
  elseif ($RunId) {
    $d = Join-Path $cache "run-$RunId"
    if (-not (Get-ChildItem $d -Filter 'Orbital-Setup-*.exe' -ErrorAction SilentlyContinue)) {
      New-Item -ItemType Directory -Force -Path $d | Out-Null
      Log "downloading CI run $RunId artifact Orbital-Windows-installer"
      & gh run download $RunId --repo $Repo -n Orbital-Windows-installer -D $d
      if ($LASTEXITCODE -ne 0) { throw "gh run download $RunId failed" }
    }
    $Installer = (Get-ChildItem $d -Filter 'Orbital-Setup-*.exe' | Select-Object -First 1).FullName
  } else { Write-Host 'need -Installer, -RunId or -Tag'; exit 2 }
}
if ($Mode -eq 'upgrade' -and -not $OldInstaller) {
  if (-not $OldTag) {
    $tags = & gh release list --repo $Repo --exclude-drafts --exclude-pre-releases --limit 50 --json tagName -q '.[].tagName'
    $OldTag = $tags | Where-Object { $_ -ne $Tag } | Select-Object -First 1
  }
  if (-not $OldTag) { Write-Host 'could not find a release to upgrade from'; exit 2 }
  Log "upgrade from $OldTag"
  $OldInstaller = Get-ReleaseExe $OldTag
}
foreach ($f in @($Installer, $OldInstaller)) {
  if ($f -and -not (Test-Path $f)) { Write-Host "installer not found: $f"; exit 2 }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$work = Join-Path $FreshStateDir "work\$Mode-$stamp"
New-Item -ItemType Directory -Force -Path $work, $Out | Out-Null
$Out = (Resolve-Path $Out).Path

$newName = Split-Path -Leaf $Installer
$scen = @('--installer', "$FreshGuestDir\in\$newName", '--evidence', "$FreshGuestDir\evidence", '--require-key')
if ($Mode -eq 'upgrade') {
  $oldName = 'old-' + (Split-Path -Leaf $OldInstaller)
  $scen += @('--mode', 'upgrade', '--old-installer', "$FreshGuestDir\in\$oldName")
}
if ($ScenarioArgs) { $scen += $ScenarioArgs }
Log "mode $Mode, installer $newName$(if ($OldInstaller) { ", old $(Split-Path -Leaf $OldInstaller)" })"

$rc = 7
$consoleLocal = Join-Path $Out 'console.log'
try {
  # ---- clean VM
  Restore-Clean
  Log "starting $FreshVm (headless)"
  $null = VBox startvm $FreshVm --type headless
  Wait-Ssh 600
  Wait-Desktop 300
  Log 'guest is up, tester is logged on'
  $null = VBoxTry controlvm $FreshVm setvideomodehint 1600 900 32
  GuestPs "`$h=[datetime]::Parse('$((Get-Date).ToUniversalTime().ToString('o'))').ToUniversalTime(); `$g=(Get-Date).ToUniversalTime(); 'guest clock skew: {0:N0} s' -f (`$g-`$h).TotalSeconds"

  # ---- inputs in (the guest dir is not in the snapshot; created fresh)
  $null = GuestPs "New-Item -ItemType Directory -Force -Path '$FreshGuestDir\in','$FreshGuestDir\fresh_install' | Out-Null"
  Log 'copying installer(s) and tests/fresh_install in'
  CopyTo $Installer "C:/fi/in/$newName"
  if ($Mode -eq 'upgrade') { CopyTo $OldInstaller "C:/fi/in/$oldName" }
  $tarball = Join-Path $work 'fresh_install.tar'
  & $TarExe -cf $tarball --exclude node_modules --exclude evidence -C $FiDir .
  if ($LASTEXITCODE -ne 0) { throw 'tar of tests/fresh_install failed' }
  CopyTo $tarball 'C:/fi/fresh_install.tar'
  $null = GuestCmd "tar -xf C:\fi\fresh_install.tar -C C:\fi\fresh_install && copy /y C:\fi\fresh_install\vbox\guest-run.ps1 C:\fi\ >NUL && copy /y C:\fi\fresh_install\vbox\guest-start.ps1 C:\fi\ >NUL"
  if ($LASTEXITCODE -ne 0) { throw 'unpacking tests/fresh_install in the guest failed' }
  $argFile = Join-Path $work 'run-args.txt'
  [IO.File]::WriteAllText($argFile, ($scen -join ' '))
  CopyTo $argFile 'C:/fi/run-args.txt'

  # ---- start the scenario in tester's desktop session (key on stdin)
  # Fingerprint only (length + 8 hex of sha256), never the key: compare with
  # guest-run's line in console.log to prove the key arrived unchanged.
  $kb = [Text.Encoding]::UTF8.GetBytes($env:ORBITAL_TEST_OPENCODE_KEY)
  $kh = -join ([Security.Cryptography.SHA256]::Create().ComputeHash($kb)[0..3] | ForEach-Object { $_.ToString('x2') })
  Log "key fingerprint on the runner: len=$($kb.Length) sha256=$kh..."
  Log 'starting the scenario in the guest desktop session (this takes a while)'
  # No BOM on the pipe (the runner's PowerShell otherwise prefixes EF BB BF).
  $OutputEncoding = New-Object Text.UTF8Encoding $false
  $env:ORBITAL_TEST_OPENCODE_KEY | & $SshExe @SshOpts -p $FreshSshPort "$FreshVmUser@127.0.0.1" "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File C:\fi\guest-start.ps1"
  $startRc = $LASTEXITCODE
  if ($startRc -ne 0) { throw "guest-start.ps1 exited $startRc" }

  # ---- poll; stream the guest console.log as it grows
  $deadline = (Get-Date).AddMinutes($TimeoutMin)
  $offset = 0
  $done = $false
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $chunk = GuestPs @"
`$f='C:\fi\state\console.log'; `$n=0; if (Test-Path `$f) { `$s=New-Object IO.FileStream `$f,'Open','Read','ReadWrite'; `$ms=New-Object IO.MemoryStream; `$s.CopyTo(`$ms); `$s.Close(); `$b=`$ms.ToArray(); `$n=`$b.Length; if (`$n -gt $offset) { [Console]::Out.Write([Text.Encoding]::UTF8.GetString(`$b, $offset, `$n - $offset)) } }
[Console]::Out.Write("``n@@META `$n " + (Test-Path 'C:\fi\state\done'))
"@
    if ($LASTEXITCODE -ne 0 -or -not $chunk) { continue }   # a dropped SSH poll is not fatal
    $lines = @($chunk)
    $meta = $lines[-1]
    if ($lines.Count -gt 1) { Write-Host (($lines[0..($lines.Count - 2)]) -join "`n") }
    if ($meta -match '^@@META (\d+) (True|False)$') {
      $offset = [int]$Matches[1]
      if ($Matches[2] -eq 'True') { $done = $true; break }
    }
  }
  if (-not $done) {
    Log "scenario did not finish within $TimeoutMin min"
    $rc = 6
  } else {
    $code = (GuestCmd 'type C:\fi\state\exitcode' | Out-String).Trim()
    $rc = if ($code -match '^-?\d+$') { [int]$code } else { 5 }
    Log "scenario exit code $rc"
  }
} catch {
  Log "rig error: $_"
  $rc = 7
} finally {
  $ErrorActionPreference = 'Continue'
  # ---- evidence out (already scrubbed inside by scenario.mjs), whatever happened
  try {
    if ((Get-VmState) -eq 'running') {
      $null = GuestCmd 'if exist C:\fi\evidence (tar -czf C:\fi\evidence.tgz -C C:\fi evidence)'
      $tgz = Join-Path $work 'evidence.tgz'
      if (CopyFrom 'C:/fi/evidence.tgz' $tgz) {
        & $TarExe -xzf $tgz -C $Out --strip-components 1
      }
      $null = CopyFrom 'C:/fi/state/console.log' $consoleLocal
      $null = CopyFrom 'C:/fi/state/npm-ci.log' (Join-Path $Out 'npm-ci.log')
    }
  } catch { Log "copying evidence out failed: $_" }

  # ---- scrub-check here too; a surviving copy of the key deletes the evidence
  & node (Join-Path $FiDir 'scrub.mjs') $Out --require-key
  if ($LASTEXITCODE -ne 0) {
    Log 'KEY FOUND in evidence after scrub (or scrub failed); deleting the copied evidence'
    Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
    $rc = 4
  } else {
    Log "evidence: $Out"
  }

  # ---- back to clean, whatever happened
  if ($Keep -and $rc -ne 0) {
    Log "-Keep: leaving $FreshVm running (restore with: VBoxManage controlvm $FreshVm poweroff; VBoxManage snapshot $FreshVm restore $FreshSnapshot)"
  } else {
    try { Restore-Clean; Log "$FreshVm is back at snapshot $FreshSnapshot" } catch { Log "restore failed: $_"; if ($rc -eq 0) { $rc = 7 } }
  }
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
exit $rc

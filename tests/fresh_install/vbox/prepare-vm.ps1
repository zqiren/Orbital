# Build the fresh-install VM once (the Windows twin of tart/prepare-golden.sh):
# VirtualBox VM `orbital-fresh-win`, Windows 11 x64 from the Simplified
# Chinese ISO, Home edition with no product key, EFI + TPM 2.0, NAT with one
# port forward 127.0.0.1:2222 -> guest 22. Unattended install with local
# admin `tester` (auto-logon), Guest Additions, OpenSSH (rig key only), then
# guest-provision.ps1 over SSH (Node 24, Codex, dsh, Playwright Chromium; no
# Orbital), a clean shutdown and snapshot `clean`.
#
#   powershell -ExecutionPolicy Bypass -File tests\fresh_install\vbox\prepare-vm.ps1 `
#     -Iso <Win11 zh-cn .iso> -AdditionsIso <VBoxGuestAdditions .iso> -NodeMsi <node-v24 x64 .msi> -OpenSshMsi <OpenSSH-Win64 .msi>
#
# Options: -MemoryMB (8192) -Cpus (4) -DiskGB (64) -ImageIndex (auto: Home
# China, else Home) -Force (delete an existing VM of that name first)
# -SkipInstall (VM exists and is installed: just re-provision + re-snapshot).
# Windows PowerShell 5.1; ASCII only. Runs as the Windows user that will own
# the VM (the runner's user). No admin needed once VirtualBox is installed.
param(
  [string]$Iso,
  [string]$AdditionsIso,
  [string]$NodeMsi,
  [string]$OpenSshMsi,
  [int]$MemoryMB = 8192,
  [int]$Cpus = 4,
  [int]$DiskGB = 64,
  [int]$ImageIndex = 0,
  [int]$ServePort = 8399,
  [int]$InstallTimeoutMin = 120,
  [switch]$Force,
  [switch]$SkipInstall
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'common.ps1')
# Continue, not Stop: in 5.1 a native command's stderr would otherwise throw.
# VBox / Wait-* / CopyTo throw on failure; the rest is checked explicitly.
$ErrorActionPreference = 'Continue'
trap { Log "FAILED: $_"; try { Stop-Transcript | Out-Null } catch {}; exit 1 }

New-Item -ItemType Directory -Force -Path $FreshStateDir, (Join-Path $FreshStateDir 'logs') | Out-Null
$logFile = Join-Path $FreshStateDir ("logs\prepare-vm-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $logFile | Out-Null

# ---- rig key + tester password (state dir, readable by this user only)
if (-not (Test-Path $FreshSshKey)) {
  Log "generating the rig key $FreshSshKey"
  & cmd.exe /c "`"$env:SystemRoot\System32\OpenSSH\ssh-keygen.exe`" -q -t ed25519 -N `"`" -C orbital-fresh-install -f `"$FreshSshKey`""
  if ($LASTEXITCODE -ne 0) { throw 'ssh-keygen failed' }
}
& icacls.exe $FreshSshKey /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
$pwFile = Join-Path $FreshStateDir 'tester.password'
if (-not (Test-Path $pwFile)) {
  $chars = [char[]]'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  $rng = New-Object Security.Cryptography.RNGCryptoServiceProvider
  $bytes = New-Object byte[] 20
  $rng.GetBytes($bytes)
  $pw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
  [IO.File]::WriteAllText($pwFile, $pw)
  & icacls.exe $pwFile /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
}

$vmExists = (& $VBoxManage list vms) -match "^`"$FreshVm`" "
if (-not $SkipInstall) {
  foreach ($f in @($Iso, $AdditionsIso, $NodeMsi, $OpenSshMsi)) {
    if (-not $f -or -not (Test-Path $f)) { throw "missing input: '$f' (need -Iso -AdditionsIso -NodeMsi -OpenSshMsi)" }
  }
  if ($vmExists) {
    if (-not $Force) { throw "VM $FreshVm already exists (use -Force to delete and rebuild, or -SkipInstall)" }
    Log "deleting the existing $FreshVm"
    Stop-Vm
    $null = VBox unregistervm $FreshVm --delete
  }

  # ---- the VM
  Log "creating ${FreshVm}: $MemoryMB MB, $Cpus vCPU, $DiskGB GB dynamic disk, EFI + TPM 2.0"
  $null = VBox createvm --name $FreshVm --ostype Windows11_64 --register
  $null = VBox modifyvm $FreshVm --memory $MemoryMB --cpus $Cpus --firmware efi --tpm-type 2.0 `
    --graphicscontroller vboxsvga --vram 128 --ioapic on --rtc-use-utc on `
    --audio-enabled off --clipboard-mode disabled --drag-and-drop disabled `
    --usb-ohci off --usb-ehci off --usb-xhci off `
    --nic1 nat --nat-localhostreachable1 on
  # One port forward, bound to the host's loopback only (nothing on the LAN).
  $null = VBox modifyvm $FreshVm --natpf1 "ssh,tcp,127.0.0.1,$FreshSshPort,,22"
  # Secure Boot with Microsoft's keys (best effort; the unattended setup skips the check anyway).
  $sb = (VBoxTry modifynvram $FreshVm inituefivarstore) -and (VBoxTry modifynvram $FreshVm enrollmssignatures) -and
        (VBoxTry modifynvram $FreshVm enrollorclpk) -and (VBoxTry modifynvram $FreshVm secureboot --enable)
  Log "secure boot enabled: $sb"

  $cfg = ((& $VBoxManage showvminfo $FreshVm --machinereadable) | Where-Object { $_ -match '^CfgFile=' }) -replace '^CfgFile="(.*)"$', '$1'
  $disk = Join-Path (Split-Path -Parent $cfg) "$FreshVm.vdi"
  $null = VBox createmedium disk --filename $disk --size ($DiskGB * 1024) --format VDI --variant Standard
  $null = VBox storagectl $FreshVm --name SATA --add sata --controller IntelAhci --portcount 4 --hostiocache on
  $null = VBox storageattach $FreshVm --storagectl SATA --port 0 --device 0 --type hdd --medium $disk

  # ---- which image: Home China (CoreCountrySpecific, what Chinese PCs ship) else Home (Core)
  $detect = & $VBoxManage unattended detect --iso=$Iso --machine-readable 2>&1
  $detect | ForEach-Object { Log "detect: $_" }
  if (-not $ImageIndex) {
    $names = @{}
    foreach ($l in $detect) {
      if ("$l" -match '^ImageIndex(\d+)\s*=\s*"?(\d+)"?') { if (-not $names.ContainsKey($Matches[1])) { $names[$Matches[1]] = @{} }; $names[$Matches[1]].index = [int]$Matches[2] }
      if ("$l" -match '^ImageName(\d+)\s*=\s*"?(.*?)"?$') { if (-not $names.ContainsKey($Matches[1])) { $names[$Matches[1]] = @{} }; $names[$Matches[1]].name = $Matches[2] }
    }
    $homeCn = [string]::new([char[]](0x5BB6, 0x5EAD, 0x4E2D, 0x6587, 0x7248))   # Home China, as the zh-CN image names it
    $homeZh = [string]::new([char[]](0x5BB6, 0x5EAD, 0x7248))                   # Home
    $pick = $null
    foreach ($want in @($homeCn, 'Home China', 'CoreCountrySpecific', $homeZh, 'Home')) {
      foreach ($k in $names.Keys) {
        $n = "$($names[$k].name)"
        if (-not $pick -and $n -and ($n -match [regex]::Escape($want) + '(\s|$|\()') -and $n -notmatch ' N\b|Single Language|SingleLanguage') { $pick = $names[$k] }
      }
      if ($pick) { break }
    }
    if (-not $pick) { throw 'could not find a Home image on the ISO: pass -ImageIndex (see the detect lines above)' }
    $ImageIndex = if ($pick.index) { $pick.index } else { 1 }
    Log "image: #$ImageIndex"
  }

  # ---- what the guest fetches at first logon (loopback-only server; see serve.mjs)
  $serve = Join-Path $FreshStateDir 'serve'
  Remove-Item -Recurse -Force $serve -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $serve | Out-Null
  Copy-Item (Join-Path $here 'guest-firstlogon.ps1') $serve
  Copy-Item "$FreshSshKey.pub" (Join-Path $serve 'rig.pub')
  Copy-Item $pwFile (Join-Path $serve 'tester.password')
  Copy-Item $OpenSshMsi (Join-Path $serve 'OpenSSH-Win64.msi')
  $server = Start-Process node -ArgumentList @("`"$(Join-Path $here 'serve.mjs')`"", "`"$serve`"", $ServePort) -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $FreshStateDir 'logs\serve.out.log') -RedirectStandardError (Join-Path $FreshStateDir 'logs\serve.err.log')
  # Post-install command (runs at the first logon, elevated): fetch + run guest-firstlogon.ps1.
  # Encoded, because VirtualBox pastes it into a .cmd file inside an echo line.
  $boot = "`$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force C:\fi-setup | Out-Null; " +
          "for(`$i=0;`$i -lt 60;`$i++){ try { Invoke-WebRequest -UseBasicParsing http://10.0.2.2:$ServePort/guest-firstlogon.ps1 -OutFile C:\fi-setup\guest-firstlogon.ps1; break } catch { Start-Sleep 5 } }; " +
          "& C:\fi-setup\guest-firstlogon.ps1 -Base http://10.0.2.2:$ServePort *> C:\fi-setup\firstlogon.log"
  $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($boot))

  $template = Join-Path $here 'autounattend-template.xml'
  try {
    Log "unattended install (zh-CN, image #$ImageIndex, user $FreshVmUser, Guest Additions)"
    $null = VBox unattended install $FreshVm --iso=$Iso --user=$FreshVmUser --user-password-file=$pwFile `
      --admin-password-file=$pwFile --full-user-name=$FreshVmUser --image-index=$ImageIndex `
      --locale=zh_CN --country=CN --language=zh-CN --time-zone=Asia/Shanghai --hostname=orbital-fresh.local `
      --install-additions --additions-iso=$AdditionsIso `
      --script-template=$template `
      --post-install-command="powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $enc" `
      --start-vm=headless
    Log "waiting for the install + first logon (up to $InstallTimeoutMin min; SSH comes up at the end)"
    Wait-Ssh ($InstallTimeoutMin * 60)
    Log 'SSH with the rig key works'
  } finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Remove-Item -Recurse -Force $serve -ErrorAction SilentlyContinue
  }
} else {
  if (-not $vmExists) { throw "no VM $FreshVm to re-provision" }
  if ((Get-VmState) -ne 'running') { $null = VBox startvm $FreshVm --type headless }
  Wait-Ssh 900
}

# ---- provision over SSH (as tester, elevated)
Wait-Desktop 600
$null = GuestPs "New-Item -ItemType Directory -Force -Path C:\fi-setup | Out-Null"
if ($NodeMsi) { CopyTo $NodeMsi 'C:/fi-setup/node.msi' }
CopyTo (Join-Path $here 'guest-provision.ps1') 'C:/fi-setup/guest-provision.ps1'
Log 'provisioning the guest (Node 24, Codex, dsh, Playwright Chromium)'
GuestPsFile 'C:\fi-setup\guest-provision.ps1'
if ($LASTEXITCODE -ne 0) { throw "guest-provision.ps1 failed ($LASTEXITCODE)" }
# Leave no rig files behind (C:\fi is created fresh by every run).
$null = GuestPs "Remove-Item -Recurse -Force C:\fi-setup, C:\fi -ErrorAction SilentlyContinue; Remove-Item -Force C:\vboxpostinstall.log -ErrorAction SilentlyContinue"

# ---- clean shutdown, guest can no longer reach the host's loopback, snapshot
Log 'shutting the guest down cleanly'
$null = GuestCmd 'shutdown /s /t 5 /f'
for ($i = 0; $i -lt 90 -and (Get-VmState) -ne 'poweroff'; $i++) { Start-Sleep -Seconds 4 }
if ((Get-VmState) -ne 'poweroff') { throw 'guest did not power off within 6 min' }
Start-Sleep -Seconds 3
# Detach the install media (unattended attached them) so runs boot from disk only.
$info = & $VBoxManage showvminfo $FreshVm --machinereadable
foreach ($l in $info) {
  if ($l -match '^"?([^"=]+)-(\d+)-(\d+)"?="(.+\.(iso|viso))"$') {
    $null = VBoxTry storageattach $FreshVm --storagectl $Matches[1] --port $Matches[2] --device $Matches[3] --medium emptydrive
  }
}
$null = VBox modifyvm $FreshVm --nat-localhostreachable1 off
if ((& $VBoxManage snapshot $FreshVm list --machinereadable 2>$null) -match "=`"$FreshSnapshot`"") {
  Log "deleting the old snapshot $FreshSnapshot"
  $null = VBox snapshot $FreshVm delete $FreshSnapshot
}
$null = VBox snapshot $FreshVm take $FreshSnapshot --description "Win11 zh-CN Home, tester auto-logon, OpenSSH (rig key), Node 24, Codex, dsh 0.2.2-rc.1, Playwright Chromium; no Orbital. $(Get-Date -Format s)"
Log "snapshot $FreshSnapshot taken; $FreshVm is ready"
Stop-Transcript | Out-Null

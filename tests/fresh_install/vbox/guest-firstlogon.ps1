# Runs INSIDE the guest once, at tester's first logon (VirtualBox unattended
# post-install command, elevated). Fetches its inputs from prepare-vm.ps1's
# loopback web server on the host (10.0.2.2 from the guest; only reachable
# while prepare-vm turns nat-localhostreachable on) and makes the guest
# reachable over SSH with the rig key. Everything else happens over SSH in
# guest-provision.ps1. ASCII only (Windows PowerShell 5.1).
param([string]$Base = 'http://10.0.2.2:8399')
$ErrorActionPreference = 'Stop'
$dir = 'C:\fi-setup'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Start-Transcript -Path "$dir\firstlogon-transcript.log" -Append | Out-Null

function Get-RigFile([string]$name) {
  for ($i = 1; $i -le 30; $i++) {
    try { Invoke-WebRequest -UseBasicParsing "$Base/$name" -OutFile "$dir\$name"; return }
    catch { Write-Host "fetch $name failed (try $i): $_"; Start-Sleep -Seconds 5 }
  }
  throw "could not fetch $name from $Base"
}
Get-RigFile 'rig.pub'
Get-RigFile 'tester.password'
Get-RigFile 'OpenSSH-Win64.msi'

# --- OpenSSH Server (Win32-OpenSSH MSI: no Windows Update / FoD dependency)
$p = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @('/i', "$dir\OpenSSH-Win64.msi", 'ADDLOCAL=Server', '/qn', '/norestart', '/l*v', "$dir\openssh-msi.log")
Write-Host "OpenSSH msi exit $($p.ExitCode)"
if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { throw "OpenSSH install failed ($($p.ExitCode))" }
Set-Service sshd -StartupType Automatic
Start-Service sshd          # first start writes C:\ProgramData\ssh\sshd_config
Stop-Service sshd

$cfgPath = 'C:\ProgramData\ssh\sshd_config'
$cfg = Get-Content $cfgPath
# First value wins in sshd_config: put the rig's settings on top.
$top = @('# orbital fresh-install rig: key-only', 'PasswordAuthentication no', 'KbdInteractiveAuthentication no', 'PubkeyAuthentication yes', '')
Set-Content -Path $cfgPath -Value ($top + $cfg) -Encoding Ascii
if (-not ($cfg -match '^\s*Match Group administrators')) {
  Add-Content -Path $cfgPath -Value @('', 'Match Group administrators', '       AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys') -Encoding Ascii
}

# tester is an admin, so its key lives in administrators_authorized_keys,
# readable only by Administrators + SYSTEM. SIDs, not names: group names are
# translated on Chinese Windows.
$ak = 'C:\ProgramData\ssh\administrators_authorized_keys'
Set-Content -Path $ak -Value (Get-Content "$dir\rig.pub") -Encoding Ascii
& icacls.exe $ak /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Host
Start-Service sshd
& netsh.exe advfirewall firewall add rule name=orbital-fresh-sshd dir=in action=allow protocol=TCP localport=22 | Out-Host

# --- auto-logon to the desktop on every boot (Orbital's window needs a real desktop)
$wl = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
$pw = (Get-Content "$dir\tester.password" -Raw).Trim()
Set-ItemProperty $wl -Name AutoAdminLogon -Value '1' -Type String
Set-ItemProperty $wl -Name DefaultUserName -Value $env:USERNAME -Type String
Set-ItemProperty $wl -Name DefaultDomainName -Value $env:COMPUTERNAME -Type String
Set-ItemProperty $wl -Name DefaultPassword -Value $pw -Type String
Remove-ItemProperty $wl -Name AutoLogonCount -ErrorAction SilentlyContinue
Remove-Item "$dir\tester.password" -Force
# Windows Hello / passwordless-only sign-in would hide the password logon.
$pl = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device'
if (Test-Path $pl) { Set-ItemProperty $pl -Name DevicePasswordLessBuildVersion -Value 0 -Type DWord }

# --- never sleep, never lock, display always on
foreach ($a in @('monitor-timeout-ac', 'monitor-timeout-dc', 'standby-timeout-ac', 'standby-timeout-dc', 'hibernate-timeout-ac', 'hibernate-timeout-dc', 'disk-timeout-ac')) {
  & powercfg.exe /change $a 0
}
& powercfg.exe /hibernate off
# "Require sign-in after waking": never.
& powercfg.exe /SETACVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0
& powercfg.exe /SETDCVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0
& powercfg.exe /SETACTIVE SCHEME_CURRENT
$pers = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\Personalization'
New-Item -Force -Path $pers | Out-Null
Set-ItemProperty $pers -Name NoLockScreen -Value 1 -Type DWord
$sys = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System'
Set-ItemProperty $sys -Name InactivityTimeoutSecs -Value 0 -Type DWord

New-Item -ItemType File -Force -Path "$dir\firstlogon-done" | Out-Null
Write-Host 'first logon setup OK'
Stop-Transcript | Out-Null

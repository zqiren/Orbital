# Runs INSIDE the guest over SSH (as tester), from run.ps1. The test key
# arrives on stdin (never on a command line), is written to a file only
# tester (and SYSTEM) can read, and the scenario is started as a scheduled
# task in tester's DESKTOP session: anything started straight from SSH runs
# outside the logged-in desktop, so the Orbital window would never appear
# and the desktop screenshot / window check would come back empty. Highest
# run level, because the Inno installer needs admin. guest-run.ps1 reads the
# key into its env and deletes the file before node starts.
# ASCII only (Windows PowerShell 5.1).
$ErrorActionPreference = 'Stop'
$fi = 'C:\fi'
# Read stdin as raw bytes and decode as UTF-8 ourselves: [Console]::In
# decodes with the guest's ANSI code page (GBK), which turns a UTF-8 BOM
# from the runner's PowerShell pipe into CJK characters fused with the
# key's first byte. Strip any BOM and whitespace.
$in = [Console]::OpenStandardInput()
$ms = New-Object IO.MemoryStream
$buf = New-Object byte[] 4096
while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) { $ms.Write($buf, 0, $n) }
$key = [Text.Encoding]::UTF8.GetString($ms.ToArray()).Split("`n")[0].Trim().TrimStart([char]0xFEFF).Trim()
$ms = $null; $buf = $null
if (-not $key) { Write-Host 'no key on stdin'; exit 2 }

$secret = Join-Path $fi 'secret'
New-Item -ItemType Directory -Force -Path $secret | Out-Null
& icacls.exe $secret /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F" /grant:r '*S-1-5-18:(OI)(CI)F' | Out-Null
[IO.File]::WriteAllText((Join-Path $secret 'key'), $key.Trim())
$key = $null

$state = Join-Path $fi 'state'
Remove-Item -Recurse -Force $state -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $state | Out-Null

# The scenario must run with the token of tester's desktop shell, the one a
# user's double-click gives. A process started by Task Scheduler gets a
# logon session of its own, and in it Python keyring's CredWrite fails with
# 1312 (ERROR_NO_SUCH_LOGON_SESSION): Orbital launched that way could not
# save a provider card, and the installer's --setup-sandbox could not store
# the sandbox password. Orbital launched through explorer stores both fine
# (probed on this VM). So the task only asks the desktop's explorer to open
# guest-run.cmd; explorer starts it with the shell's own token (not elevated).
#
# The installer then elevates itself as it does for a user. Its UAC "Yes"
# is given by setting admin consent to "elevate without prompting" here, in
# the throwaway guest only (the snapshot is restored after every run).
$uac = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System'
Set-ItemProperty $uac -Name ConsentPromptBehaviorAdmin -Value 0 -Type DWord
Set-ItemProperty $uac -Name PromptOnSecureDesktop -Value 0 -Type DWord

$cmd = Join-Path $fi 'guest-run.cmd'
[IO.File]::WriteAllText($cmd, "@start `"`" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $fi\guest-run.ps1`r`n")
$name = 'orbital-fresh-run'
$action = New-ScheduledTaskAction -Execute 'explorer.exe' -Argument $cmd -WorkingDirectory $fi
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $name
$console = Join-Path $state 'console.log'
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 2
  if (Test-Path $console) { Write-Host 'guest-run started through the desktop shell'; exit 0 }
}
$ti = Get-ScheduledTaskInfo -TaskName $name
Write-Host "guest-run did not start within 60 s (launcher task last result $($ti.LastTaskResult))"
Remove-Item -Force (Join-Path $secret 'key') -ErrorAction SilentlyContinue
exit 3

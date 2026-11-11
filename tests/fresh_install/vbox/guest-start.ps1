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
$key = [Console]::In.ReadLine()
if (-not $key -or -not $key.Trim()) { Write-Host 'no key on stdin'; exit 2 }

$secret = Join-Path $fi 'secret'
New-Item -ItemType Directory -Force -Path $secret | Out-Null
& icacls.exe $secret /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F" /grant:r '*S-1-5-18:(OI)(CI)F' | Out-Null
[IO.File]::WriteAllText((Join-Path $secret 'key'), $key.Trim())
$key = $null

$state = Join-Path $fi 'state'
Remove-Item -Recurse -Force $state -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $state | Out-Null

$name = 'orbital-fresh-run'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $fi\guest-run.ps1" -WorkingDirectory $fi
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 3
$t = Get-ScheduledTask -TaskName $name
Write-Host "scheduled task ${name}: $($t.State)"
if ($t.State -ne 'Running' -and -not (Test-Path (Join-Path $state 'done'))) {
  $i = Get-ScheduledTaskInfo -TaskName $name
  Write-Host "task did not start: last result $($i.LastTaskResult)"
  Remove-Item -Force (Join-Path $secret 'key') -ErrorAction SilentlyContinue
  exit 3
}
exit 0

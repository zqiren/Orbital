# Install the GitHub Actions runner for the Windows fresh-install VM on this
# PC: registered to the PRIVATE repo zqiren/orbital-vm-ci (never the public
# Orbital repo) with the extra label orbital-winvm, running as THIS Windows
# user (VirtualBox VMs belong to a user; a service account can't see them),
# started at logon from HKCU\...\Run (no admin, no service). Jobs get a
# portable Node 24 first on PATH (the runner's .path), next to git and gh.
#
#   powershell -ExecutionPolicy Bypass -File setup-runner.ps1 -Zip <actions-runner-win-x64-*.zip> -NodeZip <node-v24.*-win-x64.zip>
# Needs gh logged in as a repo admin (mints the registration token; never printed).
# Windows PowerShell 5.1; ASCII only.
param(
  [string]$Zip,
  [string]$NodeZip,
  [string]$Dir = 'D:\actions-runner-winvm',
  [string]$Repo = 'zqiren/orbital-vm-ci',
  [string]$Name = "$env:COMPUTERNAME-winvm",
  [string]$Labels = 'orbital-winvm',
  [string]$Proxy = ''          # e.g. http://127.0.0.1:7890 if GitHub is only reachable through it
)
$ErrorActionPreference = 'Continue'
function Fail([string]$m) { Write-Host "setup-runner FAILED: $m"; exit 1 }

if (-not (Test-Path "$Dir\config.cmd")) {
  if (-not $Zip -or -not (Test-Path $Zip)) { Fail "need -Zip (actions-runner-win-x64-*.zip)" }
  New-Item -ItemType Directory -Force -Path $Dir | Out-Null
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dir)
}

# Portable Node 24 for jobs (the PC's own node stays as it is).
$nodeDir = $null
if ($NodeZip) {
  $root = Join-Path (Split-Path -Parent $Dir) 'node24'
  $nodeDir = Join-Path $root ([IO.Path]::GetFileNameWithoutExtension($NodeZip))
  if (-not (Test-Path "$nodeDir\node.exe")) {
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($NodeZip, $root)
  }
  & "$nodeDir\node.exe" --version
}

if (-not (Test-Path "$Dir\.runner")) {
  $token = & gh api -X POST "repos/$Repo/actions/runners/registration-token" -q .token
  if ($LASTEXITCODE -ne 0 -or -not $token) { Fail "could not get a registration token for $Repo (gh auth?)" }
  Push-Location $Dir
  & .\config.cmd --unattended --url "https://github.com/$Repo" --token $token --name $Name --labels $Labels --work _work --replace
  $rc = $LASTEXITCODE
  Pop-Location
  $token = $null
  if ($rc -ne 0) { Fail "config.cmd exited $rc" }
}

# PATH for jobs: Node 24 first, then this user's normal PATH (git, gh).
$path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
if ($nodeDir) { $path = "$nodeDir;$path" }
[IO.File]::WriteAllText("$Dir\.path", $path)
if ($Proxy) {
  [IO.File]::WriteAllText("$Dir\.env", "https_proxy=$Proxy`nhttp_proxy=$Proxy`nno_proxy=127.0.0.1,localhost`n")
}

# Start at logon, hidden, as this user.
$cmd = "powershell.exe -NoProfile -WindowStyle Hidden -Command `"Start-Process -WindowStyle Hidden -FilePath '$Dir\run.cmd' -WorkingDirectory '$Dir'`""
Set-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'orbital-winvm-runner' -Value $cmd
Write-Host "logon start: HKCU Run\orbital-winvm-runner"

if (-not (Get-Process Runner.Listener -ErrorAction SilentlyContinue)) {
  Start-Process -WindowStyle Hidden -FilePath "$Dir\run.cmd" -WorkingDirectory $Dir
  Write-Host 'runner started'
} else { Write-Host 'runner already running' }

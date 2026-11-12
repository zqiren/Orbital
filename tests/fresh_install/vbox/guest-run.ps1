# Runs INSIDE the guest as the scheduled task guest-start.ps1 registers: in
# tester's desktop session, elevated. Reads the key file into the env and
# deletes it before node starts, runs `npm ci` + scenario.mjs with the args
# run.ps1 wrote to C:\fi\run-args.txt, and leaves console.log, exitcode and
# a `done` marker in C:\fi\state for the host to poll.
# ASCII only (Windows PowerShell 5.1).
$ErrorActionPreference = 'Continue'
$fi = 'C:\fi'
$state = Join-Path $fi 'state'
$console = Join-Path $state 'console.log'
New-Item -ItemType Directory -Force -Path $state | Out-Null
function Note([string]$m) { [IO.File]::AppendAllText($console, "[guest-run] $m`r`n") }
$rc = 5
try {
  $keyFile = Join-Path $fi 'secret\key'
  if (-not (Test-Path $keyFile)) { throw 'no key file' }
  $env:ORBITAL_TEST_OPENCODE_KEY = [IO.File]::ReadAllText($keyFile).Trim()
  Remove-Item -Force $keyFile
  if (Test-Path $keyFile) { throw 'could not delete the key file' }
  $kb = [Text.Encoding]::UTF8.GetBytes($env:ORBITAL_TEST_OPENCODE_KEY)
  $kh = -join ([Security.Cryptography.SHA256]::Create().ComputeHash($kb)[0..3] | ForEach-Object { $_.ToString('x2') })
  Note "key fingerprint in the guest: len=$($kb.Length) sha256=$kh..."
  $env:AGENT_OS_TELEMETRY_DISABLED = '1'
  # A task gets the env from the registry; make sure npm's prefix is on PATH.
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
  $env:NPM_CONFIG_PREFIX = [Environment]::GetEnvironmentVariable('NPM_CONFIG_PREFIX', 'User')
  $session = (Get-Process -Id $PID).SessionId
  Note "session $session, user $env:USERNAME, node $(& node --version)"

  Set-Location (Join-Path $fi 'fresh_install')
  & cmd.exe /c "npm ci --no-audit --no-fund > `"$state\npm-ci.log`" 2>&1"
  if ($LASTEXITCODE -ne 0) {
    Note "npm ci failed ($LASTEXITCODE); see npm-ci.log"
    $rc = 3
  } else {
    $argline = (Get-Content -Raw (Join-Path $fi 'run-args.txt')).Trim()
    Note "node scenario.mjs $argline"
    & cmd.exe /c "node scenario.mjs $argline >> `"$console`" 2>&1"
    $rc = $LASTEXITCODE
    Note "scenario exit code $rc"
  }
} catch {
  Note "error: $_"
} finally {
  Remove-Item Env:\ORBITAL_TEST_OPENCODE_KEY -ErrorAction SilentlyContinue
  Remove-Item -Force (Join-Path $fi 'secret\key') -ErrorAction SilentlyContinue
  [IO.File]::WriteAllText((Join-Path $state 'exitcode'), "$rc")
  [IO.File]::WriteAllText((Join-Path $state 'done'), (Get-Date).ToString('o'))
}
exit $rc

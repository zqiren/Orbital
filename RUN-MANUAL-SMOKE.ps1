# RUN-MANUAL-SMOKE.ps1
#
# Boots the integrated build (test/full-integration worktree) for manual smoke testing.
#
# - Strips CLAUDECODE / CLAUDE_CODE_ENTRYPOINT so the daemon child process does NOT
#   inherit Claude Code env vars (rules out loop-prevention as a confound).
# - Daemon on http://127.0.0.1:8000 from the integrated worktree's code.
# - Vite on http://127.0.0.1:5173 from the integrated worktree's web/.
# - Run this from a fresh PowerShell (not from inside Claude Code) when possible.
#   The daemon will work either way, but a non-Claude-Code parent is the cleanest
#   environmental control.
#
# Usage: .\RUN-MANUAL-SMOKE.ps1
# Stop:  .\RUN-MANUAL-SMOKE.ps1 -Stop

param(
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$IntegratedWorktree = "D:\orbital-public\.claude\worktrees\test-full-integration"
$DaemonLog = "$env:TEMP\orbital-smoke-daemon.log"
$ViteLog   = "$env:TEMP\orbital-smoke-vite.log"
$ReproWorkspace = "D:\repro-smoke"

function Stop-OrbitalDaemonAndVite {
    Write-Host "=== Stopping daemon (port 8000) and vite (port 5173) ==="
    $listeners = netstat -ano | Select-String -Pattern ':8000 .*LISTEN', ':5173 .*LISTEN'
    foreach ($line in $listeners) {
        $parts = ($line -split '\s+') | Where-Object { $_ -ne '' }
        $pidToKill = $parts[-1]
        if ($pidToKill -and $pidToKill -ne '0') {
            Write-Host "  Killing PID $pidToKill"
            taskkill /F /PID $pidToKill 2>&1 | Out-Null
        }
    }
    Start-Sleep -Seconds 2
    $remaining = (netstat -ano | Select-String -Pattern ':8000 .*LISTEN', ':5173 .*LISTEN').Count
    Write-Host "  Remaining listeners on 8000/5173: $remaining"
}

if ($Stop) {
    Stop-OrbitalDaemonAndVite
    Write-Host ""
    Write-Host "Done. Logs preserved at:"
    Write-Host "  daemon: $DaemonLog"
    Write-Host "  vite:   $ViteLog"
    exit 0
}

# --- Pre-flight checks ---
Write-Host "=== Pre-flight ==="

if (-not (Test-Path $IntegratedWorktree)) {
    Write-Host "ERROR: Integrated worktree missing at $IntegratedWorktree"
    Write-Host "Re-run the integrated-build dispatch to recreate it."
    exit 1
}
Push-Location $IntegratedWorktree
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
$head   = (& git rev-parse HEAD).Trim()
Pop-Location
if ($branch -ne "test/full-integration") {
    Write-Host "ERROR: Worktree is on branch '$branch', expected 'test/full-integration'"
    exit 1
}
Write-Host "  Worktree OK: $branch @ $($head.Substring(0,7))"

$settingsJson = "$IntegratedWorktree\orbital-data\settings.json"
if (-not (Test-Path $settingsJson)) {
    Write-Host "ERROR: Missing $settingsJson"
    Write-Host "It should contain provider=moonshot, model=kimi-k2.5, base_url=https://api.moonshot.cn/v1"
    exit 1
}
$settings = Get-Content $settingsJson -Raw | ConvertFrom-Json
Write-Host "  Settings OK: provider=$($settings.llm.provider) model=$($settings.llm.model)"

if (-not (Test-Path "$ReproWorkspace\orbital\PROJECT_STATE.md")) {
    Write-Host "ERROR: Missing seeded workspace at $ReproWorkspace\orbital\"
    Write-Host "Re-run the integrated-build dispatch to recreate it."
    exit 1
}
Write-Host "  Workspace OK: $ReproWorkspace (seeded with PROJECT_STATE.md, DECISIONS.md, LESSONS.md)"

Stop-OrbitalDaemonAndVite

# --- Start daemon (no CLAUDECODE inheritance) ---
Write-Host ""
Write-Host "=== Starting daemon (CLAUDECODE/CLAUDE_CODE_ENTRYPOINT stripped) ==="
$daemonEnv = @{}
foreach ($entry in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
    if ($entry.Key -ne "CLAUDECODE" -and $entry.Key -ne "CLAUDE_CODE_ENTRYPOINT") {
        $daemonEnv[$entry.Key] = $entry.Value
    }
}
$daemonProc = Start-Process -FilePath "python" `
    -ArgumentList "-m","uvicorn","agent_os.api.app:create_app","--factory","--port","8000","--host","0.0.0.0" `
    -WorkingDirectory $IntegratedWorktree `
    -RedirectStandardOutput $DaemonLog -RedirectStandardError "$DaemonLog.err" `
    -PassThru -NoNewWindow -Environment $daemonEnv
Write-Host "  Daemon PID: $($daemonProc.Id)"
Write-Host "  Daemon log: $DaemonLog"

Write-Host "  Waiting for daemon to be ready..."
$ready = $false
for ($i = 1; $i -le 20; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v2/settings" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host "  ERROR: Daemon not ready after 20s. Check $DaemonLog and $DaemonLog.err"
    exit 1
}
Write-Host "  Daemon ready"

# --- Start vite from the integrated worktree's web/ ---
Write-Host ""
Write-Host "=== Starting vite ==="
$viteWeb = "$IntegratedWorktree\web"
if (-not (Test-Path "$viteWeb\node_modules")) {
    Write-Host "  Linking node_modules from main worktree..."
    cmd /c mklink /D "$viteWeb\node_modules" "D:\orbital-public\web\node_modules" | Out-Null
}
$viteProc = Start-Process -FilePath "npx" `
    -ArgumentList "vite","--host","127.0.0.1","--port","5173" `
    -WorkingDirectory $viteWeb `
    -RedirectStandardOutput $ViteLog -RedirectStandardError "$ViteLog.err" `
    -PassThru -NoNewWindow -Environment $daemonEnv
Write-Host "  Vite PID: $($viteProc.Id)"
Write-Host "  Vite log: $ViteLog"

Write-Host "  Waiting for vite..."
$viteReady = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:5173/" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) { $viteReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $viteReady) {
    Write-Host "  WARN: Vite not ready after 30s. Daemon is up; UI may still be starting. Check $ViteLog."
} else {
    Write-Host "  Vite ready"
}

Write-Host ""
Write-Host "=== READY FOR MANUAL SMOKE TEST ==="
Write-Host ""
Write-Host "  Daemon : http://127.0.0.1:8000"
Write-Host "  UI     : http://127.0.0.1:5173"
Write-Host "  Branch : test/full-integration @ $($head.Substring(0,7))"
Write-Host ""
Write-Host "Reads RUN-MANUAL-SMOKE.md for the observation checklist."
Write-Host ""
Write-Host "Stop with: .\RUN-MANUAL-SMOKE.ps1 -Stop"

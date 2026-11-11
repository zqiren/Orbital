# Runs INSIDE the guest once (over SSH as tester, elevated), from
# prepare-vm.ps1, before snapshot `clean` is taken. Installs what a user
# would have before trying Orbital: Node 24, the Codex and DeepSeek Harness
# CLIs the way their READMEs say (npm -g into %APPDATA%\npm, as ci.yml's
# fresh-install-windows job does), plus the test's own Playwright Chromium.
# Installs NO Orbital and leaves no worker logins; the last block proves it.
# Mirrors tart/guest-provision.sh. ASCII only (Windows PowerShell 5.1).
param(
  [string]$NodeMsi = 'C:\fi-setup\node.msi',
  [string]$CodexVersion = 'latest',
  [string]$DshVersion = '0.2.2-rc.1',          # npm dist-tags on @deepseek-ai/dsh are not reliable; pin
  [string]$PlaywrightVersion = '1.60.0'
)
$ErrorActionPreference = 'Continue'   # native stderr (npm warnings) must not throw in 5.1
function Fail([string]$m) { Write-Host "PROVISION FAILED: $m"; exit 1 }
function Refresh-Path {
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

# --- Node 24 LTS (official MSI; adds C:\Program Files\nodejs and %APPDATA%\npm to PATH)
if (-not (Test-Path 'C:\Program Files\nodejs\node.exe')) {
  $p = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @('/i', $NodeMsi, '/qn', '/norestart', '/l*v', 'C:\fi-setup\node-msi.log')
  Write-Host "node msi exit $($p.ExitCode)"
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "node install exit $($p.ExitCode)" }
}
Refresh-Path
& node --version; & npm --version
if ($LASTEXITCODE -ne 0) { Fail 'npm not on PATH after the Node install' }

# --- npm global prefix %APPDATA%\npm, and NPM_CONFIG_PREFIX (beats any npmrc), as ci.yml does
$prefix = Join-Path $env:APPDATA 'npm'
New-Item -ItemType Directory -Force -Path $prefix | Out-Null
[Environment]::SetEnvironmentVariable('NPM_CONFIG_PREFIX', $prefix, 'User')
$env:NPM_CONFIG_PREFIX = $prefix
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $prefix)) {
  [Environment]::SetEnvironmentVariable('Path', (($userPath, $prefix) -join ';').TrimStart(';'), 'User')
}
Refresh-Path
& npm config set prefix $prefix
Write-Host "npm global prefix: $(& npm prefix -g)"

# --- worker CLIs. npm 11 holds back install scripts unless allowed; allow dsh's own.
& npm install -g --no-audit --no-fund "@openai/codex@$CodexVersion"
if ($LASTEXITCODE -ne 0) { Fail 'npm i -g @openai/codex' }
& npm install -g --no-audit --no-fund '--allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,protobufjs' "@deepseek-ai/dsh@$DshVersion"
if ($LASTEXITCODE -ne 0) { Fail 'npm i -g @deepseek-ai/dsh' }
if (-not (Test-Path "$prefix\codex.cmd")) { Fail "codex.cmd not in $prefix" }
if (-not (Test-Path "$prefix\dsh.cmd")) { Fail "dsh.cmd not in $prefix" }
& "$prefix\codex.cmd" --version
& "$prefix\dsh.cmd" -V

# --- the test's Playwright Chromium (lands in %LOCALAPPDATA%\ms-playwright), from a scratch folder
$warm = Join-Path $env:TEMP 'fi-warm'
Remove-Item -Recurse -Force $warm -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $warm | Out-Null
Set-Content -Path "$warm\package.json" -Encoding Ascii -Value ('{"name":"fi-warm","private":true,"dependencies":{"playwright":"' + $PlaywrightVersion + '"}}')
Push-Location $warm
& npm install --no-audit --no-fund
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'npm install playwright' }
& npx playwright install chromium
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'playwright install chromium' }
Pop-Location
Remove-Item -Recurse -Force $warm -ErrorAction SilentlyContinue
Get-ChildItem (Join-Path $env:LOCALAPPDATA 'ms-playwright') | ForEach-Object { Write-Host "playwright browser: $($_.Name)" }

# --- tester's desktop: no screen saver
$desk = 'HKCU:\Control Panel\Desktop'
Set-ItemProperty $desk -Name ScreenSaveActive -Value '0'
Set-ItemProperty $desk -Name ScreenSaverIsSecure -Value '0'
Remove-ItemProperty $desk -Name SCRNSAVE.EXE -ErrorAction SilentlyContinue

# --- Windows Update paused as long as Windows allows (the pause cap raised to
# 20 years first). Defender stays on: users have it.
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
New-Item -Force -Path $ux -ErrorAction SilentlyContinue | Out-Null
Set-ItemProperty $ux -Name FlightSettingsMaxPauseDays -Value 7300 -Type DWord
$now = (Get-Date).ToUniversalTime()
$until = $now.AddDays(7300)
$fmt = 'yyyy-MM-ddTHH:mm:ssZ'
foreach ($n in @('PauseFeatureUpdatesStartTime', 'PauseQualityUpdatesStartTime', 'PauseUpdatesStartTime')) {
  Set-ItemProperty $ux -Name $n -Value $now.ToString($fmt) -Type String
}
foreach ($n in @('PauseFeatureUpdatesEndTime', 'PauseQualityUpdatesEndTime', 'PauseUpdatesExpiryTime')) {
  Set-ItemProperty $ux -Name $n -Value $until.ToString($fmt) -Type String
}
try {
  $mp = Get-MpComputerStatus
  Write-Host "Defender: AMServiceEnabled=$($mp.AMServiceEnabled) RealTimeProtectionEnabled=$($mp.RealTimeProtectionEnabled)"
} catch { Write-Host "Defender status unavailable: $_" }

# --- facts for the record
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "guest OS: $($os.Caption) $($os.Version) build $($os.BuildNumber) lang $($os.OSLanguage) locale $($os.Locale) codepage $($os.CodeSet)"
Write-Host "culture: $((Get-Culture).Name) ui: $((Get-UICulture).Name)"

# --- nothing of Orbital, no worker logins. The version probes above may have
# created empty CLI homes; a new user's have none.
Remove-Item -Recurse -Force (Join-Path $HOME '.codex'), (Join-Path $HOME '.dsh') -ErrorAction SilentlyContinue
$bad = @('C:\Orbital', (Join-Path $env:APPDATA 'Orbital'), (Join-Path $HOME 'orbital'), (Join-Path $HOME '.orbital'),
         (Join-Path $HOME '.codex'), (Join-Path $HOME '.dsh')) | Where-Object { Test-Path $_ }
if ($bad) { Fail "guest is not clean: $($bad -join ', ')" }
Write-Host 'guest provision OK'
exit 0

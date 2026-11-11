// Orbital — fresh-install test: what differs per OS (install, launch, quit,
// where the data lives). Everything else in the scenario is OS-agnostic.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';

const HOME = os.homedir();
export const IS_MAC = process.platform === 'darwin';
export const IS_WIN = process.platform === 'win32';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function sh(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', timeout: opts.timeout ?? 120_000, ...opts });
  return { code: r.status, out: (r.stdout || '') + (r.stderr || ''), error: r.error };
}

function mustSh(cmd, args, opts) {
  const r = sh(cmd, args, opts);
  if (r.code !== 0) throw new Error(`${cmd} ${args.join(' ')} exited ${r.code}: ${r.out.trim().slice(-800)}`);
  return r.out;
}

/** Where Orbital keeps things on this OS (mirrors agent_os/desktop/migration.py + main.py). */
export function orbitalPaths() {
  if (IS_MAC) {
    return {
      app: '/Applications/Orbital.app',
      exe: '/Applications/Orbital.app/Contents/MacOS/Orbital',
      dataDir: path.join(HOME, 'Library', 'Application Support', 'Orbital'),
      logDir: path.join(HOME, 'Library', 'Logs', 'Orbital'),
      pidDir: path.join(HOME, 'orbital'),
      extra: [path.join(HOME, '.orbital')],
    };
  }
  if (IS_WIN) {
    const appData = process.env.APPDATA || path.join(HOME, 'AppData', 'Roaming');
    return {
      app: 'C:\\Orbital',
      exe: 'C:\\Orbital\\bin\\Orbital.exe',
      dataDir: path.join(appData, 'Orbital'),
      logDir: path.join(appData, 'Orbital', 'logs'),
      pidDir: path.join(HOME, 'orbital'),
      extra: [path.join(HOME, '.orbital')],
    };
  }
  throw new Error(`unsupported platform ${process.platform}`);
}

export function orbitalRunning() {
  if (IS_MAC) return sh('pgrep', ['-x', 'Orbital']).code === 0;
  const r = sh('tasklist', ['/FI', 'IMAGENAME eq Orbital.exe', '/NH']);
  return /Orbital\.exe/i.test(r.out);
}

/** Step 1: nothing of Orbital on this machine. Returns the checked paths. */
export function assertFresh() {
  const p = orbitalPaths();
  const present = [p.app, p.dataDir, p.logDir, p.pidDir, ...p.extra].filter((x) => fs.existsSync(x));
  const problems = present.map((x) => `exists: ${x}`);
  if (orbitalRunning()) problems.push('an Orbital process is running');
  if (problems.length) throw new Error(`machine is NOT fresh — ${problems.join('; ')}`);
  return [p.app, p.dataDir, p.logDir, p.pidDir, ...p.extra];
}

/** Step 2: install like a user. Returns a short description. */
export function install(installer, evidenceDir) {
  if (!fs.existsSync(installer)) throw new Error(`installer not found: ${installer}`);
  const p = orbitalPaths();
  if (IS_MAC) {
    const mnt = fs.mkdtempSync(path.join(os.tmpdir(), 'orbital-dmg-'));
    mustSh('hdiutil', ['attach', '-nobrowse', '-readonly', '-noautoopen', '-mountpoint', mnt, installer], {
      timeout: 600_000,
    });
    try {
      const appName = fs.readdirSync(mnt).find((n) => n.endsWith('.app'));
      if (!appName) throw new Error(`no .app inside ${installer}`);
      // Finder's drag-to-Applications replaces an existing bundle outright.
      if (fs.existsSync(p.app)) fs.rmSync(p.app, { recursive: true, force: true });
      mustSh('ditto', [path.join(mnt, appName), p.app], { timeout: 600_000 });
    } finally {
      sh('hdiutil', ['detach', mnt, '-force'], { timeout: 120_000 });
    }
    return `ditto'd ${path.basename(installer)} into ${p.app}`;
  }
  const log = path.join(evidenceDir, `inno-${path.basename(installer)}.log`);
  mustSh(installer, ['/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', `/LOG=${log}`], {
    timeout: 1_200_000,
  });
  if (!fs.existsSync(p.exe)) throw new Error(`installer finished but ${p.exe} is missing`);
  return `Inno Setup silent install of ${path.basename(installer)} -> ${p.exe}`;
}

/** The installed build's version string (bundle / uninstall registry). */
export function installedVersion() {
  const p = orbitalPaths();
  if (IS_MAC) {
    const r = sh('plutil', ['-extract', 'CFBundleShortVersionString', 'raw', path.join(p.app, 'Contents', 'Info.plist')]);
    return r.code === 0 ? r.out.trim() : null;
  }
  const r = sh('powershell', [
    '-NoProfile',
    '-Command',
    "(Get-ItemProperty 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*','HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*' -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like 'Orbital*' } | Select-Object -First 1).DisplayVersion",
  ]);
  return r.out.trim() || null;
}

/** Signature / notarization facts for the evidence (macOS); never throws. */
export function signatureFacts() {
  const p = orbitalPaths();
  if (IS_MAC) {
    const verify = sh('codesign', ['--verify', '--deep', '--strict', '--verbose=2', p.app], { timeout: 300_000 });
    const details = sh('codesign', ['-dv', '--verbose=2', p.app]);
    const spctl = sh('spctl', ['--assess', '--type', 'execute', '--verbose=4', p.app], { timeout: 300_000 });
    const authority = (details.out.match(/^Authority=.*$/gm) || []).join(' | ');
    const runtime = /flags=.*runtime/.test(details.out);
    return {
      codesign_verify_ok: verify.code === 0,
      authority,
      hardened_runtime: runtime,
      spctl_ok: spctl.code === 0,
      spctl: spctl.out.trim().split('\n').slice(0, 3).join(' | '),
    };
  }
  const r = sh('powershell', [
    '-NoProfile',
    '-Command',
    `(Get-AuthenticodeSignature '${p.exe}').Status`,
  ]);
  return { authenticode: r.out.trim() };
}

/** Step 3: launch the installed app the way a user does (not --smoke-test). */
export function launch() {
  const p = orbitalPaths();
  if (IS_MAC) {
    // A LaunchServices launch does not inherit this shell's env. Hand the
    // telemetry opt-out to the GUI session so a test install never reports
    // to production telemetry (CI sets it workflow-wide; the Tart rig too).
    let envNote = '';
    if (process.env.AGENT_OS_TELEMETRY_DISABLED) {
      sh('launchctl', ['setenv', 'AGENT_OS_TELEMETRY_DISABLED', process.env.AGENT_OS_TELEMETRY_DISABLED]);
      envNote = ' (AGENT_OS_TELEMETRY_DISABLED set in the GUI session via launchctl setenv)';
    }
    mustSh('open', [p.app]);
    return `open ${p.app}${envNote}`;
  }
  const child = spawn(p.exe, [], { detached: true, stdio: 'ignore', cwd: path.dirname(p.exe) });
  child.unref();
  return `started ${p.exe}`;
}

/** The daemon's real port: the PID-file sidecar records it (spec 134 A4). */
export function readDaemonMeta() {
  const meta = path.join(orbitalPaths().pidDir, 'daemon.pid.meta');
  try {
    return JSON.parse(fs.readFileSync(meta, 'utf8'));
  } catch {
    return null;
  }
}

export async function waitForDaemon({ timeoutMs = 240_000, note = () => {} } = {}) {
  const t0 = Date.now();
  let lastErr = '';
  while (Date.now() - t0 < timeoutMs) {
    const meta = readDaemonMeta();
    const port = meta && Number(meta.port);
    if (port) {
      try {
        const r = await fetch(`http://127.0.0.1:${port}/api/v2/settings`, { signal: AbortSignal.timeout(5000) });
        if (r.ok) {
          note(`daemon answers on port ${port} (pid ${meta.pid}) after ${Math.round((Date.now() - t0) / 1000)}s`);
          return { port, meta };
        }
        lastErr = `HTTP ${r.status}`;
      } catch (e) {
        lastErr = e.message;
      }
    }
    await sleep(2000);
  }
  throw new Error(`daemon never answered within ${timeoutMs / 1000}s (last: ${lastErr || 'no daemon.pid.meta'})`);
}

/** Quit like a user (app menu Quit / tray Quit), escalating only if it hangs. */
export async function quit({ note = () => {} } = {}) {
  if (IS_MAC) {
    sh('osascript', ['-e', 'tell application "Orbital" to quit'], { timeout: 60_000 });
  } else {
    sh('taskkill', ['/IM', 'Orbital.exe', '/T']);
  }
  for (let i = 0; i < 30 && orbitalRunning(); i++) await sleep(1000);
  if (orbitalRunning()) {
    note('graceful quit did not finish in 30s; ending the process');
    if (IS_MAC) sh('pkill', ['-x', 'Orbital']);
    else sh('taskkill', ['/IM', 'Orbital.exe', '/T', '/F']);
    for (let i = 0; i < 15 && orbitalRunning(); i++) await sleep(1000);
  }
  if (orbitalRunning()) throw new Error('Orbital is still running after quit');
}

/** Full-screen capture: proves (or not) that the native window opened. */
export function desktopScreenshot(file) {
  if (IS_MAC) return sh('screencapture', ['-x', file]).code === 0;
  const ps = `Add-Type -AssemblyName System.Windows.Forms,System.Drawing;
$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;
$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;
$g=[System.Drawing.Graphics]::FromImage($bmp);$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);
$bmp.Save('${file}',[System.Drawing.Imaging.ImageFormat]::Png)`;
  return sh('powershell', ['-NoProfile', '-Command', ps]).code === 0;
}

/** Does the OS report an Orbital window? Best effort; null when unknown. */
export function nativeWindowFacts() {
  if (IS_MAC) {
    const r = sh('osascript', [
      '-l',
      'JavaScript',
      '-e',
      `ObjC.import('CoreGraphics');
       const list = ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly, 0))) || [];
       JSON.stringify(list.filter(w => w.kCGWindowOwnerName === 'Orbital').map(w => ({name: w.kCGWindowName || '', layer: w.kCGWindowLayer, bounds: w.kCGWindowBounds})));`,
    ]);
    try {
      return { windows: JSON.parse(r.out.trim() || '[]') };
    } catch {
      return { windows: null, raw: r.out.trim().slice(0, 300) };
    }
  }
  const r = sh('powershell', [
    '-NoProfile',
    '-Command',
    "Get-Process Orbital -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -ExpandProperty MainWindowTitle",
  ]);
  return { windows: r.out.trim() ? r.out.trim().split(/\r?\n/) : [] };
}

/** Copy daemon/app logs into evidence (scrubbed later with everything else). */
export function collectLogs(evidenceDir) {
  const p = orbitalPaths();
  const out = path.join(evidenceDir, 'logs');
  fs.mkdirSync(out, { recursive: true });
  const copied = [];
  for (const [label, dir] of [
    ['data-logs', path.join(p.dataDir, 'logs')],
    ['app-logs', p.logDir],
  ]) {
    if (!fs.existsSync(dir)) continue;
    const dest = path.join(out, label);
    fs.cpSync(dir, dest, { recursive: true });
    copied.push(label);
  }
  for (const name of ['daemon.pid.meta']) {
    const f = path.join(p.pidDir, name);
    if (fs.existsSync(f)) fs.copyFileSync(f, path.join(out, name));
  }
  return copied;
}

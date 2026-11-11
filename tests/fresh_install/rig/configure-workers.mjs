// Orbital — fresh-install test rig: set up the worker CLIs the way a user
// with an OpenCode Go key would, BEFORE Orbital is installed. Orbital never
// patches a worker's own configuration (product rule), so this is the
// user's side of the fence, done by hand in real life:
//
//   - Codex: a custom model provider in ~/.codex/config.toml pointing at the
//     Go Responses endpoint (with the x-opencode-session header Go requires),
//     and `codex login --with-api-key` so a GUI-launched app needs no env.
//   - dsh: nothing in ~/.dsh. Its key and base URL go in through Orbital's
//     Settings UI later. dsh cannot send custom headers on its
//     DEEPSEEK_BASE_URL route, so the rig runs go-header-proxy.mjs on
//     127.0.0.1 and the base URL points there.
//
// The key comes from the environment and is never printed or logged.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const IS_WIN = process.platform === 'win32';

export const GO_BASE = 'https://opencode.ai/zen/go/v1';
export const PROXY_PORT = 18787;

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', shell: IS_WIN, timeout: 120_000, ...opts });
  return { code: r.status, out: ((r.stdout || '') + (r.stderr || '')).trim() };
}

export function cliVersions() {
  const v = {};
  for (const [name, args] of [
    ['node', ['--version']],
    ['codex', ['--version']],
    ['dsh', ['-V']],
    ['mcode', ['--version']],
  ]) {
    const r = run(name, args);
    v[name] = r.code === 0 ? r.out.split('\n').pop() : null;
  }
  return v;
}

export function configureCodex({ key, model = 'gpt-6-luna', note = () => {} }) {
  const codexHome = process.env.CODEX_HOME || path.join(os.homedir(), '.codex');
  fs.mkdirSync(codexHome, { recursive: true });
  const cfg = path.join(codexHome, 'config.toml');
  if (fs.existsSync(cfg) && !fs.readFileSync(cfg, 'utf8').includes('# orbital-fresh-install rig')) {
    throw new Error(`${cfg} already exists and is not the rig's; refusing to overwrite a real config`);
  }
  fs.writeFileSync(
    cfg,
    [
      '# orbital-fresh-install rig: Codex on OpenCode Go, as a user would set it up.',
      `model = "${model}"`,
      'model_provider = "opencode-go"',
      '',
      '[model_providers.opencode-go]',
      'name = "OpenCode Go"',
      `base_url = "${GO_BASE}"`,
      'wire_api = "responses"',
      'requires_openai_auth = true',
      `http_headers = { "x-opencode-session" = "orbital-fresh-install-codex" }`,
      '',
    ].join('\n'),
  );
  const r = spawnSync('codex', ['login', '--with-api-key'], {
    input: key,
    encoding: 'utf8',
    shell: IS_WIN,
    timeout: 120_000,
  });
  if (r.status !== 0) throw new Error(`codex login --with-api-key exited ${r.status}`);
  const status = run('codex', ['login', 'status']);
  note(`codex: config.toml -> Go Responses (${model}); login status exit ${status.code}`);
  return cfg;
}

/**
 * dsh's default model, in dsh's own profile file (what DSH Desktop writes
 * when the user picks a model there; Orbital mirrors its
 * `agent-default-model` row when no model is chosen in Orbital). dsh's
 * stock default `deepseek-flash` is not an OpenCode Go model.
 */
export function configureDsh({ model = 'deepseek-official/deepseek-v4-pro', note = () => {} }) {
  const dshHome = process.env.DSH_HOME || path.join(os.homedir(), '.dsh');
  const file = path.join(dshHome, 'profiles', 'desktop', 'cordis.patch.yml');
  if (fs.existsSync(file) && !fs.readFileSync(file, 'utf8').includes('# orbital-fresh-install rig')) {
    throw new Error(`${file} already exists and is not the rig's; refusing to overwrite a real config`);
  }
  const [provider, id] = model.split('/', 2);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(
    file,
    [
      '# orbital-fresh-install rig: the default model a user would pick in DSH Desktop.',
      '- id: agent-default-model',
      '  config:',
      `    provider: ${provider}`,
      `    model: ${id}`,
      '',
    ].join('\n'),
  );
  note(`dsh: default model ${model} in ${file}`);
  return file;
}

/** Start the header-injecting forwarder for dsh; returns its base URL. */
export async function startGoProxy({ logFile, note = () => {} }) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  const out = fs.openSync(logFile, 'a');
  const child = spawn(
    process.execPath,
    [path.join(HERE, 'go-header-proxy.mjs'), '--port', String(PROXY_PORT)],
    { detached: true, stdio: ['ignore', out, out], windowsHide: true },
  );
  child.unref();
  const url = `http://127.0.0.1:${PROXY_PORT}`;
  for (let i = 0; i < 30; i++) {
    try {
      // Unauthenticated probe: any HTTP answer means the forwarder is up.
      await fetch(url + '/v1/models', { signal: AbortSignal.timeout(5000) });
      note(`go-header-proxy up at ${url} (pid ${child.pid})`);
      return { url, pid: child.pid };
    } catch {
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  throw new Error('go-header-proxy did not start');
}

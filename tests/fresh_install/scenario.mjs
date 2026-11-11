#!/usr/bin/env node
// Orbital — fresh-install scenario.
//
// Replays a brand-new user's first minutes on a machine that has never seen
// Orbital, against an INSTALLER (the CI-built one — never a local build).
// One script for macOS and Windows; see README.md.
//
//   node scenario.mjs --installer <dmg|exe> --evidence <dir>
//                     [--mode fresh|upgrade] [--old-installer <dmg|exe>]
//                     [--require-key] [--codex-model gpt-6-luna]
//                     [--manager-model deepseek-v4-pro] [--acp dsh]
//                     [--workspace-root <dir>]
//
// The OpenCode Go test key comes from ORBITAL_TEST_OPENCODE_KEY (or
// OPENCODE_GO_TEST_KEY) and is never printed. Exit 1 when any step failed.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { webkit, chromium } from 'playwright';

import { Run, NotVerified } from './lib/steps.mjs';
import * as plat from './lib/platform.mjs';
import { Api, waitFor, text, rowsAfterUser } from './lib/orbital.mjs';
import * as ui from './lib/ui.mjs';
import { scrubAndVerify, secretsFromEnv } from './lib/scrub.mjs';
import { cliVersions, configureCodex, configureDsh, startGoProxy } from './rig/configure-workers.mjs';

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : dflt;
};
const flag = (name) => argv.includes(`--${name}`);

const opts = {
  mode: arg('mode', 'fresh'),
  installer: arg('installer'),
  oldInstaller: arg('old-installer'),
  evidence: path.resolve(arg('evidence', path.join(process.cwd(), 'evidence'))),
  requireKey: flag('require-key'),
  codexModel: arg('codex-model', 'gpt-6-luna'),
  managerModel: arg('manager-model', 'deepseek-v4-pro'),
  // dsh's model: set by the rig in dsh's own profile (see README); --dsh-model sets it in Orbital instead.
  dshModel: arg('dsh-model', ''),
  dshProfileModel: arg('dsh-profile-model', 'deepseek-official/deepseek-v4-pro'),
  acp: arg('acp', 'dsh'),
  workspaceRoot: path.resolve(arg('workspace-root', path.join(os.homedir(), 'orbital-fresh-ws'))),
  browser: arg('browser', plat.IS_MAC ? 'webkit' : 'chromium'),
};
if (!opts.installer || (opts.mode === 'upgrade' && !opts.oldInstaller)) {
  console.error('usage: node scenario.mjs --installer <file> [--mode upgrade --old-installer <file>] --evidence <dir>');
  process.exit(2);
}

const KEY = secretsFromEnv()[0] || '';
const run = new Run(opts.evidence, {
  mode: opts.mode,
  os: `${process.platform} ${os.release()} ${os.arch()}`,
  installer: path.basename(opts.installer),
  old_installer: opts.oldInstaller ? path.basename(opts.oldInstaller) : undefined,
  key_present: Boolean(KEY),
  browser: opts.browser,
});

const LONG_TURN = 600_000; // a real LLM turn plus worker spawns; generous on purpose

// ------------------------------------------------------------------- state
const S = { port: null, api: null, browser: null, context: null, page: null, projects: [] };

function needKey() {
  if (!KEY) {
    if (opts.requireKey) throw new Error('ORBITAL_TEST_OPENCODE_KEY is not set (--require-key)');
    throw new NotVerified('no test key in env (ORBITAL_TEST_OPENCODE_KEY)');
  }
}

async function openBrowser(note) {
  const engine = opts.browser === 'webkit' ? webkit : chromium;
  S.browser = await engine.launch({ headless: true });
  S.context = await S.browser.newContext({ viewport: { width: 1440, height: 900 }, locale: 'en-US' });
  S.page = await S.context.newPage();
  run.page = S.page;
  S.page.on('pageerror', (e) => fs.appendFileSync(path.join(run.dir, 'page-errors.log'), `${e.message}\n`));
  S.page.on('console', (m) => {
    if (m.type() === 'error') fs.appendFileSync(path.join(run.dir, 'page-console-errors.log'), `${m.text()}\n`);
  });
  await ui.openApp(S.page, S.port);
  note(`driving the installed app's UI with Playwright ${opts.browser} at http://127.0.0.1:${S.port}/`);
}

async function closeBrowser() {
  try {
    await S.browser?.close();
  } catch {}
  S.browser = S.context = S.page = null;
  run.page = null;
}

function workspace(name) {
  const dir = path.join(opts.workspaceRoot, name.toLowerCase().replace(/[^a-z0-9]+/g, '-'));
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

async function projectId(name) {
  const p = await waitFor(() => S.api.projectByName(name), { timeoutMs: 60_000, label: `project ${name} in /projects` });
  if (!S.projects.find((x) => x.project_id === p.project_id)) S.projects.push(p);
  return p.project_id;
}

const reWorkerFailed = (slug) =>
  new RegExp(`\\[Sub-agent\\] ${slug} (failed|stopped with error|requires input)`);

/** A worker's answer containing PONG after the user row `marker`. */
function findWorkerPong(rows, marker, slug) {
  const after = rowsAfterUser(rows, marker);
  if (!after) return null;
  const reply = after.find(
    (r) =>
      r.role === 'assistant' &&
      r.source !== 'management' &&
      (r.sub_agent_handle === slug || r.source === slug) &&
      !r.sub_agent_in_flight &&
      /\bPONG\b/i.test(text(r)),
  );
  if (reply) return { row: reply, via: `assistant row source=${reply.source} handle=${reply.sub_agent_handle || ''}` };
  const marker2 = after.find(
    (r) => r.role === 'system' && new RegExp(`\\[Sub-agent\\] ${slug} completed\\. Summary: [^]*PONG`, 'i').test(text(r)),
  );
  if (marker2) return { row: marker2, via: 'completed marker' };
  const failed = after.find((r) => r.role === 'system' && reWorkerFailed(slug).test(text(r)));
  if (failed) return { fail: text(failed).slice(0, 400) };
  return null;
}

async function latestRows(pid) {
  const s = await S.api.latestSession(pid);
  if (!s) return { session: null, rows: [] };
  return { session: s, rows: await S.api.chat(pid, s.session_id) };
}

/** Manager's final text reply after `marker`, once the session is no longer running. */
async function waitManagerReply(pid, marker, { timeoutMs = LONG_TURN, accept = () => true } = {}) {
  return waitFor(
    async () => {
      const { session, rows } = await latestRows(pid);
      if (!session) return null;
      const after = rowsAfterUser(rows, marker);
      if (!after) return null;
      const errs = after.filter((r) => r.role === 'system' && /error|failed/i.test(text(r)) && r.source !== 'daemon');
      const replies = after.filter(
        // Bookkeeping turns (memory record, asks) carry a _kind; the user's answer does not.
        (r) => r.role === 'assistant' && r.source === 'management' && !r._kind && text(r).trim() && !(r.tool_calls || []).length,
      );
      const busy = ['running', 'pending_approval', 'queued'].includes(session.status);
      const last = replies[replies.length - 1];
      if (last && !busy && accept(after, last)) return { session, rows, after, reply: last };
      if (!busy && !last && errs.length) return { fail: text(errs[errs.length - 1]).slice(0, 400) };
      return null;
    },
    { timeoutMs, intervalMs: 4000, label: 'manager reply' },
  );
}

// ------------------------------------------------------------------- steps
async function stepPrecondition() {
  return run.step('1-precondition', 'Machine is fresh (no Orbital, no Orbital data)', async (note) => {
    const checked = plat.assertFresh();
    note(`none present: ${checked.join(', ')}`);
  });
}

async function stepRig() {
  return run.step('0-rig', "Worker CLIs set up the user's way (Codex + dsh on OpenCode Go)", async (note) => {
    const v = cliVersions();
    note(`versions: ${Object.entries(v).map(([k, x]) => `${k}=${x ?? 'missing'}`).join(', ')}`);
    if (!v.codex) throw new Error('codex CLI not installed');
    if (opts.acp === 'dsh' && !v.dsh) throw new Error('dsh CLI not installed');
    needKey();
    configureCodex({ key: KEY, model: opts.codexModel, note });
    if (opts.acp === 'dsh' && opts.dshProfileModel) configureDsh({ model: opts.dshProfileModel, note });
    S.proxy = await startGoProxy({ logFile: path.join(run.dir, 'rig', 'go-header-proxy.log'), note });
    return v;
  });
}

async function stepInstall(installer, id = '2-install') {
  return run.step(id, `Install ${path.basename(installer)} like a user`, async (note) => {
    note(plat.install(installer, run.dir));
    const version = plat.installedVersion();
    note(`installed version: ${version}`);
    const sig = plat.signatureFacts();
    note(`signature: ${JSON.stringify(sig)}`);
    return { version, signature: sig };
  });
}

async function stepLaunch(id = '3-launch', needs = ['2-install']) {
  return run.step(
    id,
    'Launch the installed app normally; daemon serves the SPA',
    async (note) => {
      note(plat.launch());
      const { port } = await plat.waitForDaemon({ note });
      S.port = port;
      S.api = new Api(port);
      const html = await (await fetch(`http://127.0.0.1:${port}/`)).text();
      if (!/<div id="root"|<script/i.test(html)) throw new Error('daemon root did not serve the SPA');
      // Native window: give it a moment, then record what the OS sees.
      await new Promise((r) => setTimeout(r, 8000));
      const win = plat.nativeWindowFacts();
      note(`native window (OS view): ${JSON.stringify(win)}`);
      const shot = path.join(run.dir, 'screenshots', `${id}-desktop.png`);
      note(plat.desktopScreenshot(shot) ? `desktop screenshot: screenshots/${id}-desktop.png` : 'desktop screenshot failed');
      await openBrowser(note);
      return { port, native_window: win };
    },
    { needs },
  );
}

// uiModel: only for a build older than v0.22.0 (spec 172 B): those override
// ~/.codex/config.toml's model, so a Go user had to choose it in Orbital.
async function stepKeyless({ id = '4-keyless', projectName = 'Fresh Keyless', prompt, needs = ['3-launch'], uiModel = '' } = {}) {
  return run.step(
    id,
    'Keyless: skip the key in onboarding, pin Codex, get a real reply',
    async (note) => {
      needKey(); // Codex itself runs on the Go key (its own login)
      const page = S.page;
      await ui.onboardingSkipKey(page, note);
      await run.shot(`${id}-after-wizard`);
      const st = await S.api.settings();
      note(`settings after wizard: api_key_set=${st?.llm?.api_key_set} api_key_skipped=${st?.setup?.api_key_skipped}`);
      await ui.createProject(page, { name: projectName, folder: workspace(projectName) }, note);
      const pid = await projectId(projectName);
      await ui.walkTourIfShown(page, note);
      if (uiModel) {
        await ui.openGlobalSettings(page);
        await ui.setSubAgentModel(page, 'codex', uiModel, note);
        await run.shot(`${id}-codex-model`);
        await ui.leaveGlobalSettings(page);
        await ui.openProject(page, projectName);
      }
      await ui.choosePin(page, 'Codex', note);
      const ph = await ui.composer(page).getAttribute('placeholder');
      note(`composer placeholder: ${ph}`);
      if (!/codex/i.test(ph || '')) throw new Error(`composer is not talking to Codex (placeholder "${ph}")`);
      const msg = prompt || 'Reply with the single word PONG and nothing else.';
      await ui.send(page, msg);
      const found = await waitFor(
        async () => {
          const { rows } = await latestRows(pid);
          return findWorkerPong(rows, msg, 'codex');
        },
        { timeoutMs: LONG_TURN, intervalMs: 4000, label: 'Codex reply' },
      );
      note(`Codex replied (${found.via}): ${text(found.row).slice(0, 120)}`);
      await waitFor(async () => (await ui.workerBubbleCount(page, 'codex')) > 0, {
        timeoutMs: 60_000,
        intervalMs: 1000,
        label: 'Codex bubble in the chat',
      });
      note('Codex reply bubble rendered in the chat');
      return { project_id: pid };
    },
    { needs },
  );
}

async function stepAddCard() {
  return run.step(
    '5a-card',
    'Add an OpenCode Go provider card in Settings (mandatory connection test)',
    async (note) => {
      needKey();
      const page = S.page;
      await ui.openGlobalSettings(page);
      await ui.addProviderCard(page, { providerId: 'opencode-go', key: KEY, model: opts.managerModel }, note);
      await run.shot('5a-card-saved');
      const cards = await S.api.cards();
      const card = (cards.credential_cards || []).find((c) => c.provider === 'opencode-go');
      if (!card) throw new Error('no opencode-go card in /settings/cards');
      note(`card: model=${card.model} key_set=${card.key_set} default=${card.id === cards.default_card_id}`);
      if (!card.key_set) throw new Error('card saved without a key');
      if (card.id !== cards.default_card_id) throw new Error('the first card did not become the default');
      const st = await S.api.settings();
      if (!st?.llm?.api_key_set) throw new Error('settings still report no API key');
    },
    { needs: ['4-keyless'] },
  );
}

async function stepWorkerSetup() {
  return run.step(
    '5b-workers',
    `Set ${opts.acp} credentials + model in Settings; workers ready`,
    async (note) => {
      needKey();
      const page = S.page;
      if (opts.acp !== 'dsh') throw new NotVerified(`ACP agent ${opts.acp}: no scripted credential path (mcode needs an interactive \`mcode login\`)`);
      await ui.setSubAgentCredential(page, 'dsh', 'DEEPSEEK_API_KEY', KEY, note);
      await ui.setSubAgentCredential(page, 'dsh', 'DEEPSEEK_BASE_URL', S.proxy.url, note);
      if (opts.dshModel) await ui.setSubAgentModel(page, 'dsh', opts.dshModel, note);
      await run.shot('5b-dsh-configured');
      const list = await waitFor(
        async () => {
          const l = await S.api.subAgents();
          const by = Object.fromEntries(l.map((e) => [e.slug, e]));
          return by.codex?.ready && by.dsh?.ready ? l : null;
        },
        { timeoutMs: 180_000, intervalMs: 5000, label: 'codex and dsh ready in /settings/sub-agents' },
      ).catch(async (e) => {
        const l = await S.api.subAgents();
        const pick = l
          .filter((x) => ['codex', 'dsh'].includes(x.slug))
          .map((x) => `${x.slug}: installed=${x.installed} ready=${x.ready} missing=${JSON.stringify(x.missing_credentials)} handshake=${JSON.stringify(x.handshake)}`);
        throw new Error(`${e.message}; ${pick.join('; ')}`);
      });
      for (const e of list.filter((x) => ['codex', 'dsh'].includes(x.slug))) {
        note(`${e.slug}: ready=${e.ready} version=${e.version} source=${JSON.stringify(e.source ?? null)} path=${e.binary_path}`);
      }
      await ui.leaveGlobalSettings(page);
    },
    { needs: ['5a-card'] },
  );
}

async function stepManagerReply() {
  return run.step(
    '5c-manager',
    'Key path: new project, message the Orbital manager, get a reply',
    async (note) => {
      needKey();
      const page = S.page;
      const name = 'Fresh Manager';
      await ui.createProject(page, { name, folder: workspace(name) }, note);
      const pid = await projectId(name);
      S.managerPid = pid;
      await ui.walkTourIfShown(page, note);
      const ph = await ui.composer(page).getAttribute('placeholder');
      note(`composer placeholder: ${ph}`);
      const msg = 'Hello! In one short sentence, tell me what you can help me with in this project.';
      await ui.send(page, msg);
      const r = await waitManagerReply(pid, msg);
      note(`manager replied: ${text(r.reply).slice(0, 160)}`);
      const snippet = text(r.reply).replace(/[*_#`>\[\]]/g, '').trim().split(/\s+/).slice(0, 5).join(' ');
      await page.getByText(snippet).first().waitFor({ timeout: 60_000 });
      note('manager reply rendered in the chat');
    },
    { needs: ['5a-card'] },
  );
}

async function stepDispatch() {
  return run.step(
    '5d-dispatch',
    `Manager dispatches "PONG" to codex and ${opts.acp}; both results come back`,
    async (note) => {
      needKey();
      const page = S.page;
      const pid = S.managerPid;
      const msg =
        `Please test your sub-agents. Dispatch the task "Reply with the single word PONG" to the codex sub-agent, ` +
        `and dispatch the same task separately to the ${opts.acp} sub-agent. Wait for both results, ` +
        `then tell me exactly what each of them answered.`;
      await ui.send(page, msg);
      const results = {};
      await waitFor(
        async () => {
          const { rows } = await latestRows(pid);
          for (const slug of ['codex', opts.acp]) {
            if (results[slug]) continue;
            const f = findWorkerPong(rows, msg, slug);
            if (f?.fail) return { fail: `${slug}: ${f.fail}` };
            if (f) {
              results[slug] = f;
              note(`${slug} result (${f.via}): ${text(f.row).slice(0, 120)}`);
            }
          }
          return results.codex && results[opts.acp];
        },
        { timeoutMs: LONG_TURN * 1.5, intervalMs: 5000, label: `codex + ${opts.acp} results` },
      );
      // The manager wakes on the results and reports; wait for that report
      // (a reply placed after both completion markers) so the next step
      // starts on an idle session.
      const doneIdx = (after, slug) =>
        after.findIndex((r) => r.role === 'system' && text(r).startsWith(`[Sub-agent] ${slug} completed`));
      const fin = await waitManagerReply(pid, msg, {
        timeoutMs: LONG_TURN,
        accept: (after, last) => {
          const i = after.indexOf(last);
          return ['codex', opts.acp].every((slug) => doneIdx(after, slug) < i);
        },
      }).catch((e) => {
        note(`no manager report after both results: ${e.message}`);
        return null;
      });
      if (fin) note(`manager report: ${text(fin.reply).slice(0, 200)}`);
      const { rows: all } = await latestRows(pid);
      const cancelled = (rowsAfterUser(all, msg) || []).filter((r) => r.role === 'tool' && /^CANCELLED/.test(text(r)));
      if (cancelled.length) note(`observation: ${cancelled.length} dispatch tool call(s) came back "CANCELLED: not executed" (parallel dispatch in one turn); the manager re-sent`);
      for (const slug of ['codex', opts.acp]) {
        const row = page.locator(`[data-testid="sub-agent-activity"][data-action="completed"]`).filter({ hasText: slug });
        const bubbles = await ui.workerBubbleCount(page, slug);
        note(`UI: ${slug} completed rows=${await row.count()} bubbles=${bubbles}`);
      }
    },
    { needs: ['5b-workers', '5c-manager'] },
  );
}

async function stepBrowser() {
  return run.step(
    '6-browser',
    "Manager's browser tool opens a page (hardened-runtime JIT class)",
    async (note) => {
      needKey();
      const page = S.page;
      const pid = S.managerPid;
      const msg =
        'Use your browser tool to open https://example.com and tell me the exact title shown on that page. ' +
        'Do not answer from memory.';
      await ui.send(page, msg);
      const r = await waitManagerReply(pid, msg, {
        accept: (after) => after.some((x) => (x.tool_calls || []).some((tc) => (tc.function?.name || tc.name) === 'browser')),
      });
      const calls = r.after.flatMap((x) => (x.tool_calls || []).map((tc) => tc.function?.name || tc.name));
      const toolRows = r.after.filter((x) => x.role === 'tool');
      const sawPage = toolRows.some((x) => /Example Domain/i.test(text(x)));
      const browserErr = toolRows.find((x) => /(Target page, context or browser has been closed|SIGTRAP|crash|failed to launch|Browser closed)/i.test(text(x)));
      note(`tool calls: ${calls.join(', ')}; a browser tool result contains "Example Domain": ${sawPage}`);
      note(`reply: ${text(r.reply).slice(0, 160)}`);
      if (browserErr) throw new Error(`browser tool error: ${text(browserErr).slice(0, 300)}`);
      if (!sawPage) throw new Error('no browser tool result showed the page ("Example Domain")');
      if (!/Example Domain/i.test(text(r.reply))) throw new Error('reply does not name the page title');
    },
    { needs: ['5c-manager'] },
  );
}

// -------------------------------------------------------------- upgrade
async function stepUpgrade() {
  const U = {};
  await stepInstall(opts.oldInstaller, '7a-install-old');
  await stepLaunch('7b-launch-old', ['7a-install-old']);
  const prompt = 'Reply with the single word PONG and nothing else.';
  await stepKeyless({ id: '7c-old-history', projectName: 'Upgrade Check', prompt, needs: ['7b-launch-old'], uiModel: opts.codexModel });
  await run.step(
    '7d-record',
    'Record the old build\'s project + history, then quit',
    async (note) => {
      const pid = await projectId('Upgrade Check');
      const { session, rows } = await latestRows(pid);
      Object.assign(U, { pid, sid: session.session_id, rowCount: rows.length, sessionName: session.name });
      note(`project ${pid}, session ${session.session_id} (${session.name}), ${rows.length} rows`);
      fs.writeFileSync(path.join(run.dir, 'upgrade-before.json'), JSON.stringify({ session, rows }, null, 2));
      await closeBrowser();
      await plat.quit({ note });
      note('old build quit');
    },
    { needs: ['7c-old-history'] },
  );
  await run.step(
    '7e-install-new',
    `Install ${path.basename(opts.installer)} over it`,
    async (note) => {
      note(plat.install(opts.installer, run.dir));
      note(`installed version now: ${plat.installedVersion()}`);
    },
    { needs: ['7d-record'] },
  );
  await stepLaunch('7f-relaunch', ['7e-install-new']);
  await run.step(
    '7g-intact',
    'After the upgrade the project and its history open intact',
    async (note) => {
      const page = S.page;
      await page.waitForTimeout(3000);
      if (await page.getByTestId('wizard-steps').isVisible().catch(() => false)) {
        throw new Error('the setup wizard came back after the upgrade');
      }
      const p = await S.api.projectByName('Upgrade Check');
      if (!p || p.project_id !== U.pid) throw new Error('project missing or re-created after the upgrade');
      const rows = await S.api.chat(U.pid, U.sid);
      note(`session ${U.sid}: ${rows.length} rows (before: ${U.rowCount})`);
      if (rows.length < U.rowCount) throw new Error('history lost rows');
      if (!findWorkerPong(rows, prompt, 'codex')?.row) throw new Error("the old Codex reply is gone from the session's history");
      await ui.openProject(page, 'Upgrade Check');
      await page.getByTestId('user-message').filter({ hasText: 'PONG' }).first().waitFor({ timeout: 60_000 });
      await waitFor(async () => (await ui.workerBubbleCount(page, 'codex')) > 0, {
        timeoutMs: 60_000,
        intervalMs: 1000,
        label: 'old Codex bubble rendered',
      });
      note('project opens on its session; the old user message and Codex reply render');
    },
    { needs: ['7f-relaunch'] },
  );
}

// ------------------------------------------------------------- evidence
async function stepEvidence() {
  return run.step('8-evidence', 'Collect logs + session history; quit the app', async (note) => {
    if (S.api) {
      for (const p of S.projects) {
        const dir = path.join(run.dir, 'sessions', p.name.replace(/[^a-z0-9]+/gi, '-'));
        fs.mkdirSync(dir, { recursive: true });
        try {
          for (const s of await S.api.sessions(p.project_id)) {
            const rows = await S.api.chat(p.project_id, s.session_id);
            fs.writeFileSync(path.join(dir, `${s.session_id}.chat.json`), JSON.stringify(rows, null, 2));
          }
        } catch (e) {
          note(`chat dump for ${p.name} failed: ${e.message}`);
        }
        const jsonlDir = path.join(p.workspace || '', 'orbital', 'sessions');
        if (p.workspace && fs.existsSync(jsonlDir)) {
          for (const f of fs.readdirSync(jsonlDir).filter((f) => f.endsWith('.jsonl'))) {
            fs.copyFileSync(path.join(jsonlDir, f), path.join(dir, f));
          }
        }
      }
      try {
        fs.writeFileSync(path.join(run.dir, 'sub-agents.json'), JSON.stringify(await S.api.subAgents(), null, 2));
      } catch {}
    }
    await closeBrowser();
    if (plat.orbitalRunning()) await plat.quit({ note });
    note(`logs copied: ${plat.collectLogs(run.dir).join(', ') || 'none'}`);
  });
}

// ------------------------------------------------------------------ main
async function main() {
  run.log(`fresh-install scenario: ${JSON.stringify({ ...opts, key: KEY ? 'set' : 'unset' })}`);
  try {
    await stepPrecondition();
    if (run.status('1-precondition') !== 'passed') return;
    await stepRig();
    if (opts.mode === 'upgrade') {
      await stepUpgrade();
    } else {
      await stepInstall(opts.installer);
      await stepLaunch();
      await stepKeyless();
      await stepAddCard();
      await stepWorkerSetup();
      await stepManagerReply();
      await stepDispatch();
      await stepBrowser();
    }
  } catch (e) {
    run.log(`scenario aborted: ${e.stack || e}`);
  } finally {
    try {
      await stepEvidence();
    } catch (e) {
      run.log(`evidence step crashed: ${e.message}`);
    }
    if (S.proxy?.pid) {
      try {
        process.kill(S.proxy.pid);
      } catch {}
    }
    run.meta.finished_at = new Date().toISOString();
    run.write();
    const scrub = scrubAndVerify(run.dir, { requireSecret: opts.requireKey });
    run.log(`scrub: ok=${scrub.ok} rewritten=${(scrub.rewritten || []).length} deleted=${(scrub.deleted || []).length}`);
    const summary = run.steps.map((s) => `${s.status.padEnd(12)} ${s.id}`).join('\n');
    console.log(`\n${summary}\n`);
    process.exit(run.failed || !scrub.ok ? 1 : 0);
  }
}

main();

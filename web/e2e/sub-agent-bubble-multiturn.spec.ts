// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Playwright spec for TASK-subagent-last-message-display: a sub-agent's full
// per-turn final message renders as a real chat bubble in the management
// transcript, ONE bubble per dispatch, each carrying its OWN turn's text (no
// cross-turn aliasing), and degrading honestly for an errored turn.
//
// Like e2e/chat-reasoning-capsule.spec.ts, this seeds on-disk fixtures and
// serves them through the REAL daemon + REAL frontend (a live model turn is a
// sandbox hazard and is deliberately avoided — see that spec's header). The
// seeded sub-agent transcript carries the SAME turn_complete boundary rows the
// real ProcessManager.consume() appends (verified separately by
// tests/unit/test_process_manager_turn_boundary.py), so /chat exercises the
// real read_sub_agent_summary split + _interleave_sub_agent_summaries pairing.
//
// EN + ZH: the sub-agent RESPONSE is backend/agent output (never translated),
// so both locales show the same text; ZH exercises the surrounding UI chrome
// and the no-clipping invariant for the full (uncapped) message at 375px.

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DAEMON_PORT = 8348;
const VITE_PORT = 5274;
const REPO_ROOT = join(HERE, '..', '..');
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const PID_FILE = join(homedir(), 'orbital', 'daemon.pid');
const API = `http://127.0.0.1:${DAEMON_PORT}`;
const SHOTS = join(REPO_ROOT, 'docs', 'screenshots', 'subagent-bubbles');

const HANDLE = 'claude-code';
const T = (s: number) => `2026-06-13T10:0${s}:00+00:00`;

// Distinct, long-ish responses so wrapping/clipping is exercised at 375px.
const R1 = 'Created hello.txt containing the text "hi". The file now exists in the workspace and is ready to use.';
const R2 = 'Created bye.txt as you requested. Both hello.txt and bye.txt now exist in the project directory together.';
const ERR_R1 = 'Investigated the login race and patched auth.py:212 with the missing await; the full test suite passes.';

// Short names: the mobile sidebar truncates a project button past ~21 chars
// (e.g. "e2e-subagent-multiturn" → "e2e-subagent-multitu…"), which breaks a
// full-string getByText locator. Keep these well under that bound.
const PROJECT_MULTI = 'sa-multiturn';
const PROJECT_ERR = 'sa-errored';

let daemon: ChildProcess | undefined;
let vite: ChildProcess | undefined;
let skipReason: string | undefined;

function pidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}
function singletonHeldByLiveProcess(): boolean {
  if (!existsSync(PID_FILE)) return false;
  const pid = Number.parseInt(readFileSync(PID_FILE, 'utf-8').trim(), 10);
  return Number.isFinite(pid) && pidAlive(pid);
}
async function waitForHttp(url: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { const r = await fetch(url); if (r.ok) return true; } catch { /* not up */ }
    await new Promise((res) => setTimeout(res, 500));
  }
  return false;
}

async function createProject(name: string): Promise<string> {
  const workspace = mkdtempSync(join(tmpdir(), 'orbital-e2e-ws-'));
  const resp = await fetch(`${API}/api/v2/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, workspace, model: 'kimi-k2', api_key: 'e2e-test-key' }),
  });
  if (!resp.ok) throw new Error(`create project ${name} failed: HTTP ${resp.status}`);
  return workspace;
}

/** Seed a sub-agent transcript with real turn_complete boundary rows; return abs path. */
function seedTranscript(workspace: string, turns: Array<Record<string, unknown>[]>): string {
  const dir = join(workspace, 'orbital', 'sub_agents', HANDLE);
  mkdirSync(dir, { recursive: true });
  const path = join(dir, 'run.jsonl');
  const lines: string[] = [];
  let t = 0;
  for (const turn of turns) {
    for (const chunk of turn) lines.push(JSON.stringify({ source: HANDLE, timestamp: T(t++), ...chunk }));
    // boundary row the real consume() appends at turn_complete (empty content).
    lines.push(JSON.stringify({ source: HANDLE, content: '', chunk_type: 'turn_complete', timestamp: T(t++) }));
  }
  writeFileSync(path, lines.join('\n') + '\n', 'utf-8');
  return path;
}

function seedSession(workspace: string, sid: string, messages: Array<Record<string, unknown>>): void {
  const dir = join(workspace, 'orbital', 'sessions');
  mkdirSync(dir, { recursive: true });
  const meta = { type: 'session_start', session_id: sid, session_uuid: sid, name: sid, timestamp: T(0) };
  const lines = [meta, ...messages]
    .map((m) => JSON.stringify({ session_id: sid, session_uuid: sid, ...m }))
    .join('\n');
  writeFileSync(join(dir, `${sid}.jsonl`), lines + '\n', 'utf-8');
}

const routed = (preview: string, tx: string, ts: string) =>
  ({ role: 'system', source: 'daemon', timestamp: ts,
     content: `[Sub-agent] Message sent to ${HANDLE}: "${preview}". Transcript: ${tx}` });
const completed = (summary: string, tx: string, ts: string) =>
  ({ role: 'system', source: 'daemon', timestamp: ts,
     content: `[Sub-agent] ${HANDLE} completed. Summary: ${summary}. Transcript: ${tx}` });
const erroredMarker = (tx: string, ts: string) =>
  ({ role: 'system', source: 'daemon', timestamp: ts,
     content: `[Sub-agent] ${HANDLE} stopped with error: write blocked. Transcript: ${tx}` });

test.beforeAll(async () => {
  if (singletonHeldByLiveProcess()) {
    skipReason = `Daemon singleton PID ${PID_FILE} held by a LIVE process (stop Orbital.app to run).`;
    return;
  }
  mkdirSync(SHOTS, { recursive: true });
  const dataDir = mkdtempSync(join(tmpdir(), 'orbital-e2e-data-'));
  daemon = spawn(
    PYTHON,
    ['-m', 'uvicorn', 'agent_os.api.app:create_app', '--factory', '--port', String(DAEMON_PORT), '--host', '127.0.0.1'],
    { cwd: dataDir, env: { ...process.env, PYTHONPATH: REPO_ROOT, PYTHON_KEYRING_BACKEND: 'in-memory', AGENT_OS_API_KEY: 'e2e-test-key' }, stdio: 'pipe' },
  );
  daemon.stderr?.on('data', (d) => process.stdout.write(`[daemon] ${d}`));
  if (!(await waitForHttp(`${API}/api/v2/settings`, 20_000))) {
    skipReason = `Isolated daemon failed to come up on :${DAEMON_PORT}.`;
    return;
  }

  // Scenario 1 — two successful dispatches → two distinct bubbles.
  const wsMulti = await createProject(PROJECT_MULTI);
  const txMulti = seedTranscript(wsMulti, [
    [{ content: '[Using tool: Write]', chunk_type: 'tool_activity' }, { content: R1, chunk_type: 'response' }],
    [{ content: '[Using tool: Write]', chunk_type: 'tool_activity' }, { content: R2, chunk_type: 'response' }],
  ]);
  seedSession(wsMulti, 'multi', [
    { role: 'user', content: 'create hello.txt with hi', source: 'user', timestamp: T(0) },
    routed('create hello.txt with hi', txMulti, T(1)),
    completed(R1.slice(0, 60), txMulti, T(2)),
    { role: 'user', content: 'now also create bye.txt', source: 'user', timestamp: T(3) },
    routed('now also create bye.txt', txMulti, T(4)),
    completed(R2.slice(0, 60), txMulti, T(5)),
  ]);

  // Scenario 2 — dispatch 1 succeeds, dispatch 2 errors → ONE bubble, honest one-liner for #2.
  const wsErr = await createProject(PROJECT_ERR);
  const txErr = seedTranscript(wsErr, [
    [{ content: '[Using tool: Edit]', chunk_type: 'tool_activity' }, { content: ERR_R1, chunk_type: 'response' }],
    [{ content: '[Using tool: Write]', chunk_type: 'tool_activity' }, { content: 'boom', chunk_type: 'error' }],
  ]);
  seedSession(wsErr, 'err', [
    { role: 'user', content: 'fix the login race', source: 'user', timestamp: T(0) },
    routed('fix the login race', txErr, T(1)),
    completed(ERR_R1.slice(0, 60), txErr, T(2)),
    { role: 'user', content: 'now also rotate the keys', source: 'user', timestamp: T(3) },
    routed('now also rotate the keys', txErr, T(4)),
    erroredMarker(txErr, T(5)),
  ]);

  vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(VITE_PORT)], {
    cwd: join(REPO_ROOT, 'web'),
    env: { ...process.env, VITE_API_URL: API, VITE_LOCAL_MODE: 'true' },
    stdio: 'pipe',
  });
  vite.stderr?.on('data', (d) => process.stdout.write(`[vite] ${d}`));
  if (!(await waitForHttp(`http://127.0.0.1:${VITE_PORT}/`, 30_000))) {
    skipReason = `Vite dev server failed to come up on :${VITE_PORT}.`;
  }
});

test.afterAll(async () => { vite?.kill('SIGTERM'); daemon?.kill('SIGTERM'); });

async function openProject(page: Page, projectName: string, locale: 'en' | 'zh'): Promise<void> {
  await page.addInitScript((loc) => { window.localStorage.setItem('orbital.locale', loc); }, locale);
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`, { waitUntil: 'domcontentloaded' });
  const proj = page.getByText(projectName).first();
  await proj.waitFor({ state: 'visible', timeout: 30_000 });
  await proj.click();
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
}

test.describe('sub-agent per-turn bubbles @ 375x667', () => {
  test('two successful dispatches render two DISTINCT full-message bubbles (EN)', async ({ page }) => {
    test.skip(!!skipReason, skipReason);
    await openProject(page, PROJECT_MULTI, 'en');

    const b1 = page.getByText(R1, { exact: false });
    const b2 = page.getByText(R2, { exact: false });
    await expect(b1).toBeVisible();
    await expect(b2).toBeVisible();
    // Each full message present exactly once → no aliasing/duplication.
    await expect(b1).toHaveCount(1);
    await expect(b2).toHaveCount(1);
    // The handle header renders for the sub-agent bubbles.
    await expect(page.getByText(HANDLE).first()).toBeVisible();

    await page.screenshot({ path: join(SHOTS, 'multiturn-en.png'), fullPage: true });
  });

  test('full (uncapped) message renders with no horizontal clipping (ZH)', async ({ page }) => {
    test.skip(!!skipReason, skipReason);
    await openProject(page, PROJECT_MULTI, 'zh');

    // Agent output is not translated — same text in ZH.
    const b1 = page.getByText(R1, { exact: false });
    await expect(b1).toBeVisible();
    await expect(page.getByText(R2, { exact: false })).toBeVisible();

    // No horizontal overflow at 375px: the bubble fits the viewport width.
    const box = await b1.boundingBox();
    expect(box).not.toBeNull();
    expect((box!.x + box!.width)).toBeLessThanOrEqual(375 + 1);
    const docOverflow = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(docOverflow).toBeLessThanOrEqual(1);

    await page.screenshot({ path: join(SHOTS, 'multiturn-zh.png'), fullPage: true });
  });

  test('errored 2nd dispatch degrades: one bubble, no aliased duplicate', async ({ page }) => {
    test.skip(!!skipReason, skipReason);
    await openProject(page, PROJECT_ERR, 'en');

    // Turn 1 full message renders exactly once.
    const ok = page.getByText(ERR_R1, { exact: false });
    await expect(ok).toBeVisible();
    await expect(ok).toHaveCount(1);   // turn-1 text NEVER aliased into the errored slot

    await page.screenshot({ path: join(SHOTS, 'errored-degradation-en.png'), fullPage: true });
  });
});

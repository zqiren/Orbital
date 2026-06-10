// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// TEST RULE 4 Playwright spec for the LOCKED queue-delete behavior:
//   "the running item's delete control is simply disabled until it idles"
//   — NOT a stop-first popup, just disabled.
//
// Asserts that, for a RUNNING queue item, the delete (X) control is rendered
// but DISABLED, and that for an IDLE (queued) item the same control is ENABLED.
// Mobile viewport 375x667 (playwright.config.ts).
//
// Harness: same isolated-daemon + Vite pattern as sub-agent-status-stop.spec.ts
// (skips cleanly when the packaged Orbital.app holds the singleton PID file —
// see that spec's header / playwright.config.ts SINGLETON NOTE). The queue
// snapshot endpoint is page.route-mocked so a real RUNNING item renders
// deterministically without spawning a live agent (a live spawn is a
// sandbox/cost hazard here; the running state is otherwise non-deterministic).

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DAEMON_PORT = 8351;
const VITE_PORT = 5281;
const REPO_ROOT = join(HERE, '..', '..');
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const PID_FILE = join(homedir(), 'orbital', 'daemon.pid');
const API = `http://127.0.0.1:${DAEMON_PORT}`;

const TS = '2026-06-06T10:00:00Z';
// Short name: the 375px sidebar ellipsizes long project names and getByText
// would never match the truncated label.
const PROJECT = 'e2e-qdel';

const RUNNING_ID = 'run-item-1';
const QUEUED_ID = 'queued-item-1';

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

function seedSession(workspace: string, sid: string): void {
  const dir = join(workspace, 'orbital', 'sessions');
  mkdirSync(dir, { recursive: true });
  const meta = { type: 'session_start', session_id: sid, session_uuid: sid, name: sid, timestamp: TS };
  const msg = { role: 'user', content: 'kick off the queue', source: 'user', timestamp: TS };
  const lines = [meta, msg]
    .map((m) => JSON.stringify({ session_id: sid, session_uuid: sid, ...m }))
    .join('\n');
  writeFileSync(join(dir, `${sid}.jsonl`), lines + '\n', 'utf-8');
}

async function createProject(name: string): Promise<string> {
  const workspace = mkdtempSync(join(tmpdir(), 'orbital-e2e-ws-'));
  const resp = await fetch(`${API}/api/v2/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, workspace, model: 'kimi-k2', api_key: 'e2e-test-key' }),
  });
  if (!resp.ok) throw new Error(`create project failed: HTTP ${resp.status}`);
  return workspace;
}

test.beforeAll(async () => {
  if (singletonHeldByLiveProcess()) {
    skipReason =
      `Daemon singleton PID file ${PID_FILE} is held by a LIVE process ` +
      `(likely the packaged Orbital.app). Stop it to run this spec.`;
    return;
  }

  const dataDir = mkdtempSync(join(tmpdir(), 'orbital-e2e-data-'));
  daemon = spawn(
    PYTHON,
    ['-m', 'uvicorn', 'agent_os.api.app:create_app', '--factory', '--port', String(DAEMON_PORT), '--host', '127.0.0.1'],
    {
      cwd: dataDir,
      env: { ...process.env, PYTHONPATH: REPO_ROOT, PYTHON_KEYRING_BACKEND: 'in-memory', AGENT_OS_API_KEY: 'e2e-test-key' },
      stdio: 'pipe',
    },
  );
  daemon.stderr?.on('data', (d) => process.stdout.write(`[daemon] ${d}`));

  if (!(await waitForHttp(`${API}/api/v2/settings`, 20_000))) {
    skipReason = `Isolated daemon failed to come up on :${DAEMON_PORT} within 20s.`;
    return;
  }

  const workspace = await createProject(PROJECT);
  seedSession(workspace, 'qdel-sess');

  vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(VITE_PORT)], {
    cwd: join(REPO_ROOT, 'web'),
    env: { ...process.env, VITE_API_URL: API, VITE_LOCAL_MODE: 'true' },
    stdio: 'pipe',
  });
  vite.stderr?.on('data', (d) => process.stdout.write(`[vite] ${d}`));

  if (!(await waitForHttp(`http://127.0.0.1:${VITE_PORT}/`, 30_000))) {
    skipReason = `Vite dev server failed to come up on :${VITE_PORT} within 30s.`;
  }
});

test.afterAll(async () => {
  vite?.kill('SIGTERM');
  daemon?.kill('SIGTERM');
});

/**
 * Route-mock the queue snapshot so a RUNNING item and a QUEUED (idle) item
 * render deterministically. Returns one of each so a single page can assert
 * the disabled-vs-enabled contrast.
 */
async function mockQueueSnapshot(page: Page): Promise<void> {
  const baseItem = {
    file_refs: [],
    priority: 0,
    review_before_advance: false,
    source: 'user',
    attempts: [] as unknown[],
    idempotency_key: null,
    interrupted_count: 0,
    created_at: TS,
  };
  await page.route('**/api/v2/projects/*/queue', (route) => {
    // Only intercept the snapshot GET; let mutations fall through (none fire here).
    if (route.request().method() !== 'GET') return route.continue();
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        version: 1,
        state: 'running',
        chat_session_id: 'qdel-sess',
        items: [
          { ...baseItem, id: RUNNING_ID, content: 'running task', state: 'running' },
          { ...baseItem, id: QUEUED_ID, content: 'queued task', state: 'queued' },
        ],
      }),
    });
  });
}

/** Open the project and switch to the Queue tab (projects default to Chat). */
async function openQueueTab(page: Page): Promise<void> {
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`);
  await page.getByText(PROJECT).first().click();
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Queue' }).click();
  await page.getByTestId('queue-section-now-running').waitFor({ timeout: 15_000 });
}

test.describe('queue delete control: disabled-until-idle @ 375x667', () => {
  test('RUNNING item delete control is DISABLED; QUEUED item delete control is ENABLED', async ({ page }, testInfo) => {
    test.skip(!!skipReason, skipReason);
    expect(testInfo.project.use.viewport).toEqual({ width: 375, height: 667 });

    await mockQueueSnapshot(page);
    await openQueueTab(page);

    // The running card renders with data-state="running".
    const runningCard = page.getByTestId(`queue-item-${RUNNING_ID}`);
    await expect(runningCard).toBeVisible();
    await expect(runningCard).toHaveAttribute('data-state', 'running');

    // LOCKED: its delete (X) control is present but DISABLED.
    const runningRemove = runningCard.getByRole('button', { name: /remove item/i });
    await expect(runningRemove).toBeVisible();
    await expect(runningRemove).toBeDisabled();
    await expect(runningRemove).toHaveAttribute('aria-disabled', 'true');

    // The queued (idle) card's delete control is ENABLED.
    const queuedCard = page.getByTestId(`queue-item-${QUEUED_ID}`);
    await expect(queuedCard).toBeVisible();
    await expect(queuedCard).toHaveAttribute('data-state', 'queued');
    const queuedRemove = queuedCard.getByRole('button', { name: /remove item/i });
    await expect(queuedRemove).toBeVisible();
    await expect(queuedRemove).toBeEnabled();
  });
});

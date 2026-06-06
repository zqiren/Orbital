// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// RULE 4 Playwright spec for Piece 3 Part D: the sub-agent status badge
// (working / background-running / idle) and the user stop control with the
// honest raw-detach warning. Mobile viewport 375x667 (playwright.config.ts).
//
// Harness: same isolated-daemon + Vite pattern as chat-reasoning-capsule
// (skips cleanly when the packaged Orbital.app holds the singleton PID file —
// see that spec's SINGLETON NOTE). The badge's two NEW endpoints are
// page.route-mocked so the three states render deterministically without
// spawning a live claude sub-agent (a live spawn is a sandbox/cost hazard
// here; the backend path is covered by tests/regression/test_piece3_user_stop
// and the TEST RULE 2 integration journey).

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DAEMON_PORT = 8349;
const VITE_PORT = 5279;
const REPO_ROOT = join(HERE, '..', '..');
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const PID_FILE = join(homedir(), 'orbital', 'daemon.pid');
const API = `http://127.0.0.1:${DAEMON_PORT}`;

const TS = '2026-06-06T10:00:00Z';
const PROJECT = 'e2e-subagent-badge-proj';
const BG_CMD = 'gh api repos/x/y/issues --paginate (mining loop)';

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
  const msg = { role: 'user', content: 'dispatch the radar baseline', source: 'user', timestamp: TS };
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
  seedSession(workspace, 'badge-sess');

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

/** Route-mock the two Piece-3 endpoints with a controllable status. */
async function mockSubAgentEndpoints(
  page: Page,
  status: 'running' | 'background-running' | 'idle',
  onStop?: () => void,
): Promise<void> {
  await page.route('**/api/v2/agents/*/sub-agents/status*', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        session_id: 'badge-sess',
        agents: [{
          handle: 'claude-code',
          display_name: 'Claude Code',
          status,
          background_commands:
            status === 'background-running' ? [BG_CMD] : [],
        }],
      }),
    }),
  );
  await page.route('**/api/v2/agents/*/sub-agents/claude-code/stop*', (route) => {
    onStop?.();
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        handle: 'claude-code', status: 'stopped',
        background_terminated: [BG_CMD],
        warning: 'raw-detach limitation',
      }),
    });
  });
}

async function openProject(page: Page): Promise<void> {
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`);
  await page.getByText(PROJECT).first().click();
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
}

test.describe('sub-agent badge + stop @ 375x667', () => {
  test('background-running badge renders with stop control', async ({ page }, testInfo) => {
    test.skip(!!skipReason, skipReason);
    expect(testInfo.project.use.viewport).toEqual({ width: 375, height: 667 });

    await mockSubAgentEndpoints(page, 'background-running');
    await openProject(page);

    const badge = page.getByTestId('sub-agent-badge-claude-code');
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText('Background');
    await expect(badge).toHaveAttribute('data-status', 'background-running');
    await expect(page.getByTestId('sub-agent-stop-claude-code')).toBeVisible();
  });

  test('stop flow: honest warning (commands + raw-detach), POST fired, outcome surfaced', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    let stopCalled = false;
    await mockSubAgentEndpoints(page, 'background-running', () => { stopCalled = true; });
    await openProject(page);

    await page.getByTestId('sub-agent-stop-claude-code').click();

    const dialog = page.getByTestId('sub-agent-stop-confirm');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText(BG_CMD);
    const warning = page.getByTestId('sub-agent-stop-warning');
    await expect(warning).toContainText(/nohup/);
    await expect(warning).toContainText(/cannot be guaranteed stopped/);

    await page.getByTestId('sub-agent-stop-confirm-button').click();
    await expect(page.getByTestId('sub-agent-stop-result')).toContainText(
      /1 background process/);
    expect(stopCalled).toBe(true);
  });

  test('working badge renders; idle badge has no stop control', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    await mockSubAgentEndpoints(page, 'running');
    await openProject(page);
    const badge = page.getByTestId('sub-agent-badge-claude-code');
    await expect(badge).toHaveText('Working');

    await mockSubAgentEndpoints(page, 'idle');
    await page.reload();
    await page.getByText(PROJECT).first().click();
    await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
    await expect(page.getByTestId('sub-agent-badge-claude-code')).toHaveText('Idle');
    await expect(page.getByTestId('sub-agent-stop-claude-code')).toHaveCount(0);
  });
});

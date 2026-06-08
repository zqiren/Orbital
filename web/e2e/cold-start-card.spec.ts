// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// RULE 4 Playwright spec for the cold-start consent card
// (TASK-context-md-cold-start). Mobile viewport 375x667 (iPhone SE).
//
// Asserts the DETERMINISTIC card gating (no live model needed):
//   1. An imported project (non-empty workspace, 0 sessions) shows the
//      "Scan this workspace?" consent card.
//   2. An empty-workspace project shows the normal "No messages yet" empty
//      state, NOT the card.
//   3. Clicking Skip dismisses the card and reveals the normal empty state.
//
// The Scan → agent-produces-files leg needs a real model turn and is covered
// by the integration test (session-start) + the live-daemon smoke, not here.
//
// Reuses the isolated-daemon + Vite boot harness from chat-reasoning-capsule.spec.ts.

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
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

// Names must stay <=20 chars: the Sidebar truncates project names to 20
// (Sidebar.tsx:175 truncate(project.name, 20)), which would break getByText.
const PROJECT_IMPORTED = 'cs-imported';
const PROJECT_EMPTY = 'cs-empty';

let daemon: ChildProcess | undefined;
let vite: ChildProcess | undefined;
let skipReason: string | undefined;

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function singletonHeldByLiveProcess(): boolean {
  if (!existsSync(PID_FILE)) return false;
  const pid = Number.parseInt(readFileSync(PID_FILE, 'utf-8').trim(), 10);
  return Number.isFinite(pid) && pidAlive(pid);
}

async function waitForHttp(url: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((res) => setTimeout(res, 500));
  }
  return false;
}

/** Create a project pointing at a fresh workspace; returns the workspace path. */
async function createProject(name: string): Promise<string> {
  const workspace = mkdtempSync(join(tmpdir(), 'orbital-e2e-cs-ws-'));
  const resp = await fetch(`${API}/api/v2/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, workspace, model: 'kimi-k2', api_key: 'e2e-test-key' }),
  });
  if (!resp.ok) throw new Error(`create project ${name} failed: HTTP ${resp.status}`);
  return workspace;
}

test.beforeAll(async () => {
  if (singletonHeldByLiveProcess()) {
    skipReason =
      `Daemon singleton PID file ${PID_FILE} is held by a LIVE process ` +
      `(likely the packaged Orbital.app on :8000). Stop it to run this spec.`;
    return;
  }

  const dataDir = mkdtempSync(join(tmpdir(), 'orbital-e2e-cs-data-'));
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

  // Imported project: write a real file so is_empty_workspace=false → card eligible.
  const wsImported = await createProject(PROJECT_IMPORTED);
  writeFileSync(join(wsImported, 'README.md'), '# A real imported project\n', 'utf-8');

  // Empty project: no user files → is_empty_workspace=true → no card.
  await createProject(PROJECT_EMPTY);

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

async function openProject(page: Page, projectName: string): Promise<void> {
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`);
  await page.getByText(projectName).first().click();
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
}

test.describe('cold-start consent card @ 375x667', () => {
  test('imported (non-empty) workspace shows the consent card', async ({ page }, testInfo) => {
    test.skip(!!skipReason, skipReason);
    expect(testInfo.project.use.viewport).toEqual({ width: 375, height: 667 });

    await openProject(page, PROJECT_IMPORTED);
    await expect(page.getByText('Scan this workspace?')).toBeVisible();
    await expect(page.getByRole('button', { name: /scan/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /skip/i })).toBeVisible();
  });

  test('empty workspace shows the normal empty state, not the card', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    await openProject(page, PROJECT_EMPTY);
    await expect(page.getByText('No messages yet', { exact: false })).toBeVisible();
    await expect(page.getByText('Scan this workspace?')).toHaveCount(0);
  });

  test('Skip dismisses the card and reveals the normal empty state', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    await openProject(page, PROJECT_IMPORTED);
    await expect(page.getByText('Scan this workspace?')).toBeVisible();
    await page.getByRole('button', { name: /skip/i }).click();
    await expect(page.getByText('Scan this workspace?')).toHaveCount(0);
    await expect(page.getByText('No messages yet', { exact: false })).toBeVisible();
  });
});

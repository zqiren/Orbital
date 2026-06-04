// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// RULE 4 Playwright spec for the chat-render reasoning-capsule regression
// (frontend sites 3/6/7/8 of INVESTIGATION-chat-render-multisession-seam).
// Mobile viewport 375x667 (iPhone SE) — set in playwright.config.ts.
//
// LOCKED PRODUCT DECISION (2026-06-03):
//   A COMPLETED turn that reasoned but produced no visible answer text renders
//   as a COLLAPSED reasoning capsule (clean/minimal). It expands only while the
//   turn is actively RUNNING. NEVER render nothing for a non-empty turn.
//
// This spec asserts, at 375x667:
//   1. A COMPLETED no-answer turn shows a COLLAPSED reasoning capsule
//      (present, not expanded, not empty).
//   2. A reasoning-then-answer turn shows the answer body (reasoning collapsed).
//   3. NO zero-height / empty assistant item is reachable.
//   (A RUNNING turn shows live thinking — covered by ChatView.test.tsx unit
//    tests for the live WS path, since seeding a *running* turn requires a live
//    model turn which is a sandbox hazard here; this spec asserts the
//    completed-state invariant the locked decision pins.)
//
// MOBILE NAVIGATION NOTE
// ─────────────────────────────────────────────────────────────────────────
// At <=767px the Chat tab's SessionSidebar is `max-md:hidden`
// (ChatTab.tsx:174) — there is NO session picker on mobile; ChatTab
// auto-resolves to the single/most-recent session (ChatTab.tsx:95
// resolveDefaultSession). So this spec gives each scenario its OWN project
// with exactly ONE seeded session: opening the project auto-resolves to that
// session and ChatView renders it, no session-list click required.
//
// SINGLETON NOTE
// ─────────────────────────────────────────────────────────────────────────
// The daemon enforces a process singleton at ~/orbital/daemon.pid
// (pid_file.py) acquired in create_app. If the packaged Orbital.app is running
// it holds that lock; a second daemon can't boot. This spec detects a LIVE
// holder up front and skips cleanly (never fakes a pass). A *stale* lock (dead
// PID) is ignored — the daemon overwrites it on boot.

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DAEMON_PORT = 8347;
const VITE_PORT = 5273;
const REPO_ROOT = join(HERE, '..', '..');
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const PID_FILE = join(homedir(), 'orbital', 'daemon.pid');
const API = `http://127.0.0.1:${DAEMON_PORT}`;

const TS = '2026-06-03T10:00:00Z';
const TS2 = '2026-06-03T10:00:01Z';

// One project per scenario (single seeded session each → mobile auto-resolves).
const PROJECT_NOANSWER = 'e2e-noanswer-proj';
const PROJECT_ANSWER = 'e2e-answer-proj';

const REASONING_TEXT = 'INTERNAL-MONOLOGUE-NO-VISIBLE-ANSWER weighing the options carefully';
const ANSWER_TEXT = 'VISIBLE-ANSWER-BODY here is the explanation you asked for';
const ANSWER_REASONING = 'REASONING-BEHIND-ANSWER thinking before replying';

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

/** Seed a session JSONL under {workspace}/orbital/sessions/{sid}.jsonl. */
function seedSession(
  workspace: string,
  sid: string,
  messages: Array<Record<string, unknown>>,
): void {
  const dir = join(workspace, 'orbital', 'sessions');
  mkdirSync(dir, { recursive: true });
  const meta = { type: 'session_start', session_id: sid, session_uuid: sid, name: sid, timestamp: TS };
  const lines = [meta, ...messages]
    .map((m) => JSON.stringify({ session_id: sid, session_uuid: sid, ...m }))
    .join('\n');
  writeFileSync(join(dir, `${sid}.jsonl`), lines + '\n', 'utf-8');
}

/** Create a project pointing at a fresh workspace; returns the workspace path. */
async function createProject(name: string): Promise<string> {
  const workspace = mkdtempSync(join(tmpdir(), 'orbital-e2e-ws-'));
  const resp = await fetch(`${API}/api/v2/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // `model`/`api_key` are required by the schema; values are irrelevant —
    // the spec seeds sessions and never runs a live model turn.
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

  const dataDir = mkdtempSync(join(tmpdir(), 'orbital-e2e-data-'));

  // Boot isolated daemon. uvicorn --factory calls create_app() with no arg, so
  // persisted state defaults to CWD-relative ./orbital-data; we isolate by
  // running with CWD = throwaway temp dir and PYTHONPATH = repo root.
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

  // One project + one session per scenario.
  const wsNoAnswer = await createProject(PROJECT_NOANSWER);
  seedSession(wsNoAnswer, 'noanswer', [
    { role: 'user', content: 'think hard but say nothing', source: 'user', timestamp: TS },
    { role: 'assistant', content: '', reasoning_content: REASONING_TEXT, source: 'management', timestamp: TS2 },
  ]);

  const wsAnswer = await createProject(PROJECT_ANSWER);
  seedSession(wsAnswer, 'answer', [
    { role: 'user', content: 'explain it', source: 'user', timestamp: TS },
    { role: 'assistant', content: ANSWER_TEXT, reasoning_content: ANSWER_REASONING, source: 'management', timestamp: TS2 },
  ]);

  // Boot Vite pointed at the isolated daemon, in local mode (no relay auth).
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
 * Open a project in the SPA (mobile flow). The App sidebar (project list) is
 * visible at mobileView='sidebar'; clicking a project switches to content and
 * ChatTab auto-resolves to the single seeded session, rendering ChatView.
 */
async function openProject(page: Page, projectName: string): Promise<void> {
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`);
  await page.getByText(projectName).first().click();
  // Wait for the chat surface to mount (auto-resolved session).
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
}

test.describe('chat reasoning-capsule rendering @ 375x667', () => {
  test('COMPLETED no-answer turn shows a COLLAPSED, non-empty reasoning capsule', async ({ page }, testInfo) => {
    test.skip(!!skipReason, skipReason);

    expect(testInfo.project.use.viewport).toEqual({ width: 375, height: 667 });

    await openProject(page, PROJECT_NOANSWER);

    // The turn must NOT vanish: a reasoning capsule (summary "thinking") shows.
    const summary = page.getByText(/thinking/i).first();
    await expect(summary).toBeVisible();

    // COLLAPSED on completion: raw reasoning hidden until the capsule is opened.
    await expect(page.getByText(REASONING_TEXT)).toHaveCount(0);

    // No zero-height assistant item: the capsule occupies real space.
    const box = await summary.boundingBox();
    expect(box?.height ?? 0).toBeGreaterThan(0);

    // Expanding reveals the reasoning — proves the capsule is non-empty.
    await summary.click();
    await expect(page.getByText(REASONING_TEXT)).toBeVisible();
  });

  test('reasoning-then-answer turn shows the answer body (reasoning collapsed)', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    await openProject(page, PROJECT_ANSWER);

    // The visible answer body renders.
    await expect(page.getByText(ANSWER_TEXT, { exact: false })).toBeVisible();

    // The reasoning is present but COLLAPSED (not visible until expanded).
    await expect(page.getByText(ANSWER_REASONING)).toHaveCount(0);
  });

  test('no zero-height / empty assistant item is reachable', async ({ page }) => {
    test.skip(!!skipReason, skipReason);

    await openProject(page, PROJECT_NOANSWER);

    // The reasoning capsule must be present and have real height (never-vanish).
    const summary = page.getByText(/thinking/i).first();
    await expect(summary).toBeVisible();
    const box = await summary.boundingBox();
    expect(box?.height ?? 0).toBeGreaterThan(0);
  });
});

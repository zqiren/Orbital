// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression spec for the card flex-collapse bug (DIAGNOSIS-request-credential-
// login-failures.md, E4 / TASK-credential-card-collapse-test-and-audit).
//
// ROOT CAUSE: the chat message list (ChatView.tsx:2025) is a height-constrained
// `flex flex-col overflow-y-auto` container. A direct flex child whose root has
// `overflow-hidden` gets an automatic minimum size of ZERO (CSS flexbox §4.5),
// so when the list overflows, that child is squeezed to ~2px (borders only)
// instead of the container scrolling. Children without `overflow-hidden` keep
// their min-content floor and cannot collapse. The fix is `shrink-0` on every
// card root rendered as a direct child of the list.
//
// This spec seeds an overflowing chat (40 filler messages) ending in all four
// card variants, then asserts each card's FLEX CHILD (the direct child of the
// scroll container) paints at full height, at BOTH 375x667 (iPhone SE, the
// config default) and a 1280x800 desktop viewport. Removing `shrink-0` from
// CredentialCard.tsx:90 or ApprovalCard.tsx:169 makes the pending-card tests
// fail (verified during task execution). The resolved one-liner variants
// (CredentialCard.tsx:43, ApprovalCard.tsx:103) have no `overflow-hidden`, so
// their min-content floor protects them by CSS physics; their assertions here
// are belt-and-suspenders coverage of the same invariant.
//
// Reuses the isolated-daemon + Vite boot harness from
// chat-reasoning-capsule.spec.ts (see its SINGLETON NOTE — skips cleanly when
// the daemon PID file is held by a live process).

import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DAEMON_PORT = 8349;
const VITE_PORT = 5275;
const REPO_ROOT = join(HERE, '..', '..');
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const PID_FILE = join(homedir(), 'orbital', 'daemon.pid');
const API = `http://127.0.0.1:${DAEMON_PORT}`;

// <=20 chars (Sidebar truncates project names to 20).
const PROJECT = 'cards-collapse';

// Unique per-card markers rendered inside each card variant.
const CRED_PENDING_DOMAIN = 'cred-pending.example.com';
const CRED_RESOLVED_DOMAIN = 'cred-resolved.example.com';
const APR_PENDING_WHAT = 'APPROVAL-PENDING-MARKER run ls in the workspace';
const APR_RESOLVED_WHAT = 'APPROVAL-RESOLVED-MARKER previously approved command';

const FILLER_COUNT = 40;

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

function ts(i: number): string {
  return `2026-06-12T10:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}Z`;
}

/** Seed a session JSONL under {workspace}/orbital/sessions/{sid}.jsonl. */
function seedSession(
  workspace: string,
  sid: string,
  messages: Array<Record<string, unknown>>,
): void {
  const dir = join(workspace, 'orbital', 'sessions');
  mkdirSync(dir, { recursive: true });
  const meta = { type: 'session_start', session_id: sid, session_uuid: sid, name: sid, timestamp: ts(0) };
  const lines = [meta, ...messages]
    .map((m) => JSON.stringify({ session_id: sid, session_uuid: sid, ...m }))
    .join('\n');
  writeFileSync(join(dir, `${sid}.jsonl`), lines + '\n', 'utf-8');
}

/** Create the project, or return null if it already exists (a failed test
 *  restarts the Playwright worker; beforeAll re-runs against the daemon the
 *  previous worker left up, where the project is already created + seeded). */
async function createProject(name: string): Promise<string | null> {
  const workspace = mkdtempSync(join(tmpdir(), 'orbital-e2e-cards-ws-'));
  const resp = await fetch(`${API}/api/v2/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, workspace, model: 'kimi-k2', api_key: 'e2e-test-key' }),
  });
  if (resp.ok) return workspace;
  const listed = await fetch(`${API}/api/v2/projects`);
  if (listed.ok) {
    const projects = (await listed.json()) as Array<{ name?: string }> | { projects?: Array<{ name?: string }> };
    const arr = Array.isArray(projects) ? projects : (projects.projects ?? []);
    if (arr.some((p) => p.name === name)) return null;
  }
  throw new Error(`create project ${name} failed: HTTP ${resp.status}`);
}

test.beforeAll(async () => {
  const daemonAlreadyUp = await waitForHttp(`${API}/api/v2/settings`, 1_000);

  if (!daemonAlreadyUp) {
    if (singletonHeldByLiveProcess()) {
      skipReason =
        `Daemon singleton PID file ${PID_FILE} is held by a LIVE process ` +
        `(likely the packaged Orbital.app on :8000). Stop it to run this spec.`;
      return;
    }

    const dataDir = mkdtempSync(join(tmpdir(), 'orbital-e2e-cards-data-'));
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
  }

  const ws = await createProject(PROJECT);
  if (ws === null) {
    // Project already created + seeded by a previous worker; just ensure Vite.
    if (!(await waitForHttp(`http://127.0.0.1:${VITE_PORT}/`, 1_000))) {
      skipReason = `Daemon survived a worker restart but Vite on :${VITE_PORT} did not.`;
    }
    return;
  }

  // 40 filler exchanges (overflow both viewports; CHAT_PAGE_SIZE=100 keeps all
  // of them in the initial load) followed by all four card variants.
  const filler: Array<Record<string, unknown>> = [];
  for (let i = 0; i < FILLER_COUNT; i++) {
    filler.push(
      i % 2 === 0
        ? { role: 'user', content: `filler message ${i}`, source: 'user', timestamp: ts(i + 1) }
        : { role: 'assistant', content: `filler reply ${i}`, source: 'management', timestamp: ts(i + 1) },
    );
  }

  const base = FILLER_COUNT + 1;
  seedSession(ws, 'cards', [
    ...filler,
    {
      role: 'system',
      content: 'Credential needed to continue',
      timestamp: ts(base),
      _meta: {
        approval_request: true,
        tool_name: 'request_credential',
        tool_call_id: 'cred-p-1',
        tool_args: {
          name: 'dify',
          domain: CRED_PENDING_DOMAIN,
          fields: ['email', 'password'],
          reason: 'login requires credentials',
        },
      },
    },
    {
      role: 'system',
      content: 'Credential provided earlier',
      timestamp: ts(base + 1),
      _meta: {
        approval_request: true,
        tool_name: 'request_credential',
        tool_call_id: 'cred-r-1',
        resolution: 'approved',
        tool_args: {
          name: 'dify2',
          domain: CRED_RESOLVED_DOMAIN,
          fields: ['email', 'password'],
          reason: 'already supplied',
        },
      },
    },
    {
      role: 'system',
      content: APR_PENDING_WHAT,
      timestamp: ts(base + 2),
      _meta: {
        approval_request: true,
        tool_name: 'shell',
        tool_call_id: 'apr-p-1',
        tool_args: { command: 'ls -la' },
      },
    },
    {
      role: 'system',
      content: APR_RESOLVED_WHAT,
      timestamp: ts(base + 3),
      _meta: {
        approval_request: true,
        tool_name: 'shell',
        tool_call_id: 'apr-r-1',
        resolution: 'approved',
        tool_args: { command: 'echo done' },
      },
    },
  ]);

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

async function openProject(page: Page): Promise<void> {
  await page.goto(`http://127.0.0.1:${VITE_PORT}/`);
  await page.getByText(PROJECT).first().click();
  await page.getByTestId('chat-tab').waitFor({ timeout: 15_000 });
}

/**
 * Measure the message-list FLEX CHILD containing `marker`: starting from the
 * innermost element whose text includes the marker, climb to the direct child
 * of the scroll container (the `overflow-y-auto flex-col` ancestor). Returns
 * that child's painted height plus whether the container actually overflows
 * (the precondition that creates shrink pressure — asserted so a too-short
 * seed can never turn this spec into a silent no-op).
 */
async function flexChildMetrics(
  page: Page,
  marker: string,
): Promise<{ height: number; containerOverflows: boolean } | null> {
  return page.evaluate((markerText) => {
    const leaves = Array.from(document.querySelectorAll('*')).filter(
      (el) =>
        (el.textContent ?? '').includes(markerText) &&
        !Array.from(el.children).some((c) => (c.textContent ?? '').includes(markerText)),
    );
    if (leaves.length === 0) return null;
    let node: Element = leaves[0];
    while (node.parentElement) {
      const p = node.parentElement;
      if (p.classList.contains('overflow-y-auto') && p.classList.contains('flex-col')) {
        return {
          height: node.getBoundingClientRect().height,
          containerOverflows: p.scrollHeight > p.clientHeight,
        };
      }
      node = p;
    }
    return null;
  }, marker);
}

interface CardCase {
  name: string;
  marker: string;
  /** Collapse squeezes a card to ~2px; full pending cards are hundreds of px,
   *  resolved one-liners ~36px. */
  minHeight: number;
  kind: 'cred-pending' | 'cred-resolved' | 'approval-pending' | 'approval-resolved';
}

const CARD_CASES: CardCase[] = [
  { name: 'CredentialCard pending', marker: CRED_PENDING_DOMAIN, minHeight: 100, kind: 'cred-pending' },
  { name: 'CredentialCard resolved', marker: CRED_RESOLVED_DOMAIN, minHeight: 20, kind: 'cred-resolved' },
  { name: 'ApprovalCard pending', marker: APR_PENDING_WHAT, minHeight: 100, kind: 'approval-pending' },
  { name: 'ApprovalCard resolved', marker: APR_RESOLVED_WHAT, minHeight: 20, kind: 'approval-resolved' },
];

const VIEWPORTS = [
  { name: '375x667 mobile', width: 375, height: 667 },
  { name: '1280x800 desktop', width: 1280, height: 800 },
];

for (const vp of VIEWPORTS) {
  test.describe(`card flex-collapse @ ${vp.name}`, () => {
    for (const card of CARD_CASES) {
      test(`${card.name} paints at full height in an overflowing chat`, async ({ page }) => {
        test.skip(!!skipReason, skipReason);

        await page.setViewportSize({ width: vp.width, height: vp.height });
        await openProject(page);

        const markerEl = page.getByText(card.marker, { exact: false }).first();
        await markerEl.waitFor({ timeout: 15_000 });

        const metrics = await flexChildMetrics(page, card.marker);
        expect(metrics, `flex child containing "${card.marker}" not found`).not.toBeNull();

        // Precondition: the seeded chat really overflows the scroll container,
        // otherwise there is no shrink pressure and the test proves nothing.
        expect(metrics!.containerOverflows).toBe(true);

        // The regression: without shrink-0 the overflow-hidden card root
        // collapses to ~2px. Full height means the card survived.
        expect(metrics!.height).toBeGreaterThan(card.minHeight);

        // Pending cards must also be USABLE, not merely tall.
        if (card.kind === 'cred-pending') {
          const emailInput = page.getByPlaceholder('email');
          const passwordInput = page.getByPlaceholder('password');
          await emailInput.scrollIntoViewIfNeeded();
          await expect(emailInput).toBeVisible();
          await expect(passwordInput).toBeVisible();
          await emailInput.fill('user@example.com');
          await passwordInput.fill('hunter2-not-real');
          await expect(emailInput).toHaveValue('user@example.com');
          await expect(passwordInput).toHaveValue('hunter2-not-real');
        }
        if (card.kind === 'approval-pending') {
          const approve = page.getByRole('button', { name: /^approve$/i }).first();
          await approve.scrollIntoViewIfNeeded();
          await expect(approve).toBeVisible();
        }
      });
    }
  });
}

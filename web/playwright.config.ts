// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Playwright config for the chat-render reasoning-capsule regression spec
// (e2e/chat-reasoning-capsule.spec.ts). Mobile viewport 375x667.
//
// This config intentionally does NOT auto-start a daemon via `webServer`:
// the daemon enforces a singleton PID file at ~/orbital/daemon.pid with no
// env override (agent_os/utils/pid_file.py:20), so it collides with any
// already-running packaged Orbital.app. The spec boots an isolated, fixture-
// seeded daemon + Vite itself inside a global setup helper, and SKIPS cleanly
// (not fails) when the singleton PID file is already held. See the spec header.

import { defineConfig, devices } from '@playwright/test';

const IPHONE_SE = { width: 375, height: 667 };

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  timeout: 60_000,
  use: {
    ...devices['iPhone SE'],
    viewport: IPHONE_SE, // exact 375x667 regardless of the device preset
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'mobile-chromium',
      use: { browserName: 'chromium', viewport: IPHONE_SE },
    },
  ],
});

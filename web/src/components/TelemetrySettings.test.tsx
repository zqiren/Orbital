// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

import TelemetrySettings from './TelemetrySettings';

const PING = {
  schema: 1,
  install_id: 'inst_abc123def456',
  account_id: null,
  version: '0.8.4',
  os: 'darwin',
  date: '2026-08-08',
  first_seen: '2026-08-01',
  milestones: { key_set: true, first_project: false, first_session: false, first_turn: false },
  counters: { app_starts: 1, projects_created: 0, sessions: 0, turns: 0, errors: {}, tokens_by_provider: {} },
};

let fetchCalls: { url: string; init?: RequestInit }[] = [];
let payloadResponse: unknown = { last_sent: null, next_pending: PING };
let settingsResponse: unknown = { telemetry_enabled: true };

function jsonResponse(body: unknown) {
  return Promise.resolve({ json: () => Promise.resolve(body) } as Response);
}

beforeEach(() => {
  fetchCalls = [];
  payloadResponse = { last_sent: null, next_pending: PING };
  settingsResponse = { telemetry_enabled: true };
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    fetchCalls.push({ url, init });
    if (url.includes('telemetry-payload')) return jsonResponse(payloadResponse);
    return jsonResponse(settingsResponse);
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('TelemetrySettings', () => {
  it('renders the next-pending payload JSON verbatim', async () => {
    render(<TelemetrySettings />);
    const pre = await screen.findByTestId('telemetry-next-pending');
    // Verbatim contract: the viewer shows the exact JSON, field names included.
    expect(pre.textContent).toBe(JSON.stringify(PING, null, 2));
    expect(screen.getByText('Nothing has been sent yet.')).toBeInTheDocument();
  });

  it('renders last-sent when present', async () => {
    payloadResponse = { last_sent: PING, next_pending: PING };
    render(<TelemetrySettings />);
    const pre = await screen.findByTestId('telemetry-last-sent');
    expect(pre.textContent).toContain('inst_abc123def456');
  });

  it('toggle reflects settings and persists a PUT on flip', async () => {
    render(<TelemetrySettings />);
    const toggle = await screen.findByTestId('telemetry-toggle');
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'));

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    await waitFor(() => {
      const put = fetchCalls.find((c) => c.init?.method === 'PUT');
      expect(put).toBeTruthy();
      expect(JSON.parse(String(put!.init!.body))).toEqual({ telemetry_enabled: false });
    });
  });

  it('starts checked=false when settings say disabled', async () => {
    settingsResponse = { telemetry_enabled: false };
    render(<TelemetrySettings />);
    const toggle = await screen.findByTestId('telemetry-toggle');
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'));
  });
});

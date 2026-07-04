// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ProjectConnectorToggles unit tests — Task E1 (Spec 011 §0.2/§0.6).
 *
 * Covers (per the plan's Vitest list — toggle payload):
 *  - one switch per connector; checked state mirrors enabledConnectors for
 *    globally-connected connectors;
 *  - toggling calls onChange with the full next id list (add / remove) — the
 *    parent form (SettingsView) owns the state and saves it via the existing
 *    project-update flow;
 *  - connectors not connected globally render disabled with the "connect it
 *    in Global Settings first" hint;
 *  - empty catalog renders the empty state.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Connector } from '../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import ProjectConnectorToggles from './ProjectConnectorToggles';

afterEach(() => cleanup());

function makeConnector(overrides: Partial<Connector> = {}): Connector {
  return {
    id: 'google-calendar',
    name: 'Google Calendar',
    icon: '📅',
    auth_provider: 'google',
    auth_type: 'oauth2',
    server_url: 'https://mcp.example/calendar',
    oauth_scopes: [],
    tool_overrides: {},
    featured: true,
    status: 'available',
    connected: true,
    account: 'zq@example.com',
    enabled_error: null,
    ...overrides,
  };
}

const CALENDAR = makeConnector();
const DRIVE = makeConnector({ id: 'google-drive', name: 'Google Drive', icon: '📁' });
const GMAIL = makeConnector({
  id: 'gmail',
  name: 'Gmail',
  icon: '✉️',
  status: 'pending_verification',
  connected: false,
  account: null,
});

beforeEach(() => {
  apiMock.mockReset();
});

function mockList(connectors: Connector[]) {
  apiMock.mockResolvedValue({ connectors });
}

function toggle(id: string): HTMLInputElement {
  return screen.getByTestId(`project-connector-toggle-${id}`) as HTMLInputElement;
}

describe('ProjectConnectorToggles', () => {
  it('renders one switch per connector, checked per enabledConnectors', async () => {
    mockList([CALENDAR, DRIVE]);
    render(
      <ProjectConnectorToggles
        enabledConnectors={['google-drive']}
        onChange={vi.fn()}
      />,
    );
    await waitFor(() => expect(toggle('google-calendar')).toBeInTheDocument());
    expect(toggle('google-calendar').checked).toBe(false);
    expect(toggle('google-drive').checked).toBe(true);
    expect(toggle('google-calendar').disabled).toBe(false);
  });

  it('toggling ON calls onChange with the id appended (payload the form saves)', async () => {
    mockList([CALENDAR, DRIVE]);
    const onChange = vi.fn();
    render(
      <ProjectConnectorToggles enabledConnectors={['google-drive']} onChange={onChange} />,
    );
    await waitFor(() => expect(toggle('google-calendar')).toBeInTheDocument());
    fireEvent.click(toggle('google-calendar'));
    expect(onChange).toHaveBeenCalledWith(['google-drive', 'google-calendar']);
  });

  it('toggling OFF calls onChange with the id removed', async () => {
    mockList([CALENDAR, DRIVE]);
    const onChange = vi.fn();
    render(
      <ProjectConnectorToggles
        enabledConnectors={['google-drive', 'google-calendar']}
        onChange={onChange}
      />,
    );
    await waitFor(() => expect(toggle('google-drive')).toBeInTheDocument());
    fireEvent.click(toggle('google-drive'));
    expect(onChange).toHaveBeenCalledWith(['google-calendar']);
  });

  it('a connector that is not connected globally renders disabled with the connect-first hint', async () => {
    mockList([CALENDAR, GMAIL]);
    const onChange = vi.fn();
    render(<ProjectConnectorToggles enabledConnectors={[]} onChange={onChange} />);
    await waitFor(() => expect(toggle('gmail')).toBeInTheDocument());
    // The disabled attribute is the interaction guard (real browsers swallow
    // clicks on disabled inputs; jsdom's fireEvent bypasses it, so asserting
    // a synthetic click here would test jsdom, not the component).
    expect(toggle('gmail').disabled).toBe(true);
    expect(screen.getByTestId('project-connector-hint-gmail').textContent).toBe(
      'Connect it in Global Settings first',
    );
    // Connected connectors carry no hint.
    expect(
      screen.queryByTestId('project-connector-hint-google-calendar'),
    ).not.toBeInTheDocument();
  });

  it('a stale enabled id for a disconnected connector renders unchecked (not silently active)', async () => {
    mockList([GMAIL]);
    render(<ProjectConnectorToggles enabledConnectors={['gmail']} onChange={vi.fn()} />);
    await waitFor(() => expect(toggle('gmail')).toBeInTheDocument());
    expect(toggle('gmail').checked).toBe(false);
  });

  it('renders the empty state when the catalog is empty', async () => {
    mockList([]);
    render(<ProjectConnectorToggles enabledConnectors={[]} onChange={vi.fn()} />);
    await waitFor(() =>
      expect(
        screen.getByText(
          'No connectors available. Add or connect services in Global Settings → Connectors.',
        ),
      ).toBeInTheDocument(),
    );
  });

  it('renders the load error when the fetch fails', async () => {
    apiMock.mockRejectedValue(new Error('network down'));
    render(<ProjectConnectorToggles enabledConnectors={[]} onChange={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getByText("Couldn't load connectors.")).toBeInTheDocument(),
    );
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act, waitFor } from '@testing-library/react';

const onMock = vi.fn();
const offMock = vi.fn();

vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: onMock,
    off: offMock,
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

import UpdatePill from './UpdatePill';

let statusResponse: unknown = { current: '0.8.4', update_available: false, latest: null, url: null };

function emitUpdate(version = '0.9.0', url = 'https://gh/rel') {
  const handler = onMock.mock.calls.find((c) => c[0] === 'update.available')?.[1] as
    | ((e: unknown) => void)
    | undefined;
  act(() => handler?.({ type: 'update.available', version, url }));
}

beforeEach(() => {
  onMock.mockClear();
  offMock.mockClear();
  localStorage.clear();
  statusResponse = { current: '0.8.4', update_available: false, latest: null, url: null };
  vi.stubGlobal('fetch', () =>
    Promise.resolve({ json: () => Promise.resolve(statusResponse) } as Response),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('UpdatePill', () => {
  it('renders nothing without an update', async () => {
    render(<UpdatePill />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.queryByTestId('update-pill')).toBeNull();
  });

  it('shows the pill on a WS announce, linking to the release page', async () => {
    render(<UpdatePill />);
    emitUpdate('0.9.0', 'https://gh/rel');
    const pill = await screen.findByTestId('update-pill');
    const link = pill.querySelector('a')!;
    expect(link).toHaveAttribute('href', 'https://gh/rel');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link.textContent).toContain('0.9.0');
  });

  it('shows the pill from the mount fetch (page load after the announce)', async () => {
    statusResponse = { current: '0.8.4', update_available: true, latest: '0.9.1', url: 'https://gh/r2' };
    render(<UpdatePill />);
    const pill = await screen.findByTestId('update-pill');
    expect(pill.querySelector('a')!.textContent).toContain('0.9.1');
  });

  it('dismiss hides the pill and persists per version', async () => {
    render(<UpdatePill />);
    emitUpdate('0.9.0');
    await screen.findByTestId('update-pill');
    fireEvent.click(screen.getByTestId('update-pill-dismiss'));
    expect(screen.queryByTestId('update-pill')).toBeNull();
    expect(localStorage.getItem('orbital.updateDismissed')).toBe('0.9.0');

    // Remount (next page load): same version stays hidden.
    cleanup();
    render(<UpdatePill />);
    emitUpdate('0.9.0');
    await act(async () => { await Promise.resolve(); });
    expect(screen.queryByTestId('update-pill')).toBeNull();
  });

  it('a NEWER version reappears after dismissing the previous one', async () => {
    localStorage.setItem('orbital.updateDismissed', '0.9.0');
    render(<UpdatePill />);
    emitUpdate('0.10.0');
    await waitFor(() => expect(screen.getByTestId('update-pill')).toBeInTheDocument());
  });
});

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

import UpdatePill, { useUpdatePillShown } from './UpdatePill';

function ShownProbe() {
  return <span data-testid="shown-probe">{String(useUpdatePillShown())}</span>;
}
import { LocaleProvider } from '../i18n/LocaleContext';

let statusResponse: unknown = { current: '0.8.4', update_available: false, latest: null, url: null };

function emitUpdate(version = '0.9.0', url = 'https://gh/rel', installer: unknown = null) {
  const handler = onMock.mock.calls.find((c) => c[0] === 'update.available')?.[1] as
    | ((e: unknown) => void)
    | undefined;
  act(() => handler?.({ type: 'update.available', version, url, installer }));
}

const INSTALLER = {
  global: 'https://github.com/zqiren/Orbital/releases/latest/download/Orbital-macOS.dmg',
  china: 'https://orbital-release.oss-cn-hongkong.aliyuncs.com/releases/Orbital-macOS.dmg',
};

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

  it('links straight to the installer for an English UI (GitHub)', async () => {
    render(<UpdatePill />);
    emitUpdate('0.9.0', 'https://site', INSTALLER);
    const pill = await screen.findByTestId('update-pill');
    expect(pill.querySelector('a')!).toHaveAttribute('href', INSTALLER.global);
  });

  it('a Chinese UI gets the China mirror, the same rule as the provider region default', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    render(<LocaleProvider><UpdatePill /></LocaleProvider>);
    emitUpdate('0.9.0', 'https://site', INSTALLER);
    const pill = await screen.findByTestId('update-pill');
    expect(pill.querySelector('a')!).toHaveAttribute('href', INSTALLER.china);
  });

  it('the mount fetch carries the installer too', async () => {
    statusResponse = { current: '0.8.4', update_available: true, latest: '0.9.1', url: 'https://site', installer: INSTALLER };
    render(<UpdatePill />);
    const pill = await screen.findByTestId('update-pill');
    expect(pill.querySelector('a')!).toHaveAttribute('href', INSTALLER.global);
  });

  it('no installer (Linux, or an older daemon) keeps the website link', async () => {
    render(<UpdatePill />);
    emitUpdate('0.9.0', 'https://site', null);
    const pill = await screen.findByTestId('update-pill');
    expect(pill.querySelector('a')!).toHaveAttribute('href', 'https://site');
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

  // --- in-app upgrade (agent_os/updater.py) ---------------------------------

  describe('one-click upgrade', () => {
    const base = { version: '0.24.0', progress: 0, error: null, can_install: true };
    let installResponses: unknown[] = [];
    let posts: unknown[] = [];

    beforeEach(() => {
      installResponses = [];
      posts = [];
      vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
        if (url.endsWith('/api/v2/update/install') && init?.method === 'POST') {
          posts.push(JSON.parse(String(init.body)));
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ ...base, state: 'downloading' }) } as Response);
        }
        if (url.endsWith('/api/v2/update/install')) {
          const next = installResponses.shift() ?? { ...base, state: 'downloading', progress: 50 };
          return Promise.resolve({ ok: true, json: () => Promise.resolve(next) } as Response);
        }
        return Promise.resolve({ json: () => Promise.resolve(statusResponse) } as Response);
      });
    });

    function status(install: unknown) {
      statusResponse = { current: '0.23.1', update_available: true, latest: '0.24.0',
        url: 'https://site', installer: INSTALLER, install };
    }

    it('offers the update in place and starts it with the UI region', async () => {
      status({ ...base, state: 'idle' });
      render(<UpdatePill />);
      const btn = await screen.findByTestId('update-pill-action');
      expect(btn.textContent).toBe('Update to v0.24.0');
      expect(screen.getByTestId('update-pill').querySelector('a')).toBeNull();
      await act(async () => { fireEvent.click(btn); });
      expect(posts).toEqual([{ region: 'global' }]);
      expect((await screen.findByTestId('update-pill-progress')).textContent).toBe('Downloading 0%');
    });

    it('a Chinese UI starts from the China mirror and speaks zh', async () => {
      localStorage.setItem('orbital.locale', 'zh');
      status({ ...base, state: 'idle' });
      render(<LocaleProvider><UpdatePill /></LocaleProvider>);
      const btn = await screen.findByTestId('update-pill-action');
      expect(btn.textContent).toBe('更新到 v0.24.0');
      await act(async () => { fireEvent.click(btn); });
      expect(posts).toEqual([{ region: 'china' }]);
      expect((await screen.findByTestId('update-pill-progress')).textContent).toBe('下载中 0%');
    });

    it('shows a quiet percentage and a bar while downloading, no dismiss', async () => {
      status({ ...base, state: 'downloading', progress: 37 });
      render(<UpdatePill />);
      expect((await screen.findByTestId('update-pill-progress')).textContent).toBe('Downloading 37%');
      expect(screen.getByTestId('update-pill-bar')).toHaveStyle({ width: '37%' });
      expect(screen.queryByTestId('update-pill-dismiss')).toBeNull();
    });

    it('polls while a run is in flight and follows it to restarting', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        installResponses = [{ ...base, state: 'downloading', progress: 80 },
          { ...base, state: 'verifying', progress: 100 },
          { ...base, state: 'installing', progress: 100 }];
        status({ ...base, state: 'downloading', progress: 10 });
        render(<UpdatePill />);
        await screen.findByText('Downloading 10%');
        await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
        await screen.findByText('Downloading 80%');
        await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
        await screen.findByText('Verifying…');
        await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
        await screen.findByText('Restarting…');
      } finally {
        vi.useRealTimers();
      }
    });

    it('after the quit was cancelled it offers a restart', async () => {
      status({ ...base, state: 'ready', progress: 100 });
      render(<UpdatePill />);
      expect((await screen.findByTestId('update-pill-action')).textContent).toBe('Restart to update');
    });

    it('a failure shows one line with retry and the manual download', async () => {
      status({ ...base, state: 'failed', error: 'hash' });
      render(<UpdatePill />);
      const row = await screen.findByTestId('update-pill-failed');
      expect(row.textContent).toContain('Update failed');
      expect(screen.getByTestId('update-pill-manual')).toHaveAttribute('href', INSTALLER.global);
      await act(async () => { fireEvent.click(screen.getByTestId('update-pill-retry')); });
      expect(posts).toHaveLength(1);
    });

    it('an app that cannot update itself keeps the manual download link', async () => {
      status({ ...base, state: 'idle', can_install: false });
      render(<UpdatePill />);
      const pill = await screen.findByTestId('update-pill');
      expect(pill.querySelector('a')!).toHaveAttribute('href', INSTALLER.global);
      expect(screen.queryByTestId('update-pill-action')).toBeNull();
    });

    it('a refused click falls back to the manual download', async () => {
      vi.stubGlobal('fetch', (_url: string, init?: RequestInit) => {
        if (init?.method === 'POST') return Promise.resolve({ ok: false, json: () => Promise.resolve({}) } as Response);
        return Promise.resolve({ json: () => Promise.resolve(statusResponse) } as Response);
      });
      status({ ...base, state: 'idle' });
      render(<UpdatePill />);
      const btn = await screen.findByTestId('update-pill-action');
      await act(async () => { fireEvent.click(btn); });
      expect(screen.getByTestId('update-pill-manual')).toHaveAttribute('href', INSTALLER.global);
      expect(screen.queryByTestId('update-pill-retry')).toBeNull();
    });

    // The pill takes the sidebar's connection line's place (Sidebar.tsx reads
    // useUpdatePillShown); a line that is not "connected" outranks it.
    it('reports itself shown to the sidebar footer, and not once dismissed', async () => {
      status({ ...base, state: 'idle' });
      render(<><UpdatePill /><ShownProbe /></>);
      await screen.findByTestId('update-pill');
      expect(screen.getByTestId('shown-probe').textContent).toBe('true');
      fireEvent.click(screen.getByTestId('update-pill-dismiss'));
      expect(screen.getByTestId('shown-probe').textContent).toBe('false');
    });

    it('steps aside while the connection is down, and returns on reconnect', async () => {
      status({ ...base, state: 'verifying', progress: 100 });
      const { rerender } = render(<><UpdatePill connectionState="disconnected" /><ShownProbe /></>);
      await waitFor(() => expect(screen.getByTestId('shown-probe').textContent).toBe('false'));
      expect(screen.queryByTestId('update-pill')).toBeNull();
      rerender(<><UpdatePill connectionState="connected" /><ShownProbe /></>);
      expect((await screen.findByTestId('update-pill-status')).textContent).toBe('Verifying…');
      expect(screen.getByTestId('shown-probe').textContent).toBe('true');
    });

    it('keeps "Restarting…" through the outage the restart itself causes', async () => {
      status({ ...base, state: 'installing', progress: 100 });
      render(<><UpdatePill connectionState="reconnecting" /><ShownProbe /></>);
      expect((await screen.findByTestId('update-pill-status')).textContent).toBe('Restarting…');
      expect(screen.getByTestId('shown-probe').textContent).toBe('true');
    });

    it('unmounting clears the shown flag', async () => {
      status({ ...base, state: 'idle' });
      const { rerender } = render(<><UpdatePill /><ShownProbe /></>);
      await screen.findByTestId('update-pill');
      rerender(<ShownProbe />);
      expect(screen.getByTestId('shown-probe').textContent).toBe('false');
    });

    it('dismissing a failure hides the pill for that version', async () => {
      status({ ...base, state: 'failed', error: 'download' });
      render(<UpdatePill />);
      await screen.findByTestId('update-pill-failed');
      fireEvent.click(screen.getByTestId('update-pill-dismiss'));
      expect(screen.queryByTestId('update-pill')).toBeNull();
    });
  });
});

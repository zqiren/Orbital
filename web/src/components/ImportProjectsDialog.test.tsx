// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The import dialog's way out (batch 1026): it had no close button, and its
 * Cancel and Import sat at the end of the scrolling list, so a long list hid
 * both. Now a × in the header and a footer that stays in view.
 */
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { OnboardingScan } from '../hooks/useOnboardingScan';

const scan: OnboardingScan = {
  history: { status: 'ready', sessions_total: 3, by_agent: { 'claude-code': 3 } },
  importable: {
    status: 'ready',
    candidates: Array.from({ length: 30 }, (_, i) => ({
      source: 'claude-code', name: `proj-${i}`, path: `/w/proj-${i}`, session_count: 4,
      last_activity: null, already_imported: false, kind: 'folder', minor: false,
      is_git: true, by_agent: { 'claude-code': 4 },
    })),
    groups: [],
  },
  error: false,
  settled: true,
  retry: () => {},
};

vi.mock('../hooks/useOnboardingScan', () => ({ useOnboardingScan: () => scan }));
const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));
vi.mock('./FolderBrowserPanel', () => ({ default: () => null }));

import ImportProjectsDialog from './ImportProjectsDialog';

afterEach(() => { cleanup(); apiMock.mockReset(); });

describe('ImportProjectsDialog', () => {
  it('closes from the × in the header', () => {
    const onClose = vi.fn();
    render(<ImportProjectsDialog onClose={onClose} onImported={vi.fn()} />);
    fireEvent.click(screen.getByTestId('import-dialog-close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('keeps Cancel and Import in a footer outside the scrolling list', () => {
    const onClose = vi.fn();
    render(<ImportProjectsDialog onClose={onClose} onImported={vi.fn()} />);
    const footer = screen.getByTestId('import-actions');
    const list = screen.getByTestId('import-dialog-scroll');
    expect(list.contains(footer)).toBe(false);
    expect(footer.contains(screen.getByTestId('import-submit'))).toBe(true);
    fireEvent.click(screen.getByTestId('import-skip'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});


describe('ImportProjectsDialog — cost sign (spec 160 round 2)', () => {
  it('with a key: the token sentence beside Import, counting the checked rows', async () => {
    apiMock.mockResolvedValue({ llm: { api_key_set: true }, setup: {}, import_scan_agent: null });
    render(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    const sign = await screen.findByTestId('token-cost-sign');
    expect(sign.getAttribute('data-kind')).toBe('importKey');
    expect(sign.textContent).toBe('30');
  });

  it('keyless with a ready agent: the agent sentence (round 3: no Skip needed); no agent: no sign', async () => {
    apiMock.mockResolvedValue({ llm: { api_key_set: false }, setup: { api_key_skipped: false },
      import_scan_agent: { slug: 'claude-code', name: 'Claude Code' } });
    const { unmount } = render(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    expect((await screen.findByTestId('token-cost-sign')).getAttribute('data-kind')).toBe('importAgent');
    unmount();
    apiMock.mockResolvedValue({ llm: { api_key_set: false }, setup: {}, import_scan_agent: null });
    render(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(screen.queryByTestId('token-cost-sign')).toBeNull();
  });
});

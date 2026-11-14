// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The import dialog's way out (batch 1026): it had no close button, and its
 * Cancel and Import sat at the end of the scrolling list, so a long list hid
 * both. Now a × in the header and a footer that stays in view.
 */
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
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


describe('ImportProjectsDialog — no read (batch 1114)', () => {
  it('asks the daemon nothing before import and shows no cost sign; the intro names no count', () => {
    render(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    expect(apiMock).not.toHaveBeenCalled();
    expect(screen.queryByTestId('token-cost-sign')).toBeNull();
    expect(screen.getByText('Import projects')).toBeTruthy();
    expect(screen.getByText('After import, every agent in a project shares the same project memory.')).toBeTruthy();
  });
});

describe('ImportProjectsDialog — rows that arrive after the user chose (1111 pass 2, N2)', () => {
  it('a replacement list after an uncheck brings nothing checked', () => {
    const original = scan.importable;
    const rows = original!.candidates;
    scan.importable = { status: 'scanning', candidates: rows.slice(0, 3).map((c, i) => ({ ...c, minor: i === 2 })), groups: [] };
    const { rerender } = render(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    fireEvent.click(screen.getByTestId('import-check-/w/proj-0'));
    fireEvent.click(screen.getByTestId('import-check-/w/proj-1'));
    scan.importable = { status: 'ready', candidates: rows, groups: [] };
    rerender(<ImportProjectsDialog onClose={vi.fn()} onImported={vi.fn()} />);
    const checked = screen.getAllByRole('checkbox').filter((el) => (el as HTMLInputElement).checked);
    expect(checked).toHaveLength(0);
    expect(screen.getByTestId('import-submit').textContent).toBe('Import 0 projects');
    scan.importable = original;
  });
});

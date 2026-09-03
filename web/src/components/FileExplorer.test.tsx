// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render } from '@testing-library/react';

const { apiMock } = vi.hoisted(() => ({
  apiMock: vi.fn(async () => ({ entries: [] })),
}));

vi.mock('../config', () => ({
  api: apiMock,
  BASE_URL: '',
  isRelayMode: false,
}));

vi.mock('./FilePreview', () => ({
  default: () => <div>File preview</div>,
}));

import FileExplorer from './FileExplorer';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('FileExplorer layout', () => {
  it('keeps the tree pane flex layout free of the conflicting desktop block utility', async () => {
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<FileExplorer projectId="project-1" />));
    });
    const treePane = container.firstElementChild?.firstElementChild;

    expect(treePane).toHaveClass('flex', 'flex-col', 'min-h-0');
    expect(treePane).not.toHaveClass('md:block');
  });
});

describe('FileExplorer initialPath (spec 078 D15)', () => {
  it('expands the ancestors and selects the file, fetching its content', async () => {
    apiMock.mockImplementation((async (url: string) => {
      if (url.includes('/files/content?')) return { path: 'docs/notes/plan.md', content: '# plan', type: 'text' };
      if (url.endsWith('/files')) return { entries: [{ name: 'docs', type: 'directory' }] };
      if (url.includes('path=docs%2Fnotes')) return { entries: [{ name: 'plan.md', type: 'file', size: 6 }] };
      if (url.includes('path=docs')) return { entries: [{ name: 'notes', type: 'directory' }] };
      return { entries: [] };
    }) as unknown as () => Promise<{ entries: never[] }>);
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<FileExplorer projectId="project-1" initialPath="docs/notes/plan.md" />));
    });
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });

    const calls = (apiMock.mock.calls as unknown as unknown[][]).map((c) => String(c[0]));
    expect(calls.some((u) => u.includes('path=docs') && !u.includes('notes'))).toBe(true);
    expect(calls.some((u) => u.includes('path=docs%2Fnotes'))).toBe(true);
    expect(calls.some((u) => u.includes('/files/content?path=docs%2Fnotes%2Fplan.md'))).toBe(true);
    expect(container.textContent).toContain('plan.md');
  });
});

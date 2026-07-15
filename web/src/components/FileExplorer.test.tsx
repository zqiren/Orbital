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

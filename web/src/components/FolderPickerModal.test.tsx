// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, cleanup, act, screen, fireEvent } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

// Mock the api helper + ApiError (class declared inside the hoisted factory).
let apiFn = vi.fn();
vi.mock('../config', () => {
  class ApiError extends Error {
    constructor(public status: number, public detail: string) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return {
    api: (...args: unknown[]) => apiFn(...args),
    ApiError,
    isRelayMode: false,
    BASE_URL: 'http://localhost:8000',
  };
});

import FolderPickerModal from './FolderPickerModal';

beforeEach(() => {
  apiFn = vi.fn(async (path: string) => {
    if (typeof path === 'string' && path.startsWith('/api/v2/platform/browse')) {
      return {
        path: '/home/user',
        parent: '/home',
        display_name: 'user',
        entries: [
          { name: 'projects', path: '/home/user/projects', has_children: true },
          { name: 'leaf', path: '/home/user/leaf', has_children: false },
        ],
      };
    }
    if (path === '/api/v2/platform/folders') {
      return { status: 'ok', folders: [] };
    }
    throw new Error(`unexpected path: ${path}`);
  });
});

function browseCallCount() {
  return apiFn.mock.calls.filter(
    (c) => typeof c[0] === 'string' && c[0].startsWith('/api/v2/platform/browse'),
  ).length;
}

// Flush microtasks + a macrotask several times so that any render-driven
// re-fetch loop has room to manifest (each loop iteration needs a fetch to
// resolve and a React commit).
async function settle(rounds = 6) {
  for (let i = 0; i < rounds; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

describe('FolderPickerModal', () => {
  it('browses the default folder exactly once when opened (no re-fetch loop)', async () => {
    render(<FolderPickerModal open={true} onSelect={() => {}} onClose={() => {}} />);
    await settle();
    expect(browseCallCount()).toBe(1);
  });

  it('keeps the fetched entries clickable (does not stay in a loading loop)', async () => {
    const onSelect = vi.fn();
    render(<FolderPickerModal open={true} onSelect={onSelect} onClose={() => {}} />);
    await settle();

    // A leaf folder (no children) should be selectable on click.
    const leaf = screen.getByText('leaf');
    fireEvent.click(leaf);
    expect(onSelect).toHaveBeenCalledWith('/home/user/leaf');
  });
});

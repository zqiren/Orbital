// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * FolderBrowserPanel unit tests (backlog #25) — the browsing UI extracted
 * from FolderPickerModal so it can be embedded inline in CreateProject's
 * workspace picker. Covers: browse-once-on-mount (regression guard for the
 * infinite re-fetch loop FolderPickerModal.test.tsx already guards), leaf
 * click selects, "Use this folder" selects the currently-browsed directory
 * regardless of children, and the "New folder" affordance (create -> POST
 * /api/v2/platform/mkdir -> navigate into + select the created path; a
 * mkdir error surfaces inline, never as a toast).
 */

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, cleanup, act, screen, fireEvent, waitFor } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

const { apiFn, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    constructor(public status: number, public detail: string) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return { apiFn: vi.fn(), MockApiError };
});
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: MockApiError,
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
}));

import FolderBrowserPanel from './FolderBrowserPanel';

function mockBrowseResponses() {
  apiFn.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
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
    if (path === '/api/v2/platform/mkdir' && init?.method === 'POST') {
      const body = JSON.parse(init.body || '{}');
      return { status: 'ok', path: `${body.parent}/${body.name}` };
    }
    throw new Error(`unexpected path: ${path}`);
  });
}

async function settle(rounds = 4) {
  for (let i = 0; i < rounds; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

function browseCallCount() {
  return apiFn.mock.calls.filter(
    (c) => typeof c[0] === 'string' && c[0].startsWith('/api/v2/platform/browse'),
  ).length;
}

beforeEach(() => {
  apiFn.mockClear();
  mockBrowseResponses();
});

describe('FolderBrowserPanel', () => {
  it('browses the default folder exactly once on mount (no re-fetch loop)', async () => {
    render(<FolderBrowserPanel onSelect={() => {}} />);
    await settle();
    expect(browseCallCount()).toBe(1);
  });

  it('selects a leaf folder (no children) directly on click', async () => {
    const onSelect = vi.fn();
    render(<FolderBrowserPanel onSelect={onSelect} />);
    await settle();

    fireEvent.click(screen.getByText('leaf'));
    expect(onSelect).toHaveBeenCalledWith('/home/user/leaf');
  });

  it('navigates (does not select) into a folder that has children', async () => {
    const onSelect = vi.fn();
    render(<FolderBrowserPanel onSelect={onSelect} />);
    await settle();

    fireEvent.click(screen.getByText('projects'));
    await settle();
    expect(onSelect).not.toHaveBeenCalled();
    expect(browseCallCount()).toBe(2);
  });

  it('"Use this folder" selects the currently-browsed directory regardless of children', async () => {
    const onSelect = vi.fn();
    render(<FolderBrowserPanel onSelect={onSelect} />);
    await settle();

    fireEvent.click(screen.getByText('Use this folder'));
    expect(onSelect).toHaveBeenCalledWith('/home/user');
  });

  it('creates a new folder, navigates into it, and selects it', async () => {
    const onSelect = vi.fn();
    render(<FolderBrowserPanel onSelect={onSelect} />);
    await settle();

    fireEvent.click(screen.getByText('New folder'));
    const input = screen.getByPlaceholderText('Folder name');
    fireEvent.change(input, { target: { value: 'my-app' } });
    fireEvent.click(screen.getByText('Create'));
    await settle();

    expect(apiFn).toHaveBeenCalledWith('/api/v2/platform/mkdir', {
      method: 'POST',
      body: JSON.stringify({ parent: '/home/user', name: 'my-app' }),
    });
    // Navigated into the new folder...
    expect(browseCallCount()).toBe(2);
    expect(apiFn.mock.calls.some(
      (c) => typeof c[0] === 'string' && c[0].includes(encodeURIComponent('/home/user/my-app')),
    )).toBe(true);
    // ...and selected it.
    expect(onSelect).toHaveBeenCalledWith('/home/user/my-app');
  });

  it('shows a mkdir error inline instead of a toast, and does not select', async () => {
    const onSelect = vi.fn();
    apiFn.mockImplementation(async (path: string, init?: { method?: string }) => {
      if (typeof path === 'string' && path.startsWith('/api/v2/platform/browse')) {
        return {
          path: '/home/user',
          parent: '/home',
          display_name: 'user',
          entries: [],
        };
      }
      if (path === '/api/v2/platform/folders') return { status: 'ok', folders: [] };
      if (path === '/api/v2/platform/mkdir' && init?.method === 'POST') {
        throw new MockApiError(409, 'A folder with that name already exists');
      }
      throw new Error(`unexpected path: ${path}`);
    });

    render(<FolderBrowserPanel onSelect={onSelect} />);
    await settle();

    fireEvent.click(screen.getByText('New folder'));
    fireEvent.change(screen.getByPlaceholderText('Folder name'), { target: { value: 'dup' } });
    fireEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(screen.getByText('A folder with that name already exists')).toBeTruthy();
    });
    expect(onSelect).not.toHaveBeenCalled();
  });
});

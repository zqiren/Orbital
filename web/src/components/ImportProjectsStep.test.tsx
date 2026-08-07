// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ImportProjectsStep unit tests (backlog #34).
 *
 * Covers the locked MVP contract:
 *  - disclose-then-scan: the disclosure renders and NO scan fires on mount;
 *  - Scan calls GET /api/v2/onboarding/importable-projects and renders the
 *    ranked candidates in the order the API returned them;
 *  - confirm-each: clicking Add on a row POSTs the EXISTING /api/v2/projects
 *    with {name, workspace:path} (link-only) and the row flips to "Added";
 *  - a scanner/network failure surfaces an error + retry, never a broken list;
 *  - an empty result shows the empty state.
 *
 * The api client is mocked — no network.
 */

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

let apiFn = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => apiFn(...args),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
}));

import ImportProjectsStep from './ImportProjectsStep';

const IMPORT_PATH = '/api/v2/onboarding/importable-projects';

const CANDIDATES = [
  { source: 'claude-code', name: 'orbital', path: '/Users/x/orbital', session_count: 12, last_activity: '2026-08-01T00:00:00+00:00' },
  { source: 'codex', name: 'api-svc', path: '/Users/x/api-svc', session_count: 1, last_activity: '2026-07-20T00:00:00+00:00' },
  { source: 'obsidian', name: 'notes', path: '/Users/x/notes', session_count: 0, last_activity: '2026-08-05T00:00:00+00:00' },
];

beforeEach(() => {
  apiFn = vi.fn();
});
afterEach(() => cleanup());

describe('ImportProjectsStep', () => {
  it('shows the disclosure and does NOT scan on mount (disclose-then-scan)', () => {
    render(<ImportProjectsStep />);
    expect(screen.getByTestId('import-disclosure')).toBeInTheDocument();
    expect(screen.getByText(/never your conversations/)).toBeInTheDocument();
    expect(screen.getByTestId('import-scan')).toBeInTheDocument();
    // No network before the user opts in.
    expect(apiFn).not.toHaveBeenCalled();
  });

  it('scans on click and renders candidates in the returned order', async () => {
    apiFn.mockResolvedValueOnce({ candidates: CANDIDATES });
    render(<ImportProjectsStep />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-scan'));
    });

    expect(apiFn).toHaveBeenCalledWith(IMPORT_PATH);
    await waitFor(() => expect(screen.getByTestId('import-list')).toBeInTheDocument());

    const rows = screen.getAllByRole('listitem');
    expect(rows).toHaveLength(3);
    // Order preserved (server already ranked: agents above the vault).
    expect(rows[0]).toHaveTextContent('orbital');
    expect(rows[2]).toHaveTextContent('notes');
    // Obsidian shows "Vault", agent projects show a session count.
    expect(rows[2]).toHaveTextContent('Vault');
    expect(rows[0]).toHaveTextContent('12 sessions');
    expect(rows[1]).toHaveTextContent('1 session');
  });

  it('confirming a row POSTs the existing /api/v2/projects (link-only) and flips to Added', async () => {
    apiFn.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === IMPORT_PATH) return { candidates: CANDIDATES };
      if (path === '/api/v2/projects' && options?.method === 'POST') {
        return { project_id: 'p1', name: 'orbital', workspace: '/Users/x/orbital' };
      }
      throw new Error(`unexpected ${path}`);
    });
    const onCreated = vi.fn();
    render(<ImportProjectsStep onProjectCreated={onCreated} />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-scan'));
    });
    await waitFor(() => expect(screen.getByTestId('import-list')).toBeInTheDocument());

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-add-/Users/x/orbital'));
    });

    // Exact link-only create payload against the EXISTING projects endpoint.
    expect(apiFn).toHaveBeenCalledWith('/api/v2/projects', {
      method: 'POST',
      body: JSON.stringify({ name: 'orbital', workspace: '/Users/x/orbital' }),
    });
    await waitFor(() =>
      expect(screen.getByTestId('import-added-/Users/x/orbital')).toBeInTheDocument(),
    );
    expect(onCreated).toHaveBeenCalledTimes(1);
    // Other rows keep their Add buttons.
    expect(screen.getByTestId('import-add-/Users/x/api-svc')).toBeInTheDocument();
  });

  it('surfaces a create failure inline and leaves the row retryable', async () => {
    apiFn.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === IMPORT_PATH) return { candidates: CANDIDATES };
      if (path === '/api/v2/projects' && options?.method === 'POST') {
        throw new Error('workspace does not exist');
      }
      throw new Error(`unexpected ${path}`);
    });
    render(<ImportProjectsStep />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-scan'));
    });
    await waitFor(() => expect(screen.getByTestId('import-list')).toBeInTheDocument());

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-add-/Users/x/api-svc'));
    });

    await waitFor(() =>
      expect(screen.getByTestId('import-add-error-/Users/x/api-svc')).toBeInTheDocument(),
    );
    // Still retryable.
    expect(screen.getByTestId('import-add-/Users/x/api-svc')).toBeInTheDocument();
  });

  it('shows the empty state when nothing is found', async () => {
    apiFn.mockResolvedValueOnce({ candidates: [] });
    render(<ImportProjectsStep />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-scan'));
    });
    await waitFor(() => expect(screen.getByTestId('import-empty')).toBeInTheDocument());
  });

  it('shows an error with a retry when the scan fails', async () => {
    apiFn.mockRejectedValueOnce(new Error('boom'));
    render(<ImportProjectsStep />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('import-scan'));
    });
    await waitFor(() => expect(screen.getByTestId('import-error')).toBeInTheDocument());
    expect(screen.getByTestId('import-rescan')).toBeInTheDocument();
  });
});

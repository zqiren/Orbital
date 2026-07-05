// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ScopeChip unit tests — Task C1 (Spec 012 §2c).
 *
 * Covers (per the plan's Vitest list):
 *  - chip label for each scope mode (all / selected / off);
 *  - popover selection → PUT payload (master toggle + project multi-select),
 *    including optimistic update and revert-with-notice on PUT failure;
 *  - non-scratch project renders nothing;
 *  - per-session behavior: GET re-fires when sessionId changes.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Project, SessionScope } from '../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import ScopeChip from './ScopeChip';

afterEach(() => cleanup());

function makeProject(overrides: Partial<Project> = {}): Project {
  return {
    project_id: 'proj-x',
    name: 'Some Project',
    workspace: '/tmp/x',
    model: 'm',
    api_key: '',
    base_url: null,
    autonomy: 'hands_off',
    instructions: '',
    ...overrides,
  };
}

const SCRATCH = makeProject({
  project_id: 'scratch-1',
  name: 'Quick Tasks',
  is_scratch: true,
});

const PROJECTS: Project[] = [
  SCRATCH,
  makeProject({ project_id: 'proj-a', name: 'Alpha' }),
  makeProject({ project_id: 'proj-b', name: 'Beta' }),
  makeProject({ project_id: 'proj-c', name: 'Gamma' }),
];

const SCOPE_URL = '/api/v2/agents/scratch-1/sessions/sess-1/scope';

function mockGet(scope: SessionScope) {
  apiMock.mockImplementation(async (_path: string, options?: RequestInit) => {
    if (options?.method === 'PUT') return JSON.parse(options.body as string);
    return scope;
  });
}

function renderChip(props: Partial<React.ComponentProps<typeof ScopeChip>> = {}) {
  return render(
    <ScopeChip
      project={SCRATCH}
      projects={PROJECTS}
      sessionId="sess-1"
      {...props}
    />,
  );
}

/** Last PUT call's [url, body] from the api mock, or undefined. */
function lastPut(): { url: string; body: SessionScope } | undefined {
  const call = [...apiMock.mock.calls]
    .reverse()
    .find(([, opts]) => (opts as RequestInit | undefined)?.method === 'PUT');
  if (!call) return undefined;
  return {
    url: call[0] as string,
    body: JSON.parse((call[1] as RequestInit).body as string) as SessionScope,
  };
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('ScopeChip — render gating', () => {
  it('renders nothing for a non-scratch project', () => {
    apiMock.mockResolvedValue({ mode: 'all', selected_project_ids: [] });
    renderChip({ project: makeProject({ project_id: 'proj-a' }) });
    expect(screen.queryByTestId('scope-chip')).toBeNull();
    expect(apiMock).not.toHaveBeenCalled();
  });

  it('renders for the scratch project and fetches the session scope', async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    expect(screen.getByTestId('scope-chip')).toBeTruthy();
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith(SCOPE_URL),
    );
  });

  it('shows the default (all) label and a disabled trigger when no session yet', () => {
    renderChip({ sessionId: undefined });
    expect(screen.getByTestId('scope-chip-label').textContent).toBe(
      'Access: All projects',
    );
    expect(
      (screen.getByTestId('scope-chip-trigger') as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(apiMock).not.toHaveBeenCalled();
  });
});

describe('ScopeChip — label per mode', () => {
  it("labels mode 'all' as All projects", async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: All projects',
      ),
    );
  });

  it("labels mode 'selected' with the count (plural)", async () => {
    mockGet({ mode: 'selected', selected_project_ids: ['proj-a', 'proj-b'] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: 2 projects',
      ),
    );
  });

  it("labels mode 'selected' with the count (singular)", async () => {
    mockGet({ mode: 'selected', selected_project_ids: ['proj-a'] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: 1 project',
      ),
    );
  });

  it("labels mode 'off' as This workspace", async () => {
    mockGet({ mode: 'off', selected_project_ids: [] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: This workspace',
      ),
    );
  });

  it('falls back to the default (all) label when the GET fails (new session)', async () => {
    apiMock.mockRejectedValue(new Error('404'));
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(screen.getByTestId('scope-chip-label').textContent).toBe(
      'Access: All projects',
    );
  });
});

describe('ScopeChip — popover contents', () => {
  it('lists every project except the scratch project itself, plus the footnote', async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    const popover = screen.getByTestId('scope-chip-popover');
    expect(popover).toBeTruthy();

    expect(screen.getByTestId('scope-chip-project-proj-a')).toBeTruthy();
    expect(screen.getByTestId('scope-chip-project-proj-b')).toBeTruthy();
    expect(screen.getByTestId('scope-chip-project-proj-c')).toBeTruthy();
    expect(screen.queryByTestId('scope-chip-project-scratch-1')).toBeNull();

    expect(popover.textContent).toContain(
      'Writes always stay in the Quick Tasks workspace.',
    );
  });

  it("checks the master toggle and every project row under mode 'all'", async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    expect(
      (screen.getByTestId('scope-chip-master-toggle') as HTMLInputElement).checked,
    ).toBe(true);
    for (const id of ['proj-a', 'proj-b', 'proj-c']) {
      expect(
        (screen.getByTestId(`scope-chip-project-${id}`) as HTMLInputElement).checked,
      ).toBe(true);
    }
  });
});

describe('ScopeChip — selection → PUT payload (optimistic)', () => {
  it("master toggle OFF from 'all' with no selection PUTs mode 'off'", async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-master-toggle'));
    });

    expect(lastPut()).toEqual({
      url: SCOPE_URL,
      body: { mode: 'off', selected_project_ids: [] },
    });
    // Optimistic: label already reflects the new mode.
    expect(screen.getByTestId('scope-chip-label').textContent).toBe(
      'Access: This workspace',
    );
  });

  it("master toggle OFF from 'all' with a remembered selection PUTs mode 'selected'", async () => {
    mockGet({ mode: 'all', selected_project_ids: ['proj-b'] });
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-master-toggle'));
    });

    expect(lastPut()?.body).toEqual({
      mode: 'selected',
      selected_project_ids: ['proj-b'],
    });
  });

  it("master toggle ON PUTs mode 'all'", async () => {
    mockGet({ mode: 'off', selected_project_ids: [] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: This workspace',
      ),
    );

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-master-toggle'));
    });

    expect(lastPut()?.body).toEqual({ mode: 'all', selected_project_ids: [] });
    expect(screen.getByTestId('scope-chip-label').textContent).toBe(
      'Access: All projects',
    );
  });

  it("unchecking one project under 'all' narrows to 'selected' with the rest", async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    renderChip();
    await waitFor(() => expect(apiMock).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-project-proj-b'));
    });

    expect(lastPut()?.body).toEqual({
      mode: 'selected',
      selected_project_ids: ['proj-a', 'proj-c'],
    });
    expect(screen.getByTestId('scope-chip-label').textContent).toBe(
      'Access: 2 projects',
    );
  });

  it("checking a project under 'selected' adds it to the payload", async () => {
    mockGet({ mode: 'selected', selected_project_ids: ['proj-a'] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: 1 project',
      ),
    );

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-project-proj-c'));
    });

    expect(lastPut()?.body).toEqual({
      mode: 'selected',
      selected_project_ids: ['proj-a', 'proj-c'],
    });
  });

  it("unchecking the last selected project PUTs mode 'off'", async () => {
    mockGet({ mode: 'selected', selected_project_ids: ['proj-a'] });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: 1 project',
      ),
    );

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-project-proj-a'));
    });

    expect(lastPut()?.body).toEqual({ mode: 'off', selected_project_ids: [] });
  });
});

describe('ScopeChip — PUT failure: revert + error notice', () => {
  it('reverts the optimistic label and shows an alert when the PUT rejects', async () => {
    apiMock.mockImplementation(async (_path: string, options?: RequestInit) => {
      if (options?.method === 'PUT') throw new Error('409');
      return { mode: 'all', selected_project_ids: [] } satisfies SessionScope;
    });
    renderChip();
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: All projects',
      ),
    );

    fireEvent.click(screen.getByTestId('scope-chip-trigger'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('scope-chip-master-toggle'));
    });

    // Reverted to the pre-toggle scope…
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: All projects',
      ),
    );
    // …with a visible error notice.
    const alert = screen.getByTestId('scope-chip-error');
    expect(alert.getAttribute('role')).toBe('alert');
    expect(alert.textContent).toBe("Couldn't update access scope");
  });
});

describe('ScopeChip — per-session re-fetch', () => {
  it('re-reads the scope when sessionId changes', async () => {
    mockGet({ mode: 'all', selected_project_ids: [] });
    const { rerender } = renderChip({ sessionId: 'sess-1' });
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith(SCOPE_URL));

    apiMock.mockClear();
    mockGet({ mode: 'off', selected_project_ids: [] });
    rerender(
      <ScopeChip project={SCRATCH} projects={PROJECTS} sessionId="sess-2" />,
    );

    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith(
        '/api/v2/agents/scratch-1/sessions/sess-2/scope',
      ),
    );
    await waitFor(() =>
      expect(screen.getByTestId('scope-chip-label').textContent).toBe(
        'Access: This workspace',
      ),
    );
  });
});

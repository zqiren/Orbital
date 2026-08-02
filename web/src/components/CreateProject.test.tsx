// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * CreateProject unit tests (backlog #25 — simplified New Project modal).
 *
 * Covers the Vitest scope from the task brief:
 *  - name auto-derivation state machine: derives from the workspace
 *    basename, a manual name edit stops auto-fill, and a later workspace
 *    change does not overwrite an already-manually-edited name;
 *  - "Advanced options" is collapsed by default and reveals Agent
 *    Name/Instructions/Autonomy/Budget on toggle;
 *  - a 409 (agent_name collision) surfaces inline on the name field, without
 *    auto-suffixing the name, distinct from other submit errors; a later
 *    workspace change clears that stale error rather than letting it survive
 *    the name re-deriving to a new value;
 *  - the inline folder picker is not a nested <form>: pressing Enter in its
 *    new-folder or manual-path inputs fires the picker's own action (mkdir /
 *    browse) and must NOT also submit (create) the project.
 *
 * The api client is mocked (LLMProviderSettings's wizard-mode fetches, plus
 * the platform browse/folders/mkdir endpoints for the inline-picker tests) —
 * no network. Most tests exercise the plain workspace text input rather than
 * expanding the inline picker: both paths call the same `applyWorkspace`
 * logic in CreateProject, so this keeps them decoupled from
 * FolderBrowserPanel's own network calls (covered separately in
 * FolderBrowserPanel.test.tsx). The inline-picker describe block below is the
 * exception — it specifically verifies the two components compose safely.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

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

import CreateProject from './CreateProject';

afterEach(() => cleanup());

beforeEach(() => {
  apiFn.mockReset();
  // Quiet the LLM wizard-mode warning by default (api_key configured) so it
  // doesn't clutter unrelated assertions; providers registry is unused by
  // wizard mode but LLMProviderSettings fetches it unconditionally. Also
  // stub the platform browse/folders/mkdir endpoints the inline picker calls
  // once expanded (most tests never expand it, but a shared beforeEach is
  // simpler than a second setup path for the few that do).
  apiFn.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
    if (path === '/api/v2/settings') {
      return { llm: { api_key_set: true, api_key_masked: '', base_url: null, model: 'gpt-4', sdk: 'openai', provider: 'openai' } };
    }
    if (path === '/api/v2/providers') return {};
    if (typeof path === 'string' && path.startsWith('/api/v2/platform/browse')) {
      return { path: '/home/user', parent: '/home', display_name: 'user', entries: [] };
    }
    if (path === '/api/v2/platform/folders') return { status: 'ok', folders: [] };
    if (path === '/api/v2/platform/mkdir' && init?.method === 'POST') {
      const body = JSON.parse(init.body || '{}');
      return { status: 'ok', path: `${body.parent}/${body.name}` };
    }
    return {};
  });
});

function workspaceInput() {
  return screen.getByPlaceholderText('Select a folder or type a path...') as HTMLInputElement;
}

function nameInput() {
  return screen.getByPlaceholderText('e.g., Refactor Auth Module') as HTMLInputElement;
}

describe('CreateProject — name auto-derivation', () => {
  it('derives the project name from the workspace basename', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/my-app' } });
    expect(nameInput().value).toBe('my-app');
  });

  it('stops auto-fill once the name is manually edited', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/my-app' } });
    expect(nameInput().value).toBe('my-app');

    fireEvent.change(nameInput(), { target: { value: 'Custom Name' } });
    expect(nameInput().value).toBe('Custom Name');
  });

  it('does not overwrite a manually-edited name on a later workspace change', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/my-app' } });
    fireEvent.change(nameInput(), { target: { value: 'Custom Name' } });

    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/another-repo' } });
    expect(nameInput().value).toBe('Custom Name');
  });
});

describe('CreateProject — Advanced options disclosure', () => {
  it('is collapsed by default (Agent Name/Instructions/Autonomy/Budget hidden)', () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.getByText('Advanced options')).toBeTruthy();
    expect(screen.queryByPlaceholderText('e.g., CodeBot')).toBeNull();
    expect(screen.queryByText('Autonomy Level')).toBeNull();
    expect(screen.queryByText('Budget Limit (USD)')).toBeNull();
  });

  it('reveals the advanced fields on toggle', () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.click(screen.getByText('Advanced options'));
    expect(screen.getByPlaceholderText('e.g., CodeBot')).toBeTruthy();
    expect(screen.getByText('Autonomy Level')).toBeTruthy();
    expect(screen.getByText('Budget Limit (USD)')).toBeTruthy();
  });
});

describe('CreateProject — submit error handling', () => {
  function fillValidForm() {
    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/my-app' } });
  }

  it('surfaces a 409 agent-name conflict inline on the name field, without auto-suffixing', async () => {
    const onSubmit = vi.fn().mockRejectedValue(
      new MockApiError(409, "agent_name 'my-app' already in use"),
    );
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    fillValidForm();
    fireEvent.click(screen.getByText('Deploy Agent'));

    await waitFor(() => {
      expect(screen.getByText("agent_name 'my-app' already in use")).toBeTruthy();
    });
    // Never auto-suffixed — the name field keeps exactly what was submitted.
    expect(nameInput().value).toBe('my-app');

    // A later workspace change must clear the stale 409 message rather than
    // let it survive alongside the name re-deriving to a new value.
    fireEvent.change(workspaceInput(), { target: { value: '/Users/alice/another-repo' } });
    expect(screen.queryByText("agent_name 'my-app' already in use")).toBeNull();
  });

  it('shows a generic fallback for non-409 errors', async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('network down'));
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    fillValidForm();
    fireEvent.click(screen.getByText('Deploy Agent'));

    await waitFor(() => {
      expect(screen.getByText("Couldn't create the project. Please try again.")).toBeTruthy();
    });
  });

  it('calls onSubmit without model/api_key placeholders on a valid submit', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    fillValidForm();
    fireEvent.click(screen.getByText('Deploy Agent'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).not.toHaveProperty('model');
    expect(payload).not.toHaveProperty('api_key');
    expect(payload.name).toBe('my-app');
    expect(payload.workspace).toBe('/Users/alice/my-app');
  });
});

describe('CreateProject — inline folder picker composes safely with the outer form', () => {
  async function expandPicker() {
    fireEvent.click(screen.getByText('Browse'));
    // Wait for the picker's own mount-time browse() to resolve and
    // setCurrentPath to flush, so "New folder" (disabled until a currentPath
    // is known) is actually clickable.
    await waitFor(() => {
      const btn = screen.getByText('New folder').closest('button') as HTMLButtonElement;
      expect(btn.disabled).toBe(false);
    });
  }

  it('Enter in the new-folder input creates the folder but does not submit the project', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);

    await expandPicker();
    fireEvent.click(screen.getByText('New folder'));
    const folderNameInput = screen.getByPlaceholderText('Folder name');
    fireEvent.change(folderNameInput, { target: { value: 'my-app' } });
    fireEvent.keyDown(folderNameInput, { key: 'Enter' });

    await waitFor(() => {
      expect(apiFn).toHaveBeenCalledWith('/api/v2/platform/mkdir', {
        method: 'POST',
        body: JSON.stringify({ parent: '/home/user', name: 'my-app' }),
      });
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Enter in the manual-path input navigates but does not submit the project', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);

    await expandPicker();
    const manualInput = screen.getByPlaceholderText('Type a path and press Enter...');
    fireEvent.change(manualInput, { target: { value: '/some/other/path' } });
    fireEvent.keyDown(manualInput, { key: 'Enter' });

    await waitFor(() => {
      expect(apiFn.mock.calls.some(
        (c) => typeof c[0] === 'string' && c[0].includes(encodeURIComponent('/some/other/path')),
      )).toBe(true);
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe('CreateProject — modal a11y (backlog #26c)', () => {
  function dialog() {
    return screen.getByRole('dialog');
  }

  it('exposes dialog semantics labelled by its own title', () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    const d = dialog();
    expect(d.getAttribute('aria-modal')).toBe('true');
    const labelledBy = d.getAttribute('aria-labelledby');
    expect(labelledBy).toBeTruthy();
    expect(document.getElementById(labelledBy as string)?.textContent).toBe('New Project');
  });

  it('closes on Escape', () => {
    const onCancel = vi.fn();
    render(<CreateProject onSubmit={vi.fn()} onCancel={onCancel} />);
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('ignores non-Escape keys', () => {
    const onCancel = vi.fn();
    render(<CreateProject onSubmit={vi.fn()} onCancel={onCancel} />);
    fireEvent.keyDown(document.body, { key: 'Enter' });
    fireEvent.keyDown(document.body, { key: 'a' });
    expect(onCancel).not.toHaveBeenCalled();
  });

  it('does NOT close when Escape is raised inside the embedded folder picker', async () => {
    // The picker's new-folder input handles Escape itself (cancel the inline
    // editor) and neither stops propagation nor preventDefaults. If the modal
    // also acted on it, one keypress would discard the whole half-filled form.
    const onCancel = vi.fn();
    render(<CreateProject onSubmit={vi.fn()} onCancel={onCancel} />);

    fireEvent.click(screen.getByText('Browse'));
    await waitFor(() => {
      const btn = screen.getByText('New folder').closest('button') as HTMLButtonElement;
      expect(btn.disabled).toBe(false);
    });
    fireEvent.click(screen.getByText('New folder'));
    const folderNameInput = screen.getByPlaceholderText('Folder name');

    fireEvent.keyDown(folderNameInput, { key: 'Escape' });

    // The picker closed its inline editor; the modal stayed open.
    expect(onCancel).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Folder name')).toBeNull();
    });
    expect(screen.getByRole('dialog')).toBeTruthy();
  });

  it('autofocuses the workspace field with preventScroll, deferred by two frames (WKWebView-safe)', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    const ws = workspaceInput();

    // Not focused synchronously on mount. React's `autoFocus` prop would have
    // landed focus by now, mid `animate-slide-up` — precisely the WKWebView
    // scroll-chase this defers around. Checking activeElement (not a spy) is
    // what makes this meaningful: a spy attached after render cannot observe
    // a focus that already happened during it.
    expect(document.activeElement).not.toBe(ws);

    const focusSpy = vi.spyOn(ws, 'focus');
    await waitFor(() => expect(focusSpy).toHaveBeenCalled());
    expect(focusSpy.mock.calls[0][0]).toEqual({ preventScroll: true });
    expect(document.activeElement).toBe(ws);
  });

  it('traps Tab within the dialog (Shift+Tab from the first focusable wraps to the last)', () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    const d = dialog();
    const focusables = Array.from(
      d.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    expect(focusables.length).toBeGreaterThan(1);
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    first.focus();
    fireEvent.keyDown(first, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(last);

    fireEvent.keyDown(last, { key: 'Tab' });
    expect(document.activeElement).toBe(first);
  });
});

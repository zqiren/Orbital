// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * CreateProject unit tests (Codex-style Create Project dialog).
 *
 * The contract:
 *  - Project name comes first, is typed by the user, and is autofocused;
 *  - the folder is a separate field with NO path text input in the default
 *    view — it is chosen through the embedded folder browser (pick an existing
 *    folder, or create a new one under the browsed path). Picking a folder
 *    never writes the name, and the new-folder input is prefilled from it;
 *  - "Create project" stays disabled until there is both a name and a folder;
 *  - "Advanced options" is collapsed by default and reveals Agent
 *    Name/Instructions/Autonomy/Budget on toggle;
 *  - a 409 (agent_name collision) surfaces inline on the name field, without
 *    auto-suffixing, and clears when the name is edited;
 *  - the inline folder picker is not a nested <form>: pressing Enter in its
 *    new-folder or manual-path inputs fires the picker's own action (mkdir /
 *    browse) and must NOT also submit (create) the project.
 *
 * The api client is mocked (LLMProviderSettings's wizard-mode fetches, plus
 * the platform browse/folders/mkdir endpoints the picker calls) — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { flushSync } from 'react-dom';

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

function nameInput() {
  return screen.getByPlaceholderText('Copenhagen Trip') as HTMLInputElement;
}

function createButton() {
  // By role: the dialog title carries the same words.
  return screen.getByRole('button', { name: 'Create project' }) as HTMLButtonElement;
}

/** Open the embedded browser from whichever entry point is showing (the empty
 * "Choose a folder" card, or "Change" on an already-picked row) and wait for
 * its mount-time browse() to land, so "New folder" / "Use this folder" are
 * actually clickable. */
async function expandPicker() {
  fireEvent.click(screen.queryByText('Choose a folder') ?? screen.getByText('Change'));
  await waitFor(() => {
    const btn = screen.getByText('New folder').closest('button') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });
}

/** Pick the browsed directory (mock: /home/user) via "Use this folder". */
async function pickFolder() {
  await expandPicker();
  fireEvent.click(screen.getByText('Use this folder'));
}

describe('CreateProject — name and folder are separate fields', () => {
  it('has no path text input in the default view — the folder is chosen by browsing', () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.getByText('Choose a folder')).toBeTruthy();
    expect(screen.queryByPlaceholderText('Select a folder or type a path...')).toBeNull();
    // The browser (which owns the manual path input) is collapsed until asked for.
    expect(screen.queryByPlaceholderText('Type a path and press Enter...')).toBeNull();
  });

  it('shows the picked folder as a row (name + path) and collapses the browser', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    await pickFolder();

    const row = screen.getByTestId('create-project-folder-row');
    expect(row.textContent).toContain('user');
    expect(row.textContent).toContain('/home/user');
    expect(screen.queryByText('Use this folder')).toBeNull();
    expect(screen.queryByText('Choose a folder')).toBeNull();
  });

  it('"Change" reopens the browser for an already-picked folder', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    await pickFolder();
    await expandPicker();
    expect(screen.getByText('Use this folder')).toBeTruthy();
  });

  it('never derives the project name from the folder', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    await pickFolder();
    expect(nameInput().value).toBe('');

    fireEvent.change(nameInput(), { target: { value: 'Copenhagen' } });
    await expandPicker();
    fireEvent.click(screen.getByText('Use this folder'));
    expect(nameInput().value).toBe('Copenhagen');
  });

  it('prefills the new-folder name from the project name', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(nameInput(), { target: { value: 'Copenhagen' } });
    await expandPicker();
    fireEvent.click(screen.getByText('New folder'));
    expect((screen.getByPlaceholderText('Folder name') as HTMLInputElement).value).toBe('Copenhagen');
  });

  it('keeps "Create project" disabled until there is both a name and a folder', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    expect(createButton().disabled).toBe(true);

    fireEvent.change(nameInput(), { target: { value: 'Copenhagen' } });
    expect(createButton().disabled).toBe(true);

    await pickFolder();
    expect(createButton().disabled).toBe(false);

    fireEvent.change(nameInput(), { target: { value: '   ' } });
    expect(createButton().disabled).toBe(true);
  });

  it('a submit that slips past the disabled button (implicit Enter) explains what is missing', () => {
    const onSubmit = vi.fn();
    const { container } = render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    fireEvent.submit(container.ownerDocument.getElementById('create-project-form') as HTMLFormElement);

    expect(screen.getByText('Give your project a name.')).toBeTruthy();
    expect(screen.getByText('Choose a folder for this project.')).toBeTruthy();
    expect(onSubmit).not.toHaveBeenCalled();
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
  async function fillValidForm() {
    fireEvent.change(nameInput(), { target: { value: 'my-app' } });
    await pickFolder();
  }

  it('surfaces a 409 agent-name conflict inline on the name field, without auto-suffixing', async () => {
    const onSubmit = vi.fn().mockRejectedValue(
      new MockApiError(409, "agent_name 'my-app' already in use"),
    );
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    await fillValidForm();
    fireEvent.click(createButton());

    await waitFor(() => {
      expect(screen.getByText("agent_name 'my-app' already in use")).toBeTruthy();
    });
    // Never auto-suffixed — the name field keeps exactly what was submitted.
    expect(nameInput().value).toBe('my-app');

    // The collision is about the NAME, so re-picking a folder must not clear
    // it (the name no longer changes with the folder) — editing the name does.
    await pickFolder();
    expect(screen.getByText("agent_name 'my-app' already in use")).toBeTruthy();
    fireEvent.change(nameInput(), { target: { value: 'my-app-2' } });
    expect(screen.queryByText("agent_name 'my-app' already in use")).toBeNull();
  });

  it('shows a generic fallback for non-409 errors', async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('network down'));
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    await fillValidForm();
    fireEvent.click(createButton());

    await waitFor(() => {
      expect(screen.getByText("Couldn't create the project. Please try again.")).toBeTruthy();
    });
  });

  it('calls onSubmit without model/api_key placeholders on a valid submit', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<CreateProject onSubmit={onSubmit} onCancel={vi.fn()} />);
    await fillValidForm();
    fireEvent.click(createButton());

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).not.toHaveProperty('model');
    expect(payload).not.toHaveProperty('api_key');
    expect(payload.name).toBe('my-app');
    expect(payload.workspace).toBe('/home/user');
  });
});

describe('CreateProject — inline folder picker composes safely with the outer form', () => {
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
    // The created folder becomes the project folder.
    await waitFor(() => {
      expect(screen.getByTestId('create-project-folder-row').textContent).toContain('/home/user/my-app');
    });
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
    expect(document.getElementById(labelledBy as string)?.textContent).toBe('Create project');
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

    await expandPicker();
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

  it('does NOT close on Escape in the picker when React has already unmounted the input by the time the window listener runs', async () => {
    // The fireEvent test above cannot see this. A browser runs a microtask
    // checkpoint between event listeners, and that is when React flushes the
    // Escape handler's update — so the new-folder input is DETACHED before the
    // event reaches `window`, and `picker.contains(event.target)` is false.
    // jsdom dispatches in one synchronous stack (no checkpoint between
    // listeners), so the flush is forced here from a document-level listener,
    // which sits between React's root listener and `window` on the bubble
    // path. Found in a real browser: one Escape cancelled the folder editor
    // AND discarded the whole form.
    const onCancel = vi.fn();
    render(<CreateProject onSubmit={vi.fn()} onCancel={onCancel} />);

    await expandPicker();
    fireEvent.click(screen.getByText('New folder'));
    const folderNameInput = screen.getByPlaceholderText('Folder name');

    const checkpoint = () => flushSync(() => {});
    document.addEventListener('keydown', checkpoint);
    try {
      fireEvent.keyDown(folderNameInput, { key: 'Escape' });
    } finally {
      document.removeEventListener('keydown', checkpoint);
    }

    // Precondition for the test to mean anything: the input really was gone.
    expect(folderNameInput.isConnected).toBe(false);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it('autofocuses the name field with preventScroll, deferred by two frames (WKWebView-safe)', async () => {
    render(<CreateProject onSubmit={vi.fn()} onCancel={vi.fn()} />);
    const name = nameInput();

    // Not focused synchronously on mount. React's `autoFocus` prop would have
    // landed focus by now, mid `animate-slide-up` — precisely the WKWebView
    // scroll-chase this defers around. Checking activeElement (not a spy) is
    // what makes this meaningful: a spy attached after render cannot observe
    // a focus that already happened during it.
    expect(document.activeElement).not.toBe(name);

    const focusSpy = vi.spyOn(name, 'focus');
    await waitFor(() => expect(focusSpy).toHaveBeenCalled());
    expect(focusSpy.mock.calls[0][0]).toEqual({ preventScroll: true });
    expect(document.activeElement).toBe(name);
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

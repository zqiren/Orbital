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
 *    auto-suffixing the name, distinct from other submit errors.
 *
 * The api client is mocked (LLMProviderSettings's wizard-mode fetches) — no
 * network. The folder browser is exercised via the plain workspace text
 * input rather than expanding the inline picker: both paths call the same
 * `applyWorkspace` logic in CreateProject, so this keeps these tests
 * decoupled from FolderBrowserPanel's own network calls (covered separately
 * in FolderBrowserPanel.test.tsx).
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
  // wizard mode but LLMProviderSettings fetches it unconditionally.
  apiFn.mockImplementation(async (path: string) => {
    if (path === '/api/v2/settings') {
      return { llm: { api_key_set: true, api_key_masked: '', base_url: null, model: 'gpt-4', sdk: 'openai', provider: 'openai' } };
    }
    if (path === '/api/v2/providers') return {};
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

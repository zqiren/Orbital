// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { LocaleProvider } from '../i18n/LocaleContext';

const api = vi.fn();
vi.mock('../config', () => ({
  api: (...args: unknown[]) => api(...args),
}));

type WsHandler = (event: unknown) => void;
const wsHandlers = new Map<string, Set<WsHandler>>();
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: (type: string, fn: WsHandler) => {
      if (!wsHandlers.has(type)) wsHandlers.set(type, new Set());
      wsHandlers.get(type)!.add(fn);
    },
    off: (type: string, fn: WsHandler) => {
      wsHandlers.get(type)?.delete(fn);
    },
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

import LoginCodeField from './LoginCodeField';

function emit(type: string, payload: Record<string, unknown>) {
  act(() => {
    wsHandlers.get(type)?.forEach(fn => fn({ type, ...payload }));
  });
}

function renderField(slug = 'antigravity') {
  return render(
    <LocaleProvider>
      <LoginCodeField slug={slug} />
    </LocaleProvider>,
  );
}

describe('LoginCodeField (spec 193)', () => {
  beforeEach(() => {
    api.mockReset();
    wsHandlers.clear();
  });
  afterEach(() => cleanup());

  it('shows nothing until the login asks for a code', () => {
    renderField();
    expect(screen.queryByTestId('login-code-antigravity')).toBeNull();
  });

  it('ignores another agent, and an update job', () => {
    renderField();
    emit('login.input_needed', { slug: 'codex', job_id: 'j1' });
    emit('login.input_needed', { slug: 'antigravity', job_id: 'j2', kind: 'update' });
    expect(screen.queryByTestId('login-code-antigravity')).toBeNull();
  });

  it('posts the pasted code to the job and then goes away', async () => {
    api.mockResolvedValueOnce({ status: 'sent' });
    renderField();
    emit('login.input_needed', { slug: 'antigravity', job_id: 'job42', kind: 'login' });
    const input = screen.getByTestId('login-code-input-antigravity') as HTMLInputElement;
    expect(input.placeholder).toBe('Paste the code');
    fireEvent.change(input, { target: { value: '  4/0AbCdEf  ' } });
    fireEvent.click(screen.getByTestId('login-code-submit-antigravity'));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    const [path, init] = api.mock.calls[0] as [string, RequestInit];
    expect(path).toBe('/api/v2/settings/sub-agents/antigravity/login-jobs/job42/input');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ value: '4/0AbCdEf' });
    await waitFor(() => expect(screen.queryByTestId('login-code-antigravity')).toBeNull());
  });

  it('Enter submits too', async () => {
    api.mockResolvedValueOnce({ status: 'sent' });
    renderField();
    emit('login.input_needed', { slug: 'antigravity', job_id: 'j3' });
    const input = screen.getByTestId('login-code-input-antigravity');
    fireEvent.change(input, { target: { value: 'code' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
  });

  it('keeps the field and shows the error when the daemon refuses', async () => {
    api.mockRejectedValueOnce(new Error('this login is not waiting for input'));
    renderField();
    emit('login.input_needed', { slug: 'antigravity', job_id: 'j4' });
    fireEvent.change(screen.getByTestId('login-code-input-antigravity'), { target: { value: 'x' } });
    fireEvent.click(screen.getByTestId('login-code-submit-antigravity'));
    expect(await screen.findByRole('alert')).toHaveTextContent('this login is not waiting for input');
    expect(screen.getByTestId('login-code-antigravity')).toBeTruthy();
  });

  it('a finished login hides the field', () => {
    renderField();
    emit('login.input_needed', { slug: 'antigravity', job_id: 'j5' });
    expect(screen.getByTestId('login-code-antigravity')).toBeTruthy();
    emit('login.failed', { slug: 'antigravity', job_id: 'j5', kind: 'login' });
    expect(screen.queryByTestId('login-code-antigravity')).toBeNull();
  });
});

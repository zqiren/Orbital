// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect, vi, beforeEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import {
  dismissBackgroundImport, getBackgroundImport, IDLE_IMPORT, importIndicator, importReducer,
  resetBackgroundImportForTests, retryBackgroundImport, startBackgroundImport,
} from './backgroundImport';

const REQ = { folders: ['/x/a', '/x/b'], groups: [{ id: 'g1', workspace: '/x/g' }] };
const EMPTY = { folders: [], groups: [] };

function deferred() {
  let resolve!: (v: unknown) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

beforeEach(() => {
  apiMock.mockReset();
  resetBackgroundImportForTests();
});

describe('importReducer', () => {
  it('start → pending, then settled → idle and counts the answer', () => {
    const p = importReducer(IDLE_IMPORT, { type: 'start', request: REQ });
    expect(p).toEqual({ status: 'pending', request: REQ, finished: 0 });
    expect(importReducer(p, { type: 'settled' })).toEqual({ status: 'idle', request: REQ, finished: 1 });
  });

  it('a failure keeps the request for Retry and counts the answer', () => {
    const p = importReducer(IDLE_IMPORT, { type: 'start', request: REQ });
    expect(importReducer(p, { type: 'failed' })).toEqual({ status: 'failed', request: REQ, finished: 1 });
  });

  it('a second start while one is pending is ignored', () => {
    const p = importReducer(IDLE_IMPORT, { type: 'start', request: REQ });
    expect(importReducer(p, { type: 'start', request: EMPTY })).toBe(p);
  });

  it('answers and dismiss outside their state change nothing', () => {
    expect(importReducer(IDLE_IMPORT, { type: 'settled' })).toBe(IDLE_IMPORT);
    expect(importReducer(IDLE_IMPORT, { type: 'failed' })).toBe(IDLE_IMPORT);
    expect(importReducer(IDLE_IMPORT, { type: 'dismiss' })).toBe(IDLE_IMPORT);
    const p = importReducer(IDLE_IMPORT, { type: 'start', request: REQ });
    expect(importReducer(p, { type: 'dismiss' })).toBe(p);
  });

  it('dismiss clears a failure without counting an answer', () => {
    const f = { status: 'failed' as const, request: REQ, finished: 1 };
    expect(importReducer(f, { type: 'dismiss' })).toEqual({ status: 'idle', request: REQ, finished: 1 });
  });
});

describe('importIndicator (what the sidebar shows)', () => {
  it('idle: only the count', () => {
    expect(importIndicator(IDLE_IMPORT)).toEqual({ spinner: false, failed: false });
  });
  it('pending: the spinner', () => {
    expect(importIndicator({ status: 'pending', request: REQ, finished: 0 })).toEqual({ spinner: true, failed: false });
  });
  it('failed: the notice, no spinner', () => {
    expect(importIndicator({ status: 'failed', request: REQ, finished: 1 })).toEqual({ spinner: false, failed: true });
  });
});

describe('the store', () => {
  it('a fresh page load is idle: no stale spinner after a reload', () => {
    expect(getBackgroundImport()).toEqual(IDLE_IMPORT);
  });

  it('start fires the request and returns before it answers; pending until it does', async () => {
    const d = deferred();
    apiMock.mockReturnValue(d.promise);
    const done = startBackgroundImport(REQ);
    expect(apiMock).toHaveBeenCalledWith('/api/v2/onboarding/import', { method: 'POST', body: JSON.stringify(REQ) });
    expect(getBackgroundImport().status).toBe('pending');
    d.resolve({ created: [], skipped: [] });
    await done;
    expect(getBackgroundImport()).toEqual({ status: 'idle', request: REQ, finished: 1 });
  });

  it('nothing to import: no request, stays idle', async () => {
    await startBackgroundImport(EMPTY);
    expect(apiMock).not.toHaveBeenCalled();
    expect(getBackgroundImport().status).toBe('idle');
  });

  it('one request at a time', () => {
    apiMock.mockReturnValue(new Promise(() => {}));
    void startBackgroundImport(REQ);
    void startBackgroundImport(REQ);
    expect(apiMock).toHaveBeenCalledTimes(1);
  });

  it('a failure → failed; Retry re-fires the SAME selection; success clears it', async () => {
    apiMock.mockRejectedValueOnce(new Error('daemon gone'));
    await startBackgroundImport(REQ);
    expect(getBackgroundImport().status).toBe('failed');
    apiMock.mockResolvedValueOnce({ created: [], skipped: [] });
    await retryBackgroundImport();
    expect(apiMock).toHaveBeenCalledTimes(2);
    expect(apiMock.mock.calls[1][1].body).toBe(JSON.stringify(REQ));
    expect(getBackgroundImport()).toEqual({ status: 'idle', request: REQ, finished: 2 });
  });

  it('Retry does nothing unless the last import failed', async () => {
    await retryBackgroundImport();
    expect(apiMock).not.toHaveBeenCalled();
  });

  it('dismiss hides the failure', async () => {
    apiMock.mockRejectedValueOnce(new Error('500'));
    await startBackgroundImport(REQ);
    dismissBackgroundImport();
    expect(getBackgroundImport().status).toBe('idle');
  });
});

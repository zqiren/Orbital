// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/** Spec 154 §5.1 — the one definition of "waiting on you" and "running". */
import { describe, it, expect } from 'vitest';
import { isRunning, isUnchecked, isWaiting, needsAction, laterOf, WINDOW_DAYS, type WaitingRow } from './waitingOnYou';

const NOW = Date.parse('2026-11-04T12:00:00Z');

function row(over: Partial<WaitingRow> = {}): WaitingRow {
  return {
    status: 'idle',
    worker_running: false,
    last_user_at: '2026-11-04T08:00:00Z',
    last_reply_at: '2026-11-04T08:05:00Z',
    checked_reply_at: null,
    ask: null,
    stop: null,
    ...over,
  };
}

describe('isRunning', () => {
  it('running, waiting, queued and new_session are running; idle and approval are not', () => {
    for (const status of ['running', 'waiting', 'queued', 'new_session'] as const) {
      expect(isRunning(row({ status })), status).toBe(true);
    }
    expect(isRunning(row())).toBe(false);
    expect(isRunning(row({ status: 'pending_approval' }))).toBe(false);
  });
  it('a resting manager with a working worker is running', () => {
    expect(isRunning(row({ worker_running: true }))).toBe(true);
  });
});

describe('isUnchecked', () => {
  it('agent spoke last, resting, never checked → unchecked', () => {
    expect(isUnchecked(row(), NOW)).toBe(true);
  });
  it('checked at that reply → no longer', () => {
    expect(isUnchecked(row({ checked_reply_at: '2026-11-04T08:05:00Z' }), NOW)).toBe(false);
  });
  it('a newer reply than the checked one → unchecked again', () => {
    expect(isUnchecked(row({ checked_reply_at: '2026-11-04T07:00:00Z' }), NOW)).toBe(true);
  });
  it('the user spoke last → not unchecked', () => {
    expect(isUnchecked(row({ last_user_at: '2026-11-04T09:00:00Z' }), NOW)).toBe(false);
  });
  it(`outside the ${WINDOW_DAYS}-day window → not unchecked`, () => {
    expect(isUnchecked(row({ last_reply_at: '2026-10-20T08:05:00Z', last_user_at: null }), NOW)).toBe(false);
  });
  it('a running row is not unchecked (its glyph is lit)', () => {
    expect(isUnchecked(row({ worker_running: true }), NOW)).toBe(false);
  });
});

describe('isWaiting / needsAction', () => {
  const ask = { id: 'a1', text: 'q?', due: null, created: '2026-11-04', overdue: false };
  it('an ask, an approval or an unchecked reply waits', () => {
    expect(isWaiting(row({ ask, last_reply_at: null }), NOW)).toBe(true);
    expect(isWaiting(row({ status: 'pending_approval', last_reply_at: null }), NOW)).toBe(true);
    expect(isWaiting(row(), NOW)).toBe(true);
    expect(isWaiting(row({ checked_reply_at: '2026-11-04T08:05:00Z' }), NOW)).toBe(false);
  });
  it('needs action: asks, approvals and stops the user must fix', () => {
    expect(needsAction(row({ ask }))).toBe(true);
    expect(needsAction(row({ status: 'pending_approval' }))).toBe(true);
    expect(needsAction(row({ stop: { cause: 'usage_limit', resets_at: null } }))).toBe(true);
    expect(needsAction(row({ stop: { cause: 'network_auth', resets_at: null } }))).toBe(true);
    expect(needsAction(row({ stop: { cause: 'interrupted', resets_at: null } }))).toBe(false);
    expect(needsAction(row())).toBe(false);
  });
});

describe('laterOf', () => {
  it('picks the later time and tolerates gaps', () => {
    expect(laterOf(null, null)).toBeNull();
    expect(laterOf('2026-11-01T00:00:00Z', null)).toBe('2026-11-01T00:00:00Z');
    expect(laterOf('2026-11-01T00:00:00Z', '2026-11-02T00:00:00Z')).toBe('2026-11-02T00:00:00Z');
  });
});

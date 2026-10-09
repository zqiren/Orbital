// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 108 — `hasUnseenReply`: a row is unread when the agent's last reply is
 * newer than the user's last message, the row is resting, and the reply is
 * newer than the one this device last showed.
 */
import { describe, it, expect } from 'vitest';
import { hasUnseenReply, seenKey } from './sessionUnread';
import type { SessionListEntry } from '../types';

function entry(overrides: Partial<SessionListEntry> = {}): SessionListEntry {
  return {
    session_id: 's1',
    session_uuid: 'u1',
    status: 'idle',
    last_terminal_event: null,
    last_activity_at: '2026-10-01T00:00:10Z',
    last_user_at: '2026-10-01T00:00:00Z',
    last_reply_at: '2026-10-01T00:00:10Z',
    ...overrides,
  };
}

describe('hasUnseenReply', () => {
  it('no reply → false', () => {
    expect(hasUnseenReply(entry({ last_reply_at: null }), undefined)).toBe(false);
  });

  it('absent fields (older daemon) → false', () => {
    const e = entry();
    delete e.last_reply_at;
    delete e.last_user_at;
    expect(hasUnseenReply(e, undefined)).toBe(false);
  });

  it('the user spoke after the reply → false', () => {
    expect(
      hasUnseenReply(entry({ last_user_at: '2026-10-01T00:00:20Z' }), undefined),
    ).toBe(false);
  });

  it('a reply with no user row at all (automation) counts', () => {
    expect(hasUnseenReply(entry({ last_user_at: null }), undefined)).toBe(true);
  });

  it('running / waiting / blocked / queued / starting rows → false (the glyph outranks the badge)', () => {
    for (const status of ['running', 'waiting', 'pending_approval', 'queued', 'new_session'] as const) {
      expect(hasUnseenReply(entry({ status }), undefined), status).toBe(false);
    }
  });

  it('a resting row whose worker is running → false (spec 102 lights the glyph)', () => {
    expect(hasUnseenReply(entry({ worker_running: true }), undefined)).toBe(false);
  });

  it('never seen → true', () => {
    expect(hasUnseenReply(entry(), undefined)).toBe(true);
  });

  it('reply newer than the seen marker → true', () => {
    expect(hasUnseenReply(entry(), '2026-10-01T00:00:05Z')).toBe(true);
  });

  it('reply equal to the seen marker → false', () => {
    expect(hasUnseenReply(entry(), '2026-10-01T00:00:10Z')).toBe(false);
  });

  it('reply older than the seen marker → false', () => {
    expect(hasUnseenReply(entry(), '2026-10-01T00:00:15Z')).toBe(false);
  });

  it('an unparseable reply timestamp is never unread', () => {
    expect(hasUnseenReply(entry({ last_reply_at: 'garbage' }), undefined)).toBe(false);
  });

  it('an error status row with a reply lights (the LLM-error row is a reply)', () => {
    // `error` is not a resting status today; the backend lists an errored
    // session as idle with the error on last_terminal_event, which is the
    // real shape. Assert that shape.
    expect(
      hasUnseenReply(
        entry({
          status: 'idle',
          last_terminal_event: { type: 'error', timestamp: '2026-10-01T00:00:10Z', details: 'x' },
        }),
        undefined,
      ),
    ).toBe(true);
  });
});

describe('seenKey', () => {
  it('prefers session_uuid and falls back to session_id', () => {
    expect(seenKey(entry())).toBe('u1');
    expect(seenKey(entry({ session_uuid: null }))).toBe('s1');
  });
});

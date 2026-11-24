// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 099 §3.1 / §3.3 — the pure rules behind the pin menu's usage line,
 * hover card and status dot (approved mockup, 2026-09-27).
 */

import { describe, expect, it } from 'vitest';
import type { QuotaSnapshot, QuotaWindow } from '../types';
import {
  LOW_LEFT_PCT,
  ageParts,
  formatResetTime,
  leftTone,
  pickShownWindow,
  relativeParts,
  statusDot,
  summarizeQuota,
  switchTarget,
  windowLeft,
} from './quotaDisplay';

const NOW = Date.parse('2026-09-27T19:00:00');
const inH = (h: number) => new Date(NOW + h * 3600_000).toISOString();

function snap(windows: Array<[QuotaWindow['kind'], number, string | null]>,
  extra: Partial<QuotaSnapshot> = {}): QuotaSnapshot {
  return {
    agent: 'claude-code',
    observed_at: new Date(NOW - 4 * 60_000).toISOString(),
    plan: null,
    limited: false,
    windows: windows.map(([kind, used_pct, resets_at]) => ({ kind, used_pct, resets_at })),
    ...extra,
  };
}

describe('windowLeft', () => {
  it('is 100 - used, rounded, clamped', () => {
    expect(windowLeft({ kind: 'five_hour', used_pct: 79, resets_at: inH(3) }, NOW)).toBe(21);
    expect(windowLeft({ kind: 'five_hour', used_pct: 100.4, resets_at: inH(3) }, NOW)).toBe(0);
    expect(windowLeft({ kind: 'five_hour', used_pct: 10.6, resets_at: null }, NOW)).toBe(89);
  });

  it('is void (null) once the window has reset since the reading', () => {
    expect(windowLeft({ kind: 'five_hour', used_pct: 79, resets_at: inH(-1) }, NOW)).toBeNull();
  });
});

describe('leftTone', () => {
  it('amber below 20% left, red at 0', () => {
    expect(LOW_LEFT_PCT).toBe(20);
    expect(leftTone(20)).toBe('ok');
    expect(leftTone(19)).toBe('low');
    expect(leftTone(0)).toBe('out');
  });
});

describe('pickShownWindow', () => {
  it('shows the 5-hour window by default', () => {
    const s = snap([['five_hour', 79, inH(3)], ['weekly', 8, inH(60)]]);
    expect(pickShownWindow(s, NOW)).toMatchObject({ kind: 'five_hour', left: 21 });
  });

  it('shows weekly when it has less left; a tie stays on 5h', () => {
    expect(pickShownWindow(snap([['five_hour', 36, inH(3)], ['weekly', 91, inH(60)]]), NOW))
      .toMatchObject({ kind: 'weekly', left: 9 });
    expect(pickShownWindow(snap([['five_hour', 50, inH(3)], ['weekly', 50, inH(60)]]), NOW))
      .toMatchObject({ kind: 'five_hour' });
  });

  it('never picks a per-model weekly window', () => {
    const s = snap([['five_hour', 10, inH(3)], ['weekly_opus', 99, inH(60)]]);
    expect(pickShownWindow(s, NOW)).toMatchObject({ kind: 'five_hour', left: 90 });
  });

  it('skips a window that has reset since', () => {
    const s = snap([['five_hour', 90, inH(-1)], ['weekly', 30, inH(60)]]);
    expect(pickShownWindow(s, NOW)).toMatchObject({ kind: 'weekly', left: 70 });
  });
});

describe('summarizeQuota', () => {
  it('absence is "none" — never "fine"', () => {
    expect(summarizeQuota(undefined, NOW)).toEqual({ kind: 'none' });
    expect(summarizeQuota(null, NOW)).toEqual({ kind: 'none' });
  });

  it('a window line with the other window alongside', () => {
    const s = summarizeQuota(snap([['five_hour', 88, inH(3)], ['weekly', 12, inH(60)]]), NOW);
    expect(s).toMatchObject({ kind: 'window', tone: 'low',
      shown: { kind: 'five_hour', left: 12 }, other: { kind: 'weekly', left: 88 } });
  });

  it('limited → limit reached with the tightest window reset time', () => {
    const s = summarizeQuota(snap([['five_hour', 100, inH(3)], ['weekly', 29, inH(60)]],
      { limited: true }), NOW);
    expect(s.kind).toBe('limited');
    if (s.kind === 'limited') {
      expect(s.resetsAt?.toISOString()).toBe(inH(3));
      expect(s.other).toMatchObject({ kind: 'weekly', left: 71 });
    }
  });

  it('0% left counts as limit reached even without the flag', () => {
    expect(summarizeQuota(snap([['five_hour', 100, inH(3)]]), NOW).kind).toBe('limited');
  });

  it('every window reset since the reading → stale, no number', () => {
    expect(summarizeQuota(snap([['five_hour', 90, inH(-2)], ['weekly', 10, inH(-1)]]), NOW))
      .toEqual({ kind: 'stale' });
    // A limit whose window has since reset is not presented as current.
    expect(summarizeQuota(snap([['five_hour', 100, inH(-1)]], { limited: true }), NOW))
      .toEqual({ kind: 'stale' });
  });

  it('a limit whose binding window has reset is lifted, not moved to another window', () => {
    // Live 2026-09-27: claude limited on the 5-hour window, read at 19:31;
    // at 22:04 that window had reset but weekly (11% used) had not. The limit
    // is over: show the remaining live window, never "Limit reached · Tue".
    const s = summarizeQuota(snap([['five_hour', 100, inH(-1)], ['weekly', 11, inH(40)]],
      { limited: true }), NOW);
    expect(s).toMatchObject({ kind: 'window', shown: { kind: 'weekly', left: 89 }, tone: 'ok' });
    expect(statusDot({ ready: true, running: false,
      quota: snap([['five_hour', 100, inH(-1)], ['weekly', 11, inH(40)]], { limited: true }),
      now: NOW })).toBeNull();
  });

  it('limited with no windows at all has no reset time', () => {
    expect(summarizeQuota(snap([], { limited: true }), NOW))
      .toEqual({ kind: 'limited', resetsAt: null, other: null });
  });
});

describe('statusDot', () => {
  const ok = snap([['five_hour', 10, inH(3)]]);
  const low = snap([['five_hour', 85, inH(3)]]);
  const out = snap([['five_hour', 100, inH(3)]], { limited: true });

  it('no dot when ready, idle and not low — the common case', () => {
    expect(statusDot({ ready: true, running: false, quota: ok, now: NOW })).toBeNull();
    expect(statusDot({ ready: true, running: false, quota: undefined, now: NOW })).toBeNull();
    // An older backend omits `ready`: that is "no info", not "needs login".
    expect(statusDot({ ready: undefined, running: false, quota: undefined, now: NOW })).toBeNull();
  });

  it('green pulse while running', () => {
    expect(statusDot({ ready: true, running: true, quota: ok, now: NOW }))
      .toEqual({ color: 'green', pulse: true });
  });

  it('amber when low, red when limited; they pulse while running', () => {
    expect(statusDot({ ready: true, running: false, quota: low, now: NOW }))
      .toEqual({ color: 'amber', pulse: false });
    expect(statusDot({ ready: true, running: true, quota: low, now: NOW }))
      .toEqual({ color: 'amber', pulse: true });
    expect(statusDot({ ready: true, running: false, quota: out, now: NOW }))
      .toEqual({ color: 'red', pulse: false });
    expect(statusDot({ ready: true, running: true, quota: out, now: NOW }))
      .toEqual({ color: 'red', pulse: true });
  });

  it('grey (not ready) beats everything', () => {
    expect(statusDot({ ready: false, running: true, quota: out, now: NOW }))
      .toEqual({ color: 'grey', pulse: false });
  });

  it('a stale reading shows no warning dot', () => {
    const stale = snap([['five_hour', 95, inH(-1)]]);
    expect(statusDot({ ready: true, running: false, quota: stale, now: NOW })).toBeNull();
  });
});

describe('time formatting', () => {
  it('same-day resets show HH:MM; later ones lead with the weekday', () => {
    const today = new Date('2026-09-27T22:00:00');
    const tue = new Date('2026-09-29T17:00:00');
    expect(formatResetTime(today, NOW, 'en')).toBe('22:00');
    expect(formatResetTime(tue, NOW, 'en')).toBe('Tue 17:00');
    expect(formatResetTime(tue, NOW, 'zh')).toBe('周二 17:00');
  });

  it('relative reset: minutes, hours, days', () => {
    expect(relativeParts(new Date(NOW + 45 * 60_000), NOW)).toEqual({ unit: 'minutes', n: 45 });
    expect(relativeParts(new Date(NOW + 3 * 3600_000), NOW)).toEqual({ unit: 'hours', n: 3 });
    expect(relativeParts(new Date(NOW + 50 * 3600_000), NOW)).toEqual({ unit: 'days', n: 2 });
    // The mockup's weekly reset: 45 h away reads "in 2d", not "in 45h".
    expect(relativeParts(new Date(NOW + 45 * 3600_000), NOW)).toEqual({ unit: 'days', n: 2 });
    expect(relativeParts(new Date(NOW - 1), NOW)).toBeNull();
  });

  it('age: just now, minutes, hours, days', () => {
    const at = (ms: number) => new Date(NOW - ms).toISOString();
    expect(ageParts(at(30_000), NOW)).toEqual({ unit: 'now', n: 0 });
    expect(ageParts(at(4 * 60_000), NOW)).toEqual({ unit: 'minutes', n: 4 });
    expect(ageParts(at(5 * 3600_000), NOW)).toEqual({ unit: 'hours', n: 5 });
    expect(ageParts(at(3 * 86400_000), NOW)).toEqual({ unit: 'days', n: 3 });
    expect(ageParts('garbage', NOW)).toBeNull();
  });
});

// Spec 197: one limited fact per agent — `limited` plus a top-level
// `resets_at` — for every agent, not only claude-code and codex.
describe('summarizeQuota — the limit fact (spec 197)', () => {
  it('a limit-only entry with a future reset is limited until then', () => {
    const s = summarizeQuota(snap([], { agent: 'kimi-code', limited: true, resets_at: inH(3) }), NOW);
    expect(s.kind).toBe('limited');
    expect(s.kind === 'limited' && s.resetsAt?.toISOString()).toBe(inH(3));
  });

  it('a limit with an unknown reset still reads as limited, without a time', () => {
    const s = summarizeQuota(snap([], { limited: true, resets_at: null }), NOW);
    expect(s).toEqual({ kind: 'limited', resetsAt: null, other: null });
  });

  it('a passed reset is void', () => {
    expect(summarizeQuota(snap([], { limited: true, resets_at: inH(-1) }), NOW).kind).toBe('none');
    const withWindows = snap([['five_hour', 40, inH(2)]], { limited: true, resets_at: inH(-1) });
    expect(summarizeQuota(withWindows, NOW).kind).toBe('window');
  });

  it('the entry reset wins over the windows for the time', () => {
    const s = summarizeQuota(snap([['five_hour', 100, inH(1)], ['weekly', 30, inH(48)]],
      { limited: true, resets_at: inH(2) }), NOW);
    expect(s.kind === 'limited' && s.resetsAt?.toISOString()).toBe(inH(2));
  });

  it('the red dot for any agent', () => {
    const dot = statusDot({ ready: true, running: false, now: NOW,
      quota: snap([], { agent: 'pi', limited: true, resets_at: inH(1) }) });
    expect(dot).toEqual({ color: 'red', pulse: false });
  });
});

describe('switchTarget (spec 197)', () => {
  const avail = (entries: Record<string, Partial<QuotaSnapshot> | null>, ready: Record<string, boolean> = {}) =>
    Object.fromEntries(Object.entries(entries).map(([slug, q]) => [slug, {
      slug, name: slug, ready: ready[slug] ?? true,
      ...(q ? { quota: snap([], q) } : {}),
    }]));

  it('the first ready agent in list order that is not limited and not the current one', () => {
    const a = avail({ codex: { limited: true, resets_at: inH(2) }, 'claude-code': null, pi: null });
    expect(switchTarget(['codex', 'claude-code', 'pi'], 'codex', a, NOW)).toBe('claude-code');
  });

  it('skips a limited or not-ready agent', () => {
    const a = avail({ codex: { limited: true, resets_at: inH(2) },
      'claude-code': { limited: true, resets_at: null }, pi: null, kimi: null }, { pi: false });
    expect(switchTarget(['codex', 'claude-code', 'pi', 'kimi'], 'codex', a, NOW)).toBe('kimi');
  });

  it('none when every other agent is limited', () => {
    const a = avail({ codex: { limited: true, resets_at: inH(2) }, 'claude-code': { limited: true, resets_at: inH(1) } });
    expect(switchTarget(['codex', 'claude-code'], 'codex', a, NOW)).toBeNull();
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import type { TVars } from '../i18n/useT';
import {
  cronForDraft,
  describeCron,
  draftFromCron,
  emptyScheduleDraft,
  humanForCron,
  timezoneOptions,
  type ScheduleDraft,
} from './scheduleFormat';

const en = (k: StringKey, v?: TVars) => translate('en', k, v);
const zh = (k: StringKey, v?: TVars) => translate('zh', k, v);

function draft(overrides: Partial<ScheduleDraft> = {}): ScheduleDraft {
  return { ...emptyScheduleDraft('UTC'), ...overrides };
}

describe('cronForDraft', () => {
  it('builds each preset', () => {
    expect(cronForDraft(draft({ preset: 'daily', time: '07:05' }))).toBe('5 7 * * *');
    expect(cronForDraft(draft({ preset: 'weekdays', time: '18:30' }))).toBe('30 18 * * 1-5');
    expect(cronForDraft(draft({ preset: 'weekly', time: '09:00', weekday: 3 }))).toBe(
      '0 9 * * 3',
    );
    expect(cronForDraft(draft({ preset: 'monthly', time: '00:15', dayOfMonth: 12 }))).toBe(
      '15 0 12 * *',
    );
    expect(cronForDraft(draft({ preset: 'hourly', minute: 45 }))).toBe('45 * * * *');
  });

  it('passes a custom expression through verbatim', () => {
    expect(cronForDraft(draft({ preset: 'custom', cron: '  */5 * * * *  ' }))).toBe(
      '*/5 * * * *',
    );
  });
});

describe('describeCron', () => {
  it('phrases the preset shapes from the catalog, in the reader locale', () => {
    expect(describeCron('0 9 * * *', en)).toBe('Every day at 09:00');
    expect(describeCron('0 9 * * *', zh)).toBe('每天 09:00');
    expect(describeCron('30 18 * * 1-5', en)).toBe('Weekdays at 18:30');
    expect(describeCron('0 17 * * 5', en)).toBe('Every Friday at 17:00');
    expect(describeCron('0 17 * * 5', zh)).toBe('每周五 17:00');
    expect(describeCron('15 0 12 * *', en)).toBe('Day 12 of every month at 00:15');
    expect(describeCron('45 * * * *', en)).toBe('Every hour at :45');
  });

  it('treats dow 7 as Sunday', () => {
    expect(describeCron('0 8 * * 7', en)).toBe(describeCron('0 8 * * 0', en));
  });

  it('returns null for anything it cannot phrase', () => {
    expect(describeCron('*/15 2 * * 3', en)).toBeNull();
    expect(describeCron('0 9 * 6 *', en)).toBeNull(); // month restriction
    expect(describeCron('0 9 1 * 1', en)).toBeNull(); // both dom and dow
    expect(describeCron('0 9 * *', en)).toBeNull(); // four fields
    expect(describeCron('', en)).toBeNull();
  });
});

describe('humanForCron', () => {
  it('stores the localized caption when a preset matches', () => {
    expect(humanForCron('0 9 * * 1-5', en)).toBe('Weekdays at 09:00');
    expect(humanForCron('0 9 * * 1-5', zh)).toBe('工作日 09:00');
  });

  it('falls back to the expression itself for a custom cron — never a broken sentence', () => {
    expect(humanForCron('*/15 2 * * 3', en)).toBe('*/15 2 * * 3');
    expect(humanForCron('*/15 2 * * 3', zh)).toBe('*/15 2 * * 3');
  });
});

describe('draftFromCron', () => {
  it('round-trips every preset back out of a stored cron', () => {
    for (const d of [
      draft({ preset: 'daily', time: '07:05' }),
      draft({ preset: 'weekdays', time: '18:30' }),
      draft({ preset: 'weekly', time: '09:00', weekday: 3 }),
      draft({ preset: 'monthly', time: '00:15', dayOfMonth: 12 }),
      draft({ preset: 'hourly', minute: 45 }),
    ]) {
      const cron = cronForDraft(d);
      const parsed = draftFromCron(cron, 'UTC');
      expect(parsed.preset).toBe(d.preset);
      expect(cronForDraft(parsed)).toBe(cron);
    }
  });

  it('drops an unrecognized expression into the custom escape hatch', () => {
    const parsed = draftFromCron('*/15 2 * * 3', 'Asia/Shanghai');
    expect(parsed.preset).toBe('custom');
    expect(parsed.cron).toBe('*/15 2 * * 3');
    expect(parsed.timezone).toBe('Asia/Shanghai');
  });
});

describe('timezoneOptions', () => {
  it('always offers the trigger’s stored zone, even an exotic one', () => {
    expect(timezoneOptions('Pacific/Chatham')).toContain('Pacific/Chatham');
    expect(timezoneOptions()).toContain('UTC');
  });

  it('has no duplicates', () => {
    const zones = timezoneOptions('UTC');
    expect(new Set(zones).size).toBe(zones.length);
  });
});

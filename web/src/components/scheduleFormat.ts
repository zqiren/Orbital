// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Cron ⇄ preset translation for the automation schedule widget.
 *
 * A trigger stores BOTH `schedule.cron` (what the scheduler evaluates) and
 * `schedule.human` (what every UI surface renders as the row label). The agent
 * writes `human` as free text; a form can't, so it derives it here.
 *
 * The derivation goes through the i18n catalog — `describeCron` returns a
 * localized sentence built from a template with placeholders, never English
 * fragments concatenated together. That means a Chinese user's form produces a
 * Chinese caption, and a cron that matches a known preset can be re-described
 * live in whatever locale is reading it (see `AutomationsList`), rather than
 * being frozen in the language it was created in.
 *
 * A cron that matches no preset (the "custom cron" escape hatch) describes as
 * `null`; callers fall back to the cron expression itself, which is honest and
 * locale-neutral — never a broken half-sentence.
 *
 * No runtime cron library: this is a deliberate five-field pattern match over
 * the shapes the widget itself can produce. Full cron validation stays on the
 * server (croniter), whose 400 the form surfaces inline.
 */

import type { StringKey } from '../i18n/strings';
import type { TVars } from '../i18n/useT';

export type Translate = (key: StringKey, vars?: TVars) => string;

export type SchedulePreset =
  | 'daily'
  | 'weekdays'
  | 'weekly'
  | 'hourly'
  | 'monthly'
  | 'custom';

export const SCHEDULE_PRESETS: SchedulePreset[] = [
  'daily',
  'weekdays',
  'weekly',
  'monthly',
  'hourly',
  'custom',
];

export const PRESET_LABEL_KEYS: Record<SchedulePreset, StringKey> = {
  daily: 'trigger.sched.preset.daily',
  weekdays: 'trigger.sched.preset.weekdays',
  weekly: 'trigger.sched.preset.weekly',
  monthly: 'trigger.sched.preset.monthly',
  hourly: 'trigger.sched.preset.hourly',
  custom: 'trigger.sched.preset.custom',
};

/** Sunday-first, matching cron's 0-6. */
const DOW_KEYS: readonly StringKey[] = [
  'trigger.sched.dow.0',
  'trigger.sched.dow.1',
  'trigger.sched.dow.2',
  'trigger.sched.dow.3',
  'trigger.sched.dow.4',
  'trigger.sched.dow.5',
  'trigger.sched.dow.6',
];

export function weekdayName(dow: number, t: Translate): string {
  return t(DOW_KEYS[((dow % 7) + 7) % 7]);
}

/** The form's schedule state. `cron` is authoritative only for 'custom'. */
export interface ScheduleDraft {
  preset: SchedulePreset;
  /** "HH:MM", the value of a native <input type="time">. */
  time: string;
  /** 0 = Sunday, for the 'weekly' preset. */
  weekday: number;
  /** 1-28 (never 29-31: those silently skip months), for 'monthly'. */
  dayOfMonth: number;
  /** Minute past the hour, for 'hourly'. */
  minute: number;
  /** Raw expression, used when preset === 'custom'. */
  cron: string;
  timezone: string;
}

export const DEFAULT_TIMEZONE = 'UTC';

export function browserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_TIMEZONE;
  } catch {
    return DEFAULT_TIMEZONE;
  }
}

export function emptyScheduleDraft(timezone = browserTimezone()): ScheduleDraft {
  return {
    preset: 'daily',
    time: '09:00',
    weekday: 1,
    dayOfMonth: 1,
    minute: 0,
    cron: '0 9 * * *',
    timezone,
  };
}

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

function isNum(field: string): boolean {
  return /^\d+$/.test(field);
}

function splitTime(time: string): { hour: number; minute: number } {
  const [h, m] = time.split(':');
  const hour = Number(h);
  const minute = Number(m);
  return {
    hour: Number.isFinite(hour) ? hour : 0,
    minute: Number.isFinite(minute) ? minute : 0,
  };
}

/** Build the cron expression a draft describes. */
export function cronForDraft(draft: ScheduleDraft): string {
  const { hour, minute } = splitTime(draft.time);
  switch (draft.preset) {
    case 'daily':
      return `${minute} ${hour} * * *`;
    case 'weekdays':
      return `${minute} ${hour} * * 1-5`;
    case 'weekly':
      return `${minute} ${hour} * * ${draft.weekday}`;
    case 'monthly':
      return `${minute} ${hour} ${draft.dayOfMonth} * *`;
    case 'hourly':
      return `${draft.minute} * * * *`;
    case 'custom':
      return draft.cron.trim();
  }
}

/**
 * Localized caption for a cron expression, or null when it matches no preset
 * the widget knows how to phrase.
 */
export function describeCron(cron: string, t: Translate): string | null {
  const fields = cron.trim().split(/\s+/);
  if (fields.length !== 5) return null;
  const [min, hour, dom, mon, rawDow] = fields;
  if (mon !== '*') return null;
  // cron accepts 7 for Sunday; normalize so the label lookup stays in 0-6.
  const dow = rawDow === '7' ? '0' : rawDow;
  if (!isNum(min)) return null;

  if (hour === '*') {
    if (dom !== '*' || dow !== '*') return null;
    return t('trigger.sched.hourly', { minute: pad2(Number(min)) });
  }
  if (!isNum(hour)) return null;
  const time = `${pad2(Number(hour))}:${pad2(Number(min))}`;

  if (dom === '*' && dow === '*') return t('trigger.sched.daily', { time });
  if (dom === '*' && dow === '1-5') return t('trigger.sched.weekdays', { time });
  if (dom === '*' && isNum(dow)) {
    return t('trigger.sched.weekly', { day: weekdayName(Number(dow), t), time });
  }
  if (isNum(dom) && dow === '*') {
    return t('trigger.sched.monthly', { n: Number(dom), time });
  }
  return null;
}

/**
 * What to store in `schedule.human`: the localized preset sentence when the
 * cron matches one, otherwise the expression itself.
 */
export function humanForCron(cron: string, t: Translate): string {
  return describeCron(cron, t) ?? cron.trim();
}

/**
 * Reverse of `cronForDraft` — seeds the edit form from a stored cron so a
 * schedule created by the agent still opens on the right preset instead of
 * dumping the user into the raw-cron box.
 */
export function draftFromCron(cron: string, timezone: string): ScheduleDraft {
  const base = emptyScheduleDraft(timezone || DEFAULT_TIMEZONE);
  const trimmed = (cron || '').trim();
  base.cron = trimmed;
  const fields = trimmed.split(/\s+/);
  if (fields.length !== 5) return { ...base, preset: 'custom' };
  const [min, hour, dom, mon, rawDow] = fields;
  const dow = rawDow === '7' ? '0' : rawDow;
  if (mon !== '*' || !isNum(min)) return { ...base, preset: 'custom' };

  if (hour === '*') {
    if (dom !== '*' || dow !== '*') return { ...base, preset: 'custom' };
    return { ...base, preset: 'hourly', minute: Number(min) };
  }
  if (!isNum(hour)) return { ...base, preset: 'custom' };
  const time = `${pad2(Number(hour))}:${pad2(Number(min))}`;

  if (dom === '*' && dow === '*') return { ...base, preset: 'daily', time };
  if (dom === '*' && dow === '1-5') return { ...base, preset: 'weekdays', time };
  if (dom === '*' && isNum(dow)) {
    return { ...base, preset: 'weekly', time, weekday: Number(dow) };
  }
  if (isNum(dom) && dow === '*') {
    return { ...base, preset: 'monthly', time, dayOfMonth: Number(dom) };
  }
  return { ...base, preset: 'custom' };
}

/**
 * Timezone choices for the picker: the browser's zone first (what a user
 * almost always means), then UTC, then a short spread of common zones. A full
 * IANA list needs `Intl.supportedValuesOf`, which isn't available everywhere
 * the app runs (and is 400+ rows of scroll on a phone).
 */
const COMMON_TIMEZONES = [
  'UTC',
  'America/Los_Angeles',
  'America/New_York',
  'Europe/London',
  'Europe/Berlin',
  'Asia/Shanghai',
  'Asia/Tokyo',
  'Asia/Singapore',
  'Australia/Sydney',
];

export function timezoneOptions(current?: string): string[] {
  const out: string[] = [];
  for (const zone of [browserTimezone(), ...(current ? [current] : []), ...COMMON_TIMEZONES]) {
    if (zone && !out.includes(zone)) out.push(zone);
  }
  return out;
}

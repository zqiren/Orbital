// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — words that depend on what the daemon sends: the catalogue's
 * label keys (the prototype's keys, kept under `sources.`), a type's own
 * name, a failure's line. A key the catalogue does not have falls back to
 * readable text instead of showing the key.
 */

import { STRINGS, type StringKey } from '../../i18n/strings';
import { translate, type TVars } from '../../i18n/useT';
import type { Locale } from '../../i18n/locales';
import type { HistoryPlace, SourceFailure, SourceSummary } from './types';

export type T = (key: StringKey, vars?: TVars) => string;

export function hasKey(key: string): key is StringKey {
  return Object.prototype.hasOwnProperty.call(STRINGS, key);
}

/** A catalogue label key (`pick.assigned`) in the reader's language. */
export function labelText(t: T, labelKey: string, vars?: TVars): string {
  const k = `sources.${labelKey}`;
  if (hasKey(k)) return t(k, vars);
  const last = labelKey.split('.').pop() ?? labelKey;
  return last.replace(/[_-]+/g, ' ');
}

/** A field's placeholder: literal sample text, or a catalogue key (`web.opt`). */
export function placeholderText(t: T, ph: string | null | undefined, optional: boolean): string {
  if (ph) return hasKey(`sources.${ph}`) ? t(`sources.${ph}` as StringKey) : ph;
  return optional ? t('sources.web.opt') : '';
}

/** A type's own name (日历, GitHub, 网站检查). */
export function typeName(t: T, type: string): string {
  const k = `sources.type.${type}`;
  return hasKey(k) ? t(k) : type;
}

/**
 * What a source is called: the user's name, else the website's host, else
 * the type's own name in the reader's language (the daemon's display_name is
 * English for unnamed built-ins).
 */
export function sourceName(t: T, s: Pick<SourceSummary, 'name' | 'type' | 'display_name' | 'label'>): string {
  if (s.name) return s.name;
  if (s.type === 'website') return s.label || s.display_name || typeName(t, 'website');
  if (hasKey(`sources.type.${s.type}`)) return typeName(t, s.type);
  return s.display_name || s.type;
}

/** The feed's `source.name` is the daemon's display_name: localize a bare type word. */
export function feedSourceName(t: T, type: string, name: string): string {
  const k = `sources.type.${type}`;
  if (hasKey(k) && (!name || name === translate('en', k))) return t(k);
  return name || typeName(t, type);
}

/** A failure's line («找不到仓库 acme/web»). A param the daemon left out
 *  reads as the service's name (host) or nothing, never as «{repo}». */
export function failureText(t: T, f: SourceFailure, type: string): string {
  const k = `sources.fail.${f.code}`;
  const p = f.params ?? {};
  const vars: TVars = { svc: typeName(t, type) };
  for (const [a, b] of Object.entries(p)) vars[a] = b == null ? '' : String(b);
  if (vars.cli === undefined) vars.cli = type === 'gitlab' ? 'glab' : 'gh';
  if (!vars.host) vars.host = vars.svc;
  if (!hasKey(k)) return t('sources.fail.error');
  const tpl = t(k);
  for (const m of tpl.matchAll(/\{(\w+)\}/g)) if (vars[m[1]] === undefined) vars[m[1]] = '';
  return cjkSpacing(t(k, vars).replace(/\s{2,}/g, ' ').trim(), String(vars.svc), tpl);
}

/** «钉钉 不认这个令牌» → «钉钉不认这个令牌»: the service name dropped into a
 *  zh line keeps its space only beside Latin text. Given the line's
 *  template, only a name that came from `{svc}` is touched (the Calendar
 *  permission line names 日历 as a settings pane and keeps its spacing). */
export function cjkSpacing(s: string, name: string, tpl?: string): string {
  const cjk = /[\u3400-\u9fff]/;
  if (!name || !cjk.test(name.slice(-1))) return s;
  if (tpl !== undefined && !tpl.includes('{svc}')) return s;
  const i = s.indexOf(name + ' ');
  if (i < 0 || !cjk.test(s.charAt(i + name.length + 1))) return s;
  return s.slice(0, i + name.length) + s.slice(i + name.length + 1);
}

/** Where a bot message came from: 私聊, the group's or channel's name, or
 *  the plain word when the service gives no name (WeCom groups never do). */
export function placeName(t: T, place: HistoryPlace): string {
  if (place.kind === 'dm') return t('sources.place.dm');
  if (place.name) return place.name;
  return t(place.kind === 'channel' ? 'sources.place.channel' : 'sources.place.group');
}

/** A CLI account as shown: its login over its host, or `user@host` once
 *  for a self-managed GitLab (whose login already carries the host). */
export function accountLine(login: string, host: string | null | undefined): { name: string; host: string | null } {
  if (host && login.toLowerCase().endsWith(`@${host.toLowerCase()}`)) return { name: login, host: null };
  return { name: login, host: host || null };
}

/** «刚刚» · «2 分钟前» … for a list row's last check. */
export function relTime(t: T, iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '—';
  const s = Math.max(0, Math.floor((now - then) / 1000));
  if (s < 60) return t('sources.time.justNow');
  const m = Math.floor(s / 60);
  if (m < 60) return t('sessionItem.relTime.minutes', { n: m });
  const h = Math.floor(m / 60);
  if (h < 24) return t('sessionItem.relTime.hours', { n: h });
  return t('sessionItem.relTime.days', { n: Math.floor(h / 24) });
}

export const localeOf = (l: string): Locale => (l === 'zh' ? 'zh' : 'en');

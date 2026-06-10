// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { STRINGS, type StringEntry, type StringKey } from './strings';
import { useLocale } from './LocaleContext';
import type { Locale } from './locales';

export type TVars = Record<string, string | number>;

/** Resolve an entry for a locale with zh -> en -> key fallback. */
export function resolve(
  entry: StringEntry | undefined,
  locale: Locale,
  key: string,
): string {
  if (!entry) return key;
  const localized = locale === 'en' ? entry.en : entry.zh;
  return localized || entry.en || key;
}

/** Replace {placeholder} tokens; unknown placeholders are left intact. */
export function interpolate(text: string, vars?: TVars): string {
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (m, name) =>
    name in vars ? String(vars[name]) : m,
  );
}

export function translate(locale: Locale, key: StringKey, vars?: TVars): string {
  const entry = STRINGS[key] as StringEntry | undefined;
  return interpolate(resolve(entry, locale, key), vars);
}

/** Hook: returns t(key, vars?) bound to the current locale. */
export function useT() {
  const { locale } = useLocale();
  return (key: StringKey, vars?: TVars) => translate(locale, key, vars);
}

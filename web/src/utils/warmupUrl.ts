// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Locale } from '../i18n/locales';

/**
 * Landing page for the headed sign-in browser. accounts.google.com is
 * unreachable from mainland China and would blank-hang the warmup window
 * for the full navigation timeout, so zh users land on Bing (the same
 * engine the agent browser's China search routing already picks).
 */
export function warmupUrlForLocale(locale: Locale): string {
  return locale === 'zh' ? 'https://www.bing.com' : 'https://accounts.google.com';
}

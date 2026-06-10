// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

export type Locale = 'en' | 'zh';

export interface LocaleOption {
  code: Locale;
  label: string; // shown in the dropdown, in its own language
}

export const LOCALES: LocaleOption[] = [
  { code: 'en', label: 'English' },
  { code: 'zh', label: '中文' },
];

export const DEFAULT_LOCALE: Locale = 'en';
export const LOCALE_STORAGE_KEY = 'orbital.locale';

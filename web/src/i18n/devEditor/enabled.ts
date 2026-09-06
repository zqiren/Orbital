// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

export const I18N_EDIT_STORAGE_KEY = 'orbital.i18nEdit';

/**
 * Dev-only translation editor switch. `?i18n=edit` turns it on for this
 * browser (the flag lives in localStorage because the app rewrites the URL
 * and drops query params on navigation); `?i18n=off` turns it off again.
 */
export function devI18nEditorEnabled(): boolean {
  try {
    const mode = new URLSearchParams(window.location.search).get('i18n');
    if (mode === 'edit') localStorage.setItem(I18N_EDIT_STORAGE_KEY, '1');
    if (mode === 'off') localStorage.removeItem(I18N_EDIT_STORAGE_KEY);
    return localStorage.getItem(I18N_EDIT_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { api } from '../config';
import type { Locale } from './locales';

/**
 * Tell the daemon the interface language and the device's preferred
 * languages. Onboarding prompts hand both to the agent, which picks the
 * language it speaks to the user (2026-11-07). Fire-and-forget: a daemon
 * that is not reachable yet (or an older one without the route) is ignored.
 */
export async function reportUiLanguage(locale: Locale): Promise<void> {
  let languages: string[] = [];
  try {
    languages = Array.from(navigator?.languages ?? []).slice(0, 8);
  } catch {
    /* navigator unavailable */
  }
  try {
    await api('/api/v2/settings/ui-language', {
      method: 'PUT',
      body: JSON.stringify({ ui_language: locale, ui_languages: languages }),
    });
  } catch {
    /* not reachable or an older daemon: nothing to do */
  }
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import {
  createContext, useCallback, useContext, useEffect, useState,
  type ReactNode,
} from 'react';
import { DEFAULT_LOCALE, LOCALE_STORAGE_KEY, type Locale } from './locales';

interface LocaleContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

/**
 * First-run locale resolution: localStorage wins; otherwise fall back to the
 * device's OS/browser language (Spec 008 §8 B1) before defaulting to 'en'.
 * Exported (pure function) so it's directly testable without a DOM render.
 */
export function readInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
    if (stored === 'en' || stored === 'zh') return stored;
  } catch {
    /* localStorage unavailable (SSR/private mode) — use default */
  }
  try {
    if (navigator?.language?.startsWith('zh')) return 'zh';
  } catch {
    /* navigator unavailable — use default */
  }
  return DEFAULT_LOCALE;
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readInitialLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    try {
      localStorage.setItem(LOCALE_STORAGE_KEY, next);
    } catch {
      /* ignore persistence failure */
    }
  }, []);

  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}

// Fallback used when a component renders outside a LocaleProvider (e.g. in unit
// tests, or a stray render). Consistent with the i18n fallback philosophy: a
// missing provider degrades to the default locale rather than crashing.
const FALLBACK_CONTEXT: LocaleContextValue = {
  locale: DEFAULT_LOCALE,
  setLocale: () => {},
};

export function useLocale(): LocaleContextValue {
  return useContext(LocaleContext) ?? FALLBACK_CONTEXT;
}

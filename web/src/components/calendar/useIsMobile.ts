// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';

/**
 * App-standard mobile detection — the same `matchMedia('(max-width: 767px)')`
 * pattern used in App.tsx and SettingsRail. jsdom has no `matchMedia`, so we
 * degrade to desktop (the WeekGrid) when it is absent, keeping unit tests
 * deterministic unless they explicitly stub a matching media query.
 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return isMobile;
}

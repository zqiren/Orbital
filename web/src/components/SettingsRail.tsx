// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SettingsRail — index rail for the two settings surfaces (spec 011 §0.8).
 *
 * Both settings surfaces stay ONE scrolling document with a single Save
 * button; the rail is purely additive navigation:
 *   - Desktop: a slim sticky rail left of the settings content, scrollspy-
 *     highlighted via IntersectionObserver over `[data-settings-section]`.
 *   - Mobile (the app's matchMedia isMobile pattern): a compact jump menu.
 *
 * Entries are declared via `sections` but rendered only when a matching
 * `data-settings-section` element actually exists in the container — a
 * reserved id (e.g. 'connectors' before its section ships) never becomes a
 * dead nav entry. Rendered order follows DOM order, not array order.
 */

import { useCallback, useEffect, useState, type RefObject } from 'react';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import Select from './Select';

export interface SettingsRailSection {
  /** Matches the target element's `data-settings-section` attribute. */
  id: string;
  /** Catalog key for the entry label — reuse the group's heading key. */
  labelKey: StringKey;
}

interface SettingsRailProps {
  sections: SettingsRailSection[];
  /** The scrollable settings container the tagged sections live in. */
  containerRef: RefObject<HTMLElement | null>;
}

/**
 * Scroll the section tagged `data-settings-section={id}` into view.
 *
 * This is the generalized budget-anchor mechanism (previously a one-off ref
 * scroll in SettingsView) — both the rail clicks and the `settingsAnchor`
 * route deep-link go through here. Best-effort: no-op when the section is
 * absent or in jsdom (no scrollIntoView).
 */
export function scrollToSettingsSection(container: ParentNode, id: string): boolean {
  const el = container.querySelector<HTMLElement>(`[data-settings-section="${id}"]`);
  if (!el) return false;
  el.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  return true;
}

/** App-standard mobile detection (matchMedia pattern from App.tsx). */
function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    // jsdom has no matchMedia — degrade to desktop rendering.
    if (typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return isMobile;
}

export default function SettingsRail({ sections, containerRef }: SettingsRailProps) {
  const t = useT();
  const isMobile = useIsMobile();
  const [presentIds, setPresentIds] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // Discover which tagged sections exist, in DOM order. A MutationObserver
  // keeps the list current when sections mount late (fetched data, a future
  // 'connectors' section landing conditionally, …).
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const discover = () => {
      const ids: string[] = [];
      container.querySelectorAll('[data-settings-section]').forEach((el) => {
        const id = el.getAttribute('data-settings-section');
        if (id) ids.push(id);
      });
      setPresentIds((prev) =>
        prev.length === ids.length && prev.every((v, i) => v === ids[i]) ? prev : ids,
      );
    };
    discover();
    const mo = new MutationObserver(discover);
    mo.observe(container, { childList: true, subtree: true });
    return () => mo.disconnect();
  }, [containerRef]);

  // Scrollspy: highlight the topmost section currently in view. Root is the
  // scroll container itself; the bottom margin biases toward the section the
  // reader is actually looking at rather than one peeking in at the bottom.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof IntersectionObserver === 'undefined') return;
    const els = Array.from(
      container.querySelectorAll<HTMLElement>('[data-settings-section]'),
    );
    if (els.length === 0) return;
    const intersecting = new Map<string, boolean>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = entry.target.getAttribute('data-settings-section');
          if (id) intersecting.set(id, entry.isIntersecting);
        }
        for (const el of els) {
          const id = el.getAttribute('data-settings-section');
          if (id && intersecting.get(id)) {
            setActiveId(id);
            return;
          }
        }
      },
      { root: container, rootMargin: '0px 0px -55% 0px' },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [containerRef, presentIds]);

  const handleJump = useCallback(
    (id: string) => {
      const container = containerRef.current;
      if (!container) return;
      scrollToSettingsSection(container, id);
      setActiveId(id); // optimistic highlight; scrollspy confirms on arrival
    },
    [containerRef],
  );

  // Only sections that exist in the DOM AND are declared get an entry,
  // ordered by their position in the document.
  const byId = new Map(sections.map((s) => [s.id, s]));
  const visible = presentIds
    .map((id) => byId.get(id))
    .filter((s): s is SettingsRailSection => s !== undefined);

  if (visible.length === 0) return null;

  if (isMobile) {
    // Compact jump menu — plain sticky positioning (WKWebView-safe).
    return (
      <div className="sticky top-0 z-10 bg-background border-b border-border px-4 py-2">
        <Select
          data-testid="settings-jump-menu"
          aria-label={t('settingsRail.aria')}
          value={activeId && byId.has(activeId) ? activeId : ''}
          onChange={(e) => {
            if (e.target.value) handleJump(e.target.value);
          }}
          className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent"
        >
          <option value="" disabled>
            {t('settingsRail.jump')}
          </option>
          {visible.map((s) => (
            <option key={s.id} value={s.id}>
              {t(s.labelKey)}
            </option>
          ))}
        </Select>
      </div>
    );
  }

  return (
    <nav
      data-testid="settings-rail"
      aria-label={t('settingsRail.aria')}
      className="sticky top-0 self-start shrink-0 w-44 py-10 pl-6 pr-2 max-lg:hidden"
    >
      <ul className="space-y-0.5">
        {visible.map((s) => (
          <li key={s.id}>
            <button
              type="button"
              onClick={() => handleJump(s.id)}
              aria-current={activeId === s.id ? 'true' : undefined}
              className={`block w-full text-left text-[13px] leading-5 rounded-md px-2.5 py-1.5 transition-colors duration-150 truncate ${
                activeId === s.id
                  ? 'text-primary bg-sidebar font-medium'
                  : 'text-secondary hover:text-primary hover:bg-sidebar/60'
              }`}
            >
              {t(s.labelKey)}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

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
  /**
   * Optional chapter this entry belongs to, matching the `SettingsGroup`
   * heading in the document. On desktop it only clusters the entries (see the
   * note at the render site — the label itself would double the rail's
   * height); the mobile jump menu prints it as a real `<optgroup>` label. A
   * chapter whose sections are all absent from the DOM produces no run at all.
   */
  groupKey?: StringKey;
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

/**
 * Split entries into consecutive same-group runs, in DOM order.
 *
 * Grouping is derived from the entries that survived the "is it in the DOM"
 * filter, never from the declaration array — a chapter whose sections are all
 * absent (scratch projects drop several) simply produces no run, so neither
 * surface can print an empty cluster or a stray `<optgroup>`.
 */
function groupRuns(
  items: SettingsRailSection[],
): { groupKey?: StringKey; items: SettingsRailSection[] }[] {
  const runs: { groupKey?: StringKey; items: SettingsRailSection[] }[] = [];
  for (const item of items) {
    const last = runs[runs.length - 1];
    if (last && last.groupKey === item.groupKey) last.items.push(item);
    else runs.push({ groupKey: item.groupKey, items: [item] });
  }
  return runs;
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
          {groupRuns(visible).map((run) =>
            run.groupKey ? (
              <optgroup key={run.groupKey + run.items[0].id} label={t(run.groupKey)}>
                {run.items.map((s) => (
                  <option key={s.id} value={s.id}>
                    {t(s.labelKey)}
                  </option>
                ))}
              </optgroup>
            ) : (
              run.items.map((s) => (
                <option key={s.id} value={s.id}>
                  {t(s.labelKey)}
                </option>
              ))
            ),
          )}
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
      {/* Chapter labels are deliberately NOT repeated here. The rail is a
          twelve-line index in a 176px column; naming the chapters a second
          time doubled its height and competed with the entries it exists to
          list. The grouping survives as the gap between runs — enough to
          cluster, nothing to read. The mobile jump menu still uses real
          <optgroup> labels: a native picker gives them for free, and a flat
          thirteen-option select is genuinely worse to scan. */}
      {groupRuns(visible).map((run, runIdx) => (
        <div key={run.groupKey ?? run.items[0].id} className={runIdx > 0 ? 'mt-3.5' : ''}>
          <ul className="space-y-0.5">
            {run.items.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => handleJump(s.id)}
                  aria-current={activeId === s.id ? 'true' : undefined}
                  // The active pill tints with `card-hover`, not `sidebar`.
                  // Both settings surfaces sit on `background` (#F4F6F9), and
                  // `sidebar` (#F0F3F7) against it is 1.03:1 — deltas of 4/3/2
                  // per channel, i.e. nothing. `card-hover` is the token
                  // index.css already documents for this exact job ("full
                  // strength for the selected row and at /50 for hover"), and
                  // lands at 1.14:1 — just past the nav column's own selected
                  // row (bg-nav-hover on bg-nav, 1.11:1), so the two columns
                  // read as one system instead of the rail being three times
                  // fainter than every other selection in the app.
                  className={`block w-full text-left text-[13px] leading-5 rounded-md px-2.5 py-1.5 transition-colors duration-150 truncate ${
                    activeId === s.id
                      ? 'text-primary bg-card-hover font-medium'
                      : 'text-secondary hover:text-primary hover:bg-card-hover/50'
                  }`}
                >
                  {t(s.labelKey)}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

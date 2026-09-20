// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * FirstJourneyTour — coachmarks anchored on the REAL project UI (not a modal
 * carousel): each stop spotlights the actual control and says what it is for.
 *
 * Anchors are found by `data-tour="<id>"` attributes. A stop whose anchor is
 * not on screen is skipped — that is the whole conditional-stop mechanism:
 * no sub-agents installed → the pin mark renders nothing → no "agents" stop;
 * window too narrow to dock the panel → no "panel" stop. The counter counts
 * only the stops that will actually be shown.
 *
 * Deliberately plain for WKWebView: no animated spotlight, no transitions on
 * the geometry — positions are recomputed on resize/scroll and simply set.
 * No library.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';
import { placeCard, type Box } from './tourLogic';

interface TourStep {
  id: string;
  /** `data-tour` ids, in preference order — first one on screen wins. */
  anchors: string[];
  titleKey: StringKey;
  bodyKey: StringKey;
}

const STEPS: TourStep[] = [
  { id: 'project', anchors: ['project-header'], titleKey: 'tour.project.title', bodyKey: 'tour.project.body' },
  { id: 'composer', anchors: ['composer'], titleKey: 'tour.composer.title', bodyKey: 'tour.composer.body' },
  { id: 'agents', anchors: ['pin-select'], titleKey: 'tour.agents.title', bodyKey: 'tour.agents.body' },
  { id: 'panel', anchors: ['workspace-panel', 'panel-handle'], titleKey: 'tour.panel.title', bodyKey: 'tour.panel.body' },
  { id: 'files', anchors: ['tab-files'], titleKey: 'tour.files.title', bodyKey: 'tour.files.body' },
  { id: 'tasks', anchors: ['tab-queue'], titleKey: 'tour.tasks.title', bodyKey: 'tour.tasks.body' },
  { id: 'settings', anchors: ['project-settings'], titleKey: 'tour.settings.title', bodyKey: 'tour.settings.body' },
  // Inside a project the project list is collapsed to the edge strip, so the
  // Quick Tasks row itself is usually not on screen — the strip is where it lives.
  { id: 'quickTasks', anchors: ['quick-tasks-row', 'edge-strip'], titleKey: 'tour.quickTasks.title', bodyKey: 'tour.quickTasks.body' },
];

const CARD_WIDTH = 320;
const SPOTLIGHT_PAD = 6;

/** Laid out AND actually visible. A non-zero box is not enough: EdgeStrip
 * keeps its closed flyout painted at opacity 0 (a WKWebView compositor
 * workaround), so the Sidebar inside it — Quick Tasks row included — measures
 * a real box while showing nothing. `visibility` inherits, so it is read off
 * the element; `opacity` does not, so the ancestors are walked. */
function isOnScreen(el: HTMLElement): boolean {
  const r = el.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0) return false;
  if (getComputedStyle(el).visibility === 'hidden') return false;
  for (let node: HTMLElement | null = el; node; node = node.parentElement) {
    if (getComputedStyle(node).opacity === '0') return false;
  }
  return true;
}

function findAnchor(step: TourStep): HTMLElement | null {
  for (const id of step.anchors) {
    // querySelectorAll: the same id can exist twice (the Sidebar is rendered
    // both on the home route and inside the flyout) — take the visible one.
    for (const el of document.querySelectorAll<HTMLElement>(`[data-tour="${id}"]`)) {
      if (isOnScreen(el)) return el;
    }
  }
  return null;
}

export interface FirstJourneyTourProps {
  /** Folder name for the first stop ("Your agent works inside {folder}"). */
  folderName: string;
  /** Installed sub-agent names for the agents stop. */
  agentNames: string[];
  /** Called once, when the user finishes or skips. */
  onClose: () => void;
}

export default function FirstJourneyTour({ folderName, agentNames, onClose }: FirstJourneyTourProps) {
  const t = useT();
  // The stops that exist on THIS screen, fixed when the tour starts so the
  // counter cannot change under the user mid-tour.
  const [steps, setSteps] = useState<TourStep[] | null>(null);
  const [index, setIndex] = useState(0);
  const [anchorBox, setAnchorBox] = useState<Box | null>(null);
  const [cardHeight, setCardHeight] = useState(0);
  const cardRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLButtonElement>(null);

  // Wait for the project screen to be on the page (the tour starts the moment
  // the route changes; the header mounts a frame or two later), then freeze
  // the list of stops. Gives up quietly if it never shows.
  useEffect(() => {
    let tries = 0;
    let timer = 0;
    const probe = () => {
      if (findAnchor(STEPS[0])) {
        setSteps(STEPS.filter((s) => findAnchor(s)));
        return;
      }
      tries += 1;
      if (tries > 40) { onClose(); return; }
      timer = window.setTimeout(probe, 100);
    };
    probe();
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const step = steps?.[index];

  const measure = useCallback(() => {
    if (!step) return;
    const el = findAnchor(step);
    if (!el) { setAnchorBox(null); return; }
    const r = el.getBoundingClientRect();
    setAnchorBox({ left: r.left, top: r.top, width: r.width, height: r.height });
  }, [step]);

  useLayoutEffect(() => {
    measure();
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);
    return () => {
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure, true);
    };
  }, [measure]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardHeight(cardRef.current.offsetHeight);
  }, [step, anchorBox]);

  useEffect(() => {
    nextRef.current?.focus({ preventScroll: true });
  }, [step]);

  const isLast = !!steps && index === steps.length - 1;
  const advance = useCallback(() => {
    if (!steps) return;
    if (index >= steps.length - 1) onClose();
    else setIndex((i) => i + 1);
  }, [steps, index, onClose]);

  // Esc skips. Capture phase + stopPropagation so a dialog underneath (none is
  // expected mid-tour, but the listeners are window-level) does not also act.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onClose]);

  const vars = useMemo(
    () => ({ folder: folderName, agents: agentNames.join(', ') }),
    [folderName, agentNames],
  );

  if (!steps || !step || !anchorBox) return null;

  const pos = placeCard(
    anchorBox,
    { width: CARD_WIDTH, height: cardHeight || 180 },
    { width: window.innerWidth, height: window.innerHeight },
  );
  const titleId = `tour-title-${step.id}`;

  return createPortal(
    <div data-testid="first-journey-tour">
      {/* Click blocker: the tour is modal — a stray click on the app beneath
          would navigate away from the thing being explained. */}
      <div className="fixed inset-0 z-[200]" />
      {/* Spotlight: a hole the size of the anchor, everything else dimmed by
          one huge box-shadow. pointer-events-none — the blocker owns clicks. */}
      <div
        aria-hidden="true"
        data-testid="tour-spotlight"
        className="fixed z-[201] rounded-[10px] pointer-events-none"
        style={{
          left: anchorBox.left - SPOTLIGHT_PAD,
          top: anchorBox.top - SPOTLIGHT_PAD,
          width: anchorBox.width + SPOTLIGHT_PAD * 2,
          height: anchorBox.height + SPOTLIGHT_PAD * 2,
          boxShadow: '0 0 0 9999px rgb(15 17 26 / 0.42), 0 0 0 1.5px rgb(255 255 255 / 0.9)',
        }}
      />
      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="tour-card"
        data-step={step.id}
        className="fixed z-[202] rounded-xl border border-border bg-card p-4 shadow-[0_18px_40px_-12px_rgb(0_0_0/0.35)]"
        style={{ left: pos.left, top: pos.top, width: CARD_WIDTH }}
      >
        <h2 id={titleId} className="text-sm font-semibold text-primary">
          {t(step.titleKey)}
        </h2>
        <p className="mt-1.5 text-[13px] leading-relaxed text-secondary">
          {t(step.bodyKey, vars)}
        </p>
        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="text-2xs font-mono text-secondary" data-testid="tour-counter">
            {t('tour.counter', { n: index + 1, total: steps.length })}
          </span>
          <div className="flex items-center gap-1">
            {!isLast && (
              <button
                type="button"
                onClick={onClose}
                data-testid="tour-skip"
                className="text-xs text-secondary hover:text-primary transition-colors duration-150 px-2.5 py-1.5 max-md:min-h-[44px]"
              >
                {t('tour.skip')}
              </button>
            )}
            <button
              ref={nextRef}
              type="button"
              onClick={advance}
              data-testid="tour-next"
              className="bg-accent text-white text-xs font-medium rounded-lg px-3.5 py-1.5 hover:bg-accent/90 transition-all duration-150 max-md:min-h-[44px]"
            >
              {isLast ? t('tour.done') : t('tour.next')}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

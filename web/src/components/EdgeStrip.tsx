// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * EdgeStrip — the 20px left-edge rail shown in place of the full Sidebar
 * while a project is open on desktop (spec 078 §4).
 *
 * Hovering the strip floats the real `Sidebar` (passed in as `children`,
 * verbatim — D3) over the session column in an absolutely positioned flyout;
 * clicking (or Enter/Space on) the strip navigates home. Nothing about other
 * projects is shown while the flyout is closed except one aggregate dot
 * (§4.4): red (error) beats amber (approval pending) beats nothing. Running
 * elsewhere is deliberately not surfaced here — that's what the dot rows
 * inside the flyout are for.
 */

import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import type { Project, AgentRunStatus } from '../types';
import { useT } from '../i18n/useT';
import { getProjectDotColor } from '../utils/projectStatus';

// Hover-intent timings (§4.2). Short enough that a deliberate hover feels
// instant, long enough that a cursor passing over the 20px rail on its way
// elsewhere never pops the flyout open.
const OPEN_DELAY_MS = 120;
const CLOSE_DELAY_MS = 160;

interface EdgeStripProps {
  projects: Project[];
  /** Excluded from the aggregate dot — the strip never reports on itself. */
  currentProjectId: string;
  agentStatuses: Record<string, AgentRunStatus>;
  pendingApprovals: Record<string, number>;
  onGoHome: () => void;
  /** The real `Sidebar` element, rendered verbatim inside the flyout (D3). */
  children: ReactNode;
}

type HoverZone = 'strip' | 'flyout';

/** Aggregate dot color + count over every project except `currentProjectId`.
 *  Precedence: error beats pending-approval beats nothing (running elsewhere
 *  is not shown — D4). Uses the same `getProjectDotColor` as Sidebar's rows
 *  so the two agree byte-for-byte on what "red"/"amber" mean. */
function aggregateDot(
  projects: Project[],
  currentProjectId: string,
  agentStatuses: Record<string, AgentRunStatus>,
  pendingApprovals: Record<string, number>,
): { color: 'bg-error' | 'bg-warning' | null; count: number } {
  let color: 'bg-error' | 'bg-warning' | null = null;
  let count = 0;
  for (const project of projects) {
    if (project.project_id === currentProjectId) continue;
    const dot = getProjectDotColor(project.project_id, agentStatuses, pendingApprovals);
    if (dot !== 'bg-error' && dot !== 'bg-warning') continue;
    count += 1;
    if (dot === 'bg-error') {
      color = 'bg-error';
    } else if (color !== 'bg-error') {
      color = 'bg-warning';
    }
  }
  return { color, count };
}

export default function EdgeStrip({
  projects,
  currentProjectId,
  agentStatuses,
  pendingApprovals,
  onGoHome,
  children,
}: EdgeStripProps) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const stripRef = useRef<HTMLButtonElement>(null);
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Read from setTimeout callbacks outside React's render cycle, so this has
  // to be a ref (not state) — a value captured in a closure at schedule time
  // would go stale by the time the timer fires (see CLAUDE.md's note on
  // closures + setState updaters; same hazard, same fix here).
  const insideRef = useRef<{ strip: boolean; flyout: boolean }>({ strip: false, flyout: false });

  function clearOpenTimer() {
    if (openTimerRef.current !== null) {
      clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
  }

  function clearCloseTimer() {
    if (closeTimerRef.current !== null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }

  useEffect(() => () => {
    clearOpenTimer();
    clearCloseTimer();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleZoneEnter(zone: HoverZone) {
    insideRef.current[zone] = true;
    clearCloseTimer();
    if (open) return;
    clearOpenTimer();
    openTimerRef.current = setTimeout(() => {
      openTimerRef.current = null;
      setOpen(true);
    }, OPEN_DELAY_MS);
  }

  function handleZoneLeave(zone: HoverZone) {
    insideRef.current[zone] = false;
    // A brush-past that never reached the open delay must not open later.
    clearOpenTimer();
    if (!open) return;
    clearCloseTimer();
    closeTimerRef.current = setTimeout(() => {
      closeTimerRef.current = null;
      // Re-check at fire time: entering the flyout (or strip) in the
      // meantime cancels this close, per §4.2.
      if (!insideRef.current.strip && !insideRef.current.flyout) {
        setOpen(false);
      }
    }, CLOSE_DELAY_MS);
  }

  function closeFlyout() {
    clearOpenTimer();
    clearCloseTimer();
    insideRef.current = { strip: false, flyout: false };
    setOpen(false);
  }

  function closeAndFocusStrip() {
    closeFlyout();
    stripRef.current?.focus();
  }

  // Every interactive control inside Sidebar (project row, Calendar,
  // Workbench, New project, Settings) navigates somewhere — there is nothing
  // clickable in it that isn't a selection. So closing on any click that
  // bubbles up from the flyout's content is exactly "selecting a project (or
  // any other row) closes it" (§4.3), with no extra plumbing into Sidebar's
  // own onSelectProject/onSettings callbacks (out of this component's file
  // scope). A completed drag-reorder doesn't emit a trailing click, so this
  // doesn't fire mid-drag.
  function handleFlyoutClick() {
    closeFlyout();
  }

  // Esc closes the flyout regardless of where focus is inside it (mirrors
  // FilePreviewDrawer's window-level Esc handling).
  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: globalThis.KeyboardEvent) {
      if (e.key === 'Escape') closeAndFocusStrip();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function handleStripKeyDown(e: KeyboardEvent<HTMLButtonElement>) {
    // preventDefault so a native button's own Enter/Space → click synthesis
    // never fires alongside this handler and calls onGoHome twice.
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onGoHome();
    }
  }

  const { color: dotColor, count: dotCount } = aggregateDot(
    projects,
    currentProjectId,
    agentStatuses,
    pendingApprovals,
  );
  const dotTitle = dotCount === 1
    ? t('strip.needsYou.one')
    : t('strip.needsYou.other', { n: dotCount });

  return (
    <div className="relative h-full w-5 shrink-0">
      <button
        ref={stripRef}
        type="button"
        aria-label={t('strip.projects')}
        aria-expanded={open}
        title={t('strip.tooltip')}
        onClick={onGoHome}
        onKeyDown={handleStripKeyDown}
        onMouseEnter={() => handleZoneEnter('strip')}
        onMouseLeave={() => handleZoneLeave('strip')}
        className="flex h-full w-full shrink-0 flex-col items-center border-r border-border bg-nav pt-titlebar focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-accent"
      >
        {/* pt-titlebar (above) clears the mac band on its own; this inner
            span carries the strip's own visual padding so the two pt-*
            declarations never compete on one element. */}
        <span className="flex flex-col items-center gap-2 pt-3 pb-3">
          <ChevronRight size={12} aria-hidden="true" className="shrink-0 text-secondary" />
          {dotColor && (
            <span
              data-testid="strip-dot"
              title={dotTitle}
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${dotColor}`}
            />
          )}
        </span>
      </button>

      {/* Flyout: overlays, never pushes (App's flex root doesn't change
          width). Painted while hidden — opacity + a 6px slide, not
          display:none — because WKWebView blinks a stale layer on the first
          frame after an off-screen-to-visible display swap (spec 078 §11.7 /
          reference_wkwebview_compositor_quirks). z-20 sits above the static
          session column and chat, below drawer/modal overlays (z-30+). */}
      <div
        role="presentation"
        onMouseEnter={() => handleZoneEnter('flyout')}
        onMouseLeave={() => handleZoneLeave('flyout')}
        onClick={handleFlyoutClick}
        className={`absolute left-5 top-0 bottom-0 z-20 w-[260px] shadow-xl transition-all duration-150 ease-out motion-reduce:transition-none ${
          open
            ? 'translate-x-0 opacity-100'
            : 'pointer-events-none -translate-x-1.5 opacity-0'
        }`}
      >
        {children}
      </div>
    </div>
  );
}

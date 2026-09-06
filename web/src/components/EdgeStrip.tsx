// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * EdgeStrip — the 20px left-edge rail shown in place of the full Sidebar
 * while a project is open on desktop (spec 078 §4), and the toggle that pins
 * the Sidebar back beside it as a docked column.
 *
 * Hovering the rail floats the real `Sidebar` (passed in as `children`,
 * verbatim — D3) over the session column in an absolutely positioned flyout.
 * Clicking (or Enter/Space on) the rail PINS the list: the flyout stops
 * floating and the content to its right slides over to make room, and the
 * list stays until the rail is clicked again. The rail never navigates — the
 * old "click → home" sent the user to a route with nothing on it (the
 * project IS the work interface; every destination the list offers is inside
 * the list itself), which read as the whole UI going blank.
 *
 * The rail is cut from the list's width, not added to it: the column is 260
 * whether the rail is there or not — the same 260 every non-project route
 * (Calendar, Workbench, Settings) gives the full Sidebar — so `<main>` starts
 * at x=260 whether the list is pinned here or the user has navigated to one
 * of those routes. The list's content is inset 28px from BOTH edges of that
 * column in every state (Sidebar `narrow` vs its desktop default), so it
 * reads centered and nothing inside it shifts when the rail comes or goes.
 * To make that add up the 20px rail shares 8px with the list's own left
 * padding: the list is 248px at x=12, the rail sits above it (z-21) in the
 * same nav colour, and rows start at x=20 — the rail's right edge — so no
 * row highlight is ever clipped. The list does not move between hover and
 * pin either: the flyout already sits exactly where the docked column lives,
 * so pinning only widens this wrapper — one width transition, no transform,
 * no remount.
 *
 * Nothing about other projects is shown while the flyout is closed except
 * one aggregate dot (§4.4): red (error) beats amber (approval pending) beats
 * nothing. Running elsewhere is deliberately not surfaced here — that's what
 * the dot rows inside the flyout are for. Pinned, the real per-project dots
 * are on screen, so the aggregate dot hides.
 */

import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
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
  /** Docked: the list is a real column and hover-intent is off. */
  pinned: boolean;
  /** Click / Enter / Space on the rail. The owner flips `pinned` and persists it. */
  onTogglePin: () => void;
  /** The real `Sidebar` element (rendered `narrow`, 248px), verbatim inside the flyout (D3). */
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
  pinned,
  onTogglePin,
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

  // What the user sees: docked, or floated by hover-intent.
  const visible = pinned || open;

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
    if (pinned) return;
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
    if (pinned) return;
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

  // Pin: the pending hover timers are moot (hover-intent is off while
  // docked). Unpin: the list stays floated — the cursor is still on the rail
  // after the click, so it closes on the usual leave, and the toggle reads as
  // one continuous control rather than a mode switch. Keyboard unpin leaves
  // it floated too; Esc or a later hover-leave closes it.
  function handleToggle() {
    clearOpenTimer();
    clearCloseTimer();
    setOpen(pinned);
    onTogglePin();
  }

  // Every interactive control inside Sidebar (project row, Calendar,
  // Workbench, New project, Settings) navigates somewhere — there is nothing
  // clickable in it that isn't a selection. So closing on any click that
  // bubbles up from the flyout's content is exactly "selecting a project (or
  // any other row) closes it" (§4.3), with no extra plumbing into Sidebar's
  // own onSelectProject/onSettings callbacks (out of this component's file
  // scope). A completed drag-reorder doesn't emit a trailing click, so this
  // doesn't fire mid-drag. Docked, a selection is just a selection.
  function handleFlyoutClick() {
    if (pinned) return;
    closeFlyout();
  }

  // Esc closes the floated flyout regardless of where focus is inside it
  // (mirrors FilePreviewDrawer's window-level Esc handling). It never unpins:
  // Esc in the chat has other jobs, and a docked list is not a transient.
  useEffect(() => {
    if (!open || pinned) return;
    function onKeyDown(e: globalThis.KeyboardEvent) {
      if (e.key === 'Escape') closeAndFocusStrip();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, pinned]);

  function handleStripKeyDown(e: KeyboardEvent<HTMLButtonElement>) {
    // preventDefault so a native button's own Enter/Space → click synthesis
    // never fires alongside this handler and toggles twice.
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleToggle();
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
  const Chevron = pinned ? ChevronLeft : ChevronRight;

  return (
    // The wrapper is the only thing that changes size: 20px floated, 260px
    // (the full column) docked. The flyout below is absolutely positioned at
    // left-3 in both states, so pinning never moves the list — the width
    // transition here is what slides `<main>` over. Width, not transform,
    // because WKWebView blinks a stale layer on the first frame after a
    // transformed layer's visibility swap (reference_wkwebview_compositor_quirks).
    <div
      data-testid="edge-strip-wrapper"
      data-pinned={pinned ? 'true' : 'false'}
      className={`relative h-full shrink-0 transition-[width] duration-200 ease-out motion-reduce:transition-none ${
        pinned ? 'w-[260px]' : 'w-5'
      }`}
    >
      <button
        ref={stripRef}
        type="button"
        aria-label={t('strip.projects')}
        aria-expanded={visible}
        aria-pressed={pinned}
        title={pinned ? t('strip.tooltipPinned') : t('strip.tooltip')}
        onClick={handleToggle}
        onKeyDown={handleStripKeyDown}
        onMouseEnter={() => handleZoneEnter('strip')}
        onMouseLeave={() => handleZoneLeave('strip')}
        className={`absolute inset-y-0 left-0 z-[21] flex w-5 flex-col items-center bg-nav pt-titlebar focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-accent ${
          // The hairline is the collapsed rail's edge. Once the list is
          // visible (floated or docked) the rail is the list's left gutter —
          // same colour, above the list's padding — and a hairline there
          // would cut through it; the floated list's shadow marks its edge.
          visible ? '' : 'border-r border-border'
        }`}
      >
        {/* pt-titlebar (above) clears the mac band on its own; this inner
            span carries the strip's own visual padding so the two pt-*
            declarations never compete on one element. pt-5 centres the 12px
            chevron on the wordmark row's 20px logo (pt-4 + 10) so the two
            read as one header line when the list is visible. */}
        <span className="flex flex-col items-center gap-2 pt-5 pb-3">
          <Chevron size={12} aria-hidden="true" className="shrink-0 text-secondary" />
          {dotColor && !pinned && (
            <span
              data-testid="strip-dot"
              title={dotTitle}
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${dotColor}`}
            />
          )}
        </span>
      </button>

      {/* Flyout: overlays while floated, never pushes (only the wrapper's
          width changes, and only when pinned). 248px at x=12 — the list's
          width with the rail cut out of it, sharing 8px of padding under the
          rail (see the header comment). Painted while hidden — opacity + a 6px
          slide, not display:none — because WKWebView blinks a stale layer on
          the first frame after an off-screen-to-visible display swap (spec
          078 §11.7 / reference_wkwebview_compositor_quirks). z-20 sits above
          the static session column and chat, below drawer/modal overlays
          (z-30+). Docked, the shadow goes: a column doesn't cast one. */}
      <div
        role="presentation"
        data-testid="edge-strip-flyout"
        onMouseEnter={() => handleZoneEnter('flyout')}
        onMouseLeave={() => handleZoneLeave('flyout')}
        onClick={handleFlyoutClick}
        className={`absolute left-3 top-0 bottom-0 z-20 w-[248px] transition-all duration-150 ease-out motion-reduce:transition-none ${
          pinned
            ? 'translate-x-0 opacity-100'
            : open
              ? 'translate-x-0 opacity-100 shadow-xl'
              : 'pointer-events-none -translate-x-1.5 opacity-0'
        }`}
      >
        {children}
      </div>
    </div>
  );
}

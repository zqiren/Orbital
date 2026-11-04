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
 *
 * Spec 154 §5.9: the one thing at the top of the strip is the Workbench
 * glyph — a 14px four-square grid with a small blue dot while anything waits
 * on the user, tooltip `工作台 ⌘0`. Clicking it goes home and never toggles
 * the list. A button cannot hold a button, so it is a sibling stacked above
 * the rail; the rail keeps its toggle, hover-intent and aggregate dot. The
 * strip also frames home (no current project), so `currentProjectId` is
 * optional.
 */

import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight, LayoutGrid } from 'lucide-react';
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
  /** Excluded from the aggregate dot — the strip never reports on itself.
   *  Absent on home (the Workbench), which has no current project. */
  currentProjectId?: string;
  /** Spec 154: anything is waiting on the user (the glyph's blue dot). */
  waiting?: boolean;
  /** Spec 154: the Workbench glyph was clicked. Absent = no glyph. */
  onHome?: () => void;
  /** The Workbench is on screen (`aria-current="page"` on the glyph). */
  homeActive?: boolean;
  agentStatuses: Record<string, AgentRunStatus>;
  pendingApprovals: Record<string, number>;
  /** run-status `sub_agents_running` per project (spec 095). */
  subAgentsRunning?: Record<string, boolean>;
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
  currentProjectId: string | undefined,
  agentStatuses: Record<string, AgentRunStatus>,
  pendingApprovals: Record<string, number>,
  subAgentsRunning: Record<string, boolean>,
): { color: 'bg-error-text' | 'bg-warning-text' | null; count: number } {
  let color: 'bg-error-text' | 'bg-warning-text' | null = null;
  let count = 0;
  for (const project of projects) {
    if (project.project_id === currentProjectId) continue;
    const dot = getProjectDotColor(
      project.project_id, agentStatuses, pendingApprovals, subAgentsRunning,
    );
    if (dot !== 'bg-error-text' && dot !== 'bg-warning-text') continue;
    count += 1;
    if (dot === 'bg-error-text') {
      color = 'bg-error-text';
    } else if (color !== 'bg-error-text') {
      color = 'bg-warning-text';
    }
  }
  return { color, count };
}

/** The shortcut the glyph's tooltip names: ⌘0 on a Mac, Ctrl+0 elsewhere. */
export function homeShortcutLabel(): string {
  const nav = typeof navigator !== 'undefined' ? navigator : undefined;
  const platform = (nav as { userAgentData?: { platform?: string } } | undefined)?.userAgentData?.platform
    ?? nav?.platform ?? '';
  return /mac|iphone|ipad/i.test(platform) ? '⌘0' : 'Ctrl+0';
}

export default function EdgeStrip({
  projects,
  currentProjectId,
  waiting = false,
  onHome,
  homeActive = false,
  agentStatuses,
  pendingApprovals,
  subAgentsRunning = {},
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
    subAgentsRunning,
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
      data-tour="edge-strip"
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
        <span className="flex flex-col items-center gap-2 pb-3 pt-5">
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

      {/* Spec 154: the Workbench glyph, a sibling above the rail (a button
          cannot hold a button). It sits at the height of the list's 工作台
          row icon (2026-11-04), so collapsing or expanding the list keeps the
          icon in place: brand row 56px + 工作区 header 26px + half the 32px
          row = 98px below the titlebar band, minus half its own 22px = 87.
          Docked, the list's row is on screen at that height, so the glyph
          steps aside; floated by hover it stays, since the pointer may be on
          its way to it. The band itself passes clicks to the rail; only the
          glyph takes them. */}
      {onHome && !pinned && (
        <div className="pointer-events-none absolute inset-x-0 top-0 z-[22] w-5 pt-titlebar">
          <button
            type="button"
            data-testid="strip-workbench"
            aria-label={t('workbench.title')}
            aria-current={homeActive ? 'page' : undefined}
            aria-keyshortcuts={homeShortcutLabel() === '⌘0' ? 'Meta+0' : 'Control+0'}
            onClick={onHome}
            className={`group pointer-events-auto relative mt-[87px] flex h-[22px] w-5 items-center justify-center rounded-[5px] focus:outline-none focus-visible:ring-1 focus-visible:ring-accent ${
              homeActive ? 'text-accent' : 'text-secondary hover:text-primary'
            }`}
          >
            <LayoutGrid size={14} strokeWidth={2} aria-hidden="true" />
            {waiting && !homeActive && (
              <span
                data-testid="strip-workbench-dot"
                className="absolute right-px top-0.5 h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_0_1.5px_var(--color-nav)]"
              />
            )}
            <span
              role="tooltip"
              className="pointer-events-none absolute left-[26px] top-1/2 z-30 flex -translate-y-1/2 items-center gap-2 whitespace-nowrap rounded-[7px] bg-[#18181B] px-[9px] py-[5px] text-xs font-medium text-white opacity-0 shadow-[0_6px_18px_rgb(0_0_0/0.18)] transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-[#F4F6F9] dark:text-[#18181B]"
            >
              {t('workbench.title')}
              <kbd className="font-sans text-[11px] opacity-60">{homeShortcutLabel()}</kbd>
            </span>
          </button>
        </div>
      )}

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

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One row of the Workbench list (spec §6): a flagged `[user]` entry.
 *
 * Entry card: sentence (markdown) · project chip (global view only) ·
 * section, as a small always-visible inline label (plain text, no expander,
 * hidden when `section` is null) · age · overdue badge · two exits (Done,
 * Delete). Tapping the card body anywhere but a button is the doorway
 * (`onOpen`) — spec §5.3.
 *
 * The "Why I believe this" receipt (evidence quote, from-session reference,
 * expand/collapse) was removed 2026-07-24 (mid-task amendment) — the
 * `evidence`/`from_session`/`confidence` fields it read no longer exist on
 * `WorkbenchEntry`. `section` is now shown inline, unconditionally, instead
 * of behind a toggle.
 *
 * Styling follows the Apple-design pass (2026-07-24): soft elevated cards,
 * feedback on press (scale, 100ms), size-specific tracking, motion respects
 * `prefers-reduced-motion` via Tailwind's `motion-reduce:` variants.
 *
 * The sentence renders as markdown (2026-07-24 revision — entries carry
 * backticked paths, bold, etc.) via `react-markdown`, same engine as chat
 * bubbles (MarkdownContent.tsx), but with a minimal, INLINE-only components
 * config: it lives inside the card's own clamped `<p>`, so a nested `<p>`
 * (react-markdown's default paragraph wrapper) is collapsed to a fragment
 * rather than reusing MarkdownContent's block-styled `.markdown-content`
 * wrapper, which would add block margins wrong for a one-line card sentence.
 * Raw HTML is never rendered — react-markdown escapes it by default and no
 * rehype-raw plugin is installed. Links stop propagation and open in a new
 * tab so a link click can't also fire the card's own doorway tap.
 */

import ReactMarkdown, { type Components } from 'react-markdown';
import { useT } from '../i18n/useT';

import type { WorkbenchEntry } from './workbench/types';

/** Inline-only markdown components for the card sentence — see the file
 *  docstring above. Module-scoped: none of these depend on props. */
const CARD_MARKDOWN_COMPONENTS: Components = {
  // Collapse the paragraph wrapper to a fragment — the entry text sits
  // inside the card's own clamped <p>, and a nested <p> would be invalid
  // HTML and pull in block margins.
  p: ({ children }) => <>{children}</>,
  code: ({ children }) => (
    <code className="rounded bg-sidebar px-[0.3em] py-[0.1em] font-mono text-[0.9em]">
      {children}
    </code>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => e.stopPropagation()}
      className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
    >
      {children}
    </a>
  ),
};

export interface WorkbenchCardProps {
  entry: WorkbenchEntry;
  /** Show the project name chip — global (unlensed) view only. */
  showProjectChip: boolean;
  projectName?: string | null;
  /** Whole-card tap (doorway): spawn/resume and navigate to the project. */
  onOpen: () => void;
  /** Fulfilled ("Done") or irrelevant ("Delete"). */
  onExit?: (kind: 'fulfilled' | 'irrelevant') => void;
}

function lateLabel(t: ReturnType<typeof useT>, n: number): string {
  return t(n === 1 ? 'workbench.age.late.one' : 'workbench.age.late.other', { n });
}

function waitingLabel(t: ReturnType<typeof useT>, n: number): string {
  return t(n === 1 ? 'workbench.age.waiting.one' : 'workbench.age.waiting.other', { n });
}

/** Age badge text: "N days late" when overdue, else "waiting N days". The
 *  late count always comes from the server's `days_late` — never recomputed
 *  from `due` client-side. Returns null when there's nothing to show. */
function ageLabel(t: ReturnType<typeof useT>, entry: WorkbenchEntry): string | null {
  const { overdue, age_days, days_late } = entry;
  if (overdue && days_late != null) {
    return lateLabel(t, days_late);
  }
  if (age_days != null) {
    return waitingLabel(t, age_days);
  }
  return null;
}

/** Filled primary pill — the card's ONE emphasized action. */
const PRIMARY_BTN =
  'rounded-full bg-accent px-3.5 py-1.5 text-[12.5px] font-medium text-white ' +
  'transition-[transform,background-color] duration-100 ease-out hover:bg-accent/90 ' +
  'active:scale-[0.96] motion-reduce:transition-none motion-reduce:active:scale-100';

/** Quiet text action (secondary: Delete). */
const QUIET_BTN =
  'rounded-full px-2.5 py-1.5 text-[12.5px] font-medium text-secondary ' +
  'transition-[transform,background-color,color] duration-100 ease-out ' +
  'hover:bg-card-hover hover:text-primary active:scale-[0.96] ' +
  'motion-reduce:transition-none motion-reduce:active:scale-100';

export default function WorkbenchCard({
  entry,
  showProjectChip,
  projectName,
  onOpen,
  onExit,
}: WorkbenchCardProps) {
  const t = useT();
  const age = ageLabel(t, entry);
  const testId = `workbench-card-entry-${entry.project_id}-${entry.id}`;

  return (
    <div
      role="button"
      tabIndex={0}
      data-testid={testId}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onOpen();
      }}
      className="group flex cursor-pointer flex-col gap-2 rounded-2xl border border-border/60 bg-card px-4 py-3.5 text-left shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.03)] transition-[transform,box-shadow,border-color] duration-150 ease-out hover:-translate-y-px hover:border-border hover:shadow-[0_2px_4px_rgba(0,0,0,0.05),0_8px_24px_rgba(0,0,0,0.05)] active:scale-[0.99] motion-reduce:transition-none motion-reduce:transform-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent/60"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="line-clamp-3 min-w-0 flex-1 text-[15px] leading-snug tracking-[-0.01em] text-primary">
          <ReactMarkdown components={CARD_MARKDOWN_COMPONENTS}>{entry.text}</ReactMarkdown>
        </p>
        {entry.overdue && (
          <span
            data-testid="workbench-card-overdue-badge"
            className="shrink-0 rounded-full bg-error/10 px-2 py-0.5 text-[11px] font-medium text-error"
          >
            {t('workbench.overdue')}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {showProjectChip && projectName && (
          <span
            data-testid="workbench-card-project-chip"
            className="rounded-full bg-sidebar px-2 py-0.5 text-[11px] font-medium text-secondary"
          >
            {projectName}
          </span>
        )}
        {entry.section && (
          // Raw string from PROJECT_STATE.md's `## ` heading — no i18n key
          // (not UI chrome, per the i18n convention for backend-authored
          // strings). No expander (2026-07-24 amendment removed the receipt).
          <span data-testid="workbench-card-section" className="font-mono text-[11px] text-muted">
            {entry.section}
          </span>
        )}
        {age && (
          <span className="font-mono text-[11px] tabular-nums text-muted">{age}</span>
        )}
      </div>

      <div className="mt-0.5 flex items-center gap-2">
        <button
          type="button"
          data-testid="workbench-card-exit-fulfilled"
          onClick={(e) => {
            e.stopPropagation();
            onExit?.('fulfilled');
          }}
          className={PRIMARY_BTN}
        >
          {t('workbench.exit.fulfilled')}
        </button>
        <button
          type="button"
          data-testid="workbench-card-exit-irrelevant"
          onClick={(e) => {
            e.stopPropagation();
            onExit?.('irrelevant');
          }}
          className={QUIET_BTN}
        >
          {t('workbench.exit.irrelevant')}
        </button>
      </div>
    </div>
  );
}

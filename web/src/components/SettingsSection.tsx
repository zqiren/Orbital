// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The structural primitives for a settings document.
 *
 * Project settings had grown to fourteen sections stacked in a flat
 * `space-y-6`, and every one of them announced itself with the SAME
 * `text-sm font-medium text-primary` label used by the individual fields
 * INSIDE it. "Sub-Agents" (a section) and "Sub-agent deployment
 * instructions" (one field within it) were typographically identical, and
 * the only thing separating one section from the next was 24px of air —
 * against 8–12px between a label and its own control. A 3:1 space ratio with
 * no type step and no rule is not a boundary the eye can find, which is what
 * "crowded and confusing" actually described.
 *
 * So there are three levels here, and each is doing one job:
 *
 *   SettingsGroup   an uppercase eyebrow UNDERLINED by a full-width rule —
 *                   the chapter marker. Answers "which part of the settings
 *                   am I in".
 *   SettingsSection a 15px semibold title, separated from its neighbours by
 *                   whitespace only — the landmark you scan for, now
 *                   unmistakably heavier than any field label.
 *   field labels    unchanged (`text-sm font-medium`); they simply stopped
 *                   competing with the headings once the headings grew.
 *
 * A section's `description` hides behind an info button on the title line
 * rather than sitting under it: fourteen sections each explaining themselves
 * in two lines of prose is its own wall of text, and the explanation is worth
 * exactly one click when wanted. Default state is titles and controls only.
 *
 * The affordance is a button rather than the `<details>` idiom used for the
 * LLM provider hints, for two reasons. A `<summary>` has to CONTAIN its
 * trigger, so the disclosure would have to swallow the whole title row — and
 * a clickable settings heading already means "collapse this section" on this
 * very page (LLM Provider, Fallback Models). And a stack of `<summary>Details`
 * rows, one per section, is the same repetition the description was. An ⓘ
 * next to the title costs no row and says only what it does.
 *
 * There is exactly ONE rule per chapter and it sits directly under that
 * chapter's heading, so the rules ARE the grouping — a reader finds the
 * chapter boundaries without reading a word. Ruling every section instead
 * (the first pass) put a line everywhere and therefore said nothing about
 * where a chapter began; sections inside a chapter separate on whitespace,
 * which is enough now that each carries a 15px semibold title.
 *
 * The rule is deliberately darker than `--color-border` (#E1E5EA, tuned for
 * field outlines). A rule that separates has to be seen from across the page;
 * at token strength it reads as a smudge rather than a boundary.
 *
 * Both primitives use `first:` resets rather than a `divider` prop so that a
 * conditionally-rendered section (`{!project.is_scratch && …}`) can vanish
 * without leaving a leading rule behind — a `false` child renders no element,
 * so `:first-child` lands on whatever actually mounted.
 */

import { useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';
import { useT } from '../i18n/useT';

interface SettingsGroupProps {
  /** Chapter label, already translated. */
  title: string;
  children: ReactNode;
}

/**
 * A named band of related sections: an underlined chapter heading followed by
 * its sections. The underline is the only rule on the page, so it is what
 * marks where one chapter ends and the next begins.
 */
export function SettingsGroup({ title, children }: SettingsGroupProps) {
  return (
    <section className="mt-14 first:mt-0">
      <h2 className="mb-6 border-b border-secondary/35 pb-2.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-secondary">
        {title}
      </h2>
      {/* The sections need their own parent so `first:mt-0` lands on the first
          one rather than being shadowed by the <h2> sibling. */}
      <div>{children}</div>
    </section>
  );
}

/**
 * The ⓘ that reveals a piece of explanatory prose. Shared by section headings
 * and by the field labels whose hints got the same treatment, so the page has
 * exactly one "there is more to read here" affordance.
 */
function InfoToggle({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const t = useT();
  return (
    <button
      // type="button" is load-bearing: this renders inside the settings
      // <form>, where a bare button submits it.
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={t('settings.section.about')}
      title={t('settings.section.about')}
      className={`shrink-0 transition-colors duration-150 ${
        open ? 'text-primary' : 'text-secondary/60 hover:text-primary'
      }`}
    >
      <Info size={13} aria-hidden="true" />
    </button>
  );
}

interface LabelWithHintProps {
  /** Id of the control this labels, so the label stays clickable. */
  htmlFor?: string;
  /** Label text, already translated. */
  children: ReactNode;
  /** The prose this hides; revealed by the ⓘ. */
  hint: ReactNode;
  /** Extra classes for the label itself. */
  className?: string;
}

/**
 * A field label whose hint is behind the same ⓘ as a section description.
 *
 * Without this the page would hide the long explanations at section level and
 * still print two-line paragraphs under individual field labels — which reads
 * as a bug, not a rule.
 */
export function LabelWithHint({ htmlFor, children, hint, className = '' }: LabelWithHintProps) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="flex items-center gap-1.5 mb-1.5">
        <label htmlFor={htmlFor} className={`text-sm font-medium text-primary ${className}`}>
          {children}
        </label>
        <InfoToggle open={open} onToggle={() => setOpen((v) => !v)} />
      </div>
      {open && <p className="text-xs leading-relaxed text-secondary mb-2">{hint}</p>}
    </>
  );
}

interface SettingsSectionProps {
  /** Rail anchor + deep-link target (`data-settings-section`). */
  id: string;
  /**
   * Section heading, already translated. Omitted by the two collapsible
   * sections (LLM Provider, Fallback Models) whose own disclosure button is
   * the heading — they render it at the same weight from inside.
   */
  title?: string;
  /** One line on what the section is for; sits under the title. */
  description?: ReactNode;
  /** Trailing adornment on the title line (a Beta badge, a qualifier). */
  suffix?: ReactNode;
  children: ReactNode;
}

export default function SettingsSection({
  id,
  title,
  description,
  suffix,
  children,
}: SettingsSectionProps) {
  const [showDescription, setShowDescription] = useState(false);
  return (
    <div
      data-settings-section={id}
      // scroll-mt clears the sticky header band when the rail jumps here.
      className="mt-9 scroll-mt-6 first:mt-0"
    >
      {title && (
        <div className="mb-3">
          <div className="flex items-center gap-1.5">
            <h3 className="text-[15px] font-semibold leading-6 text-primary">
              {title}
              {suffix}
            </h3>
            {description && (
              <InfoToggle
                open={showDescription}
                onToggle={() => setShowDescription((v) => !v)}
              />
            )}
          </div>
          {description && showDescription && (
            <p className="mt-1.5 text-[13px] leading-relaxed text-secondary">
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}

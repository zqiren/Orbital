// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ChevronRight, Clock } from 'lucide-react';
import type { Trigger } from '../types';
import { useT } from '../i18n/useT';

interface TriggerStripProps {
  triggers: Trigger[];
  /** Open the Automations pane of the Tasks tab. */
  onOpen: () => void;
}

/**
 * One-line automation status, above the tab bar.
 *
 * This used to be a second management surface — expandable rows with toggles,
 * a desktop detail panel, a mobile bottom sheet and a delete flow — parallel
 * to (and drifting from) the Automations pane. It is now purely glanceable:
 * how many automations exist, how many are switched off (a paused automation
 * is a silent failure mode, so that count must stay visible from every tab),
 * and a way into the single place they are managed.
 */
export default function TriggerStrip({ triggers, onOpen }: TriggerStripProps) {
  const t = useT();

  if (triggers.length === 0) return null;

  const offCount = triggers.filter((tr) => !tr.enabled).length;

  return (
    <div className="border-b border-border">
      <button
        type="button"
        onClick={onOpen}
        data-testid="trigger-summary"
        aria-label={t('trigger.strip.openAria')}
        className="flex w-full items-center gap-2 px-6 py-1.5 text-left text-xs text-secondary transition-colors duration-100 hover:bg-sidebar/50 max-md:min-h-[44px] max-md:px-4"
      >
        <Clock size={13} aria-hidden="true" className="shrink-0 text-muted" />
        <span className="text-primary">
          {t(triggers.length === 1 ? 'trigger.summary.one' : 'trigger.summary.other', {
            n: triggers.length,
          })}
        </span>
        {offCount > 0 && (
          <span className="text-muted">{t('trigger.summary.off', { n: offCount })}</span>
        )}
        <span className="ml-auto flex items-center gap-1 text-muted">
          {t('trigger.strip.manage')}
          <ChevronRight size={13} aria-hidden="true" className="shrink-0" />
        </span>
      </button>
    </div>
  );
}

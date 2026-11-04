// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * PrivateNotesCard — spec 153's first-screen fact in an imported project:
 * "Claude Code has 145 memories of this project. Every agent here can use
 * them now." The notes were folded into `orbital/imported/claude-code/` by
 * the deferred seed before this screen rendered, so there is no button to
 * share them and no consent step; the card is a fact from disk.
 *
 * It renders above the left-off cards on the empty-conversation screen when
 * GET /projects/{pid}/imported-notes reports notes and the project has at
 * least two available workers (with one, "every agent here" is false). Until
 * the fold has completed it shows only the count sentence. A click opens the
 * imported INDEX.md in the Files panel through the chat's path opener.
 *
 * Nothing renders while the answer loads or when it fails, so the left-off
 * cards below never jump.
 */

import { useEffect, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import {
  agentDisplayName,
  parseImportedNotes,
  showNotesCard,
  type ImportedNotes,
} from '../utils/onboarding';
import MessageAvatar from './MessageAvatar';

interface PrivateNotesCardProps {
  projectId: string;
  project: { imported?: boolean; available_workers?: string[] } | null | undefined;
  /** Installed sub-agents, for the display name. */
  agents?: ReadonlyArray<{ slug: string; name: string }>;
  onOpenPath?: (path: string) => void;
}

export default function PrivateNotesCard({ projectId, project, agents, onOpenPath }: PrivateNotesCardProps) {
  const t = useT();
  const [state, setState] = useState<{ projectId: string; notes: ImportedNotes | null } | null>(null);
  const imported = project?.imported === true;

  useEffect(() => {
    if (!imported) return;
    let cancelled = false;
    api<unknown>(`/api/v2/projects/${encodeURIComponent(projectId)}/imported-notes`)
      .then((raw) => { if (!cancelled) setState({ projectId, notes: parseImportedNotes(raw) }); })
      .catch(() => { if (!cancelled) setState({ projectId, notes: null }); });
    return () => { cancelled = true; };
  }, [projectId, imported]);

  const notes = state && state.projectId === projectId ? state.notes : null;
  if (!notes || !showNotesCard(notes, project)) return null;

  const n = notes.count;
  const agent = agentDisplayName(notes.agent, agents);
  const plural = n === 1 ? 'one' : 'other';
  const text = notes.folded
    ? t(`notes.card.full.${plural}`, { agent, n })
    : t(`notes.card.pending.${plural}`, { agent, n });

  return (
    <button
      type="button"
      data-testid="private-notes-card"
      data-folded={notes.folded ? 'true' : 'false'}
      onClick={() => onOpenPath?.(notes.index_path)}
      className="group mt-6 -mb-2 flex w-full items-center gap-3 rounded-xl border border-border bg-card px-3.5 py-3 text-left shadow-[0_1px_2px_rgb(0_0_0/0.04)] transition-[border-color,box-shadow,transform] duration-150 hover:border-accent/50 hover:shadow-[0_4px_12px_-4px_rgb(0_0_0/0.12)] active:scale-[0.99] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 motion-reduce:transition-none motion-reduce:active:scale-100"
    >
      <span className="shrink-0 [&>div]:h-8 [&>div]:w-8 [&>div]:rounded-lg [&>div]:p-1">
        <MessageAvatar variant="agent" agentHandle={notes.agent} />
      </span>
      <span className="min-w-0 flex-1 text-[13px] leading-[1.45] text-primary">{text}</span>
      <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-secondary transition-colors duration-150 group-hover:text-accent">
        {t('notes.card.open')}
        <ArrowRight size={12} aria-hidden="true" />
      </span>
    </button>
  );
}

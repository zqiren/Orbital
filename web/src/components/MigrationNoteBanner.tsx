// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The warning a project carries when its credential could not be turned into
 * a working card outright (spec 082 §3.7 / §3.9), rendered above the card
 * picker. The project still RUNS — on the global default card — so this is a
 * "you probably want to look at this", never a blocker.
 *
 * The note is a machine string `"<kind>:<detail>"` written by the migration or
 * by a card deletion. An unrecognised kind is shown verbatim rather than
 * swallowed: losing a warning is worse than showing a raw one.
 */
import { AlertTriangle } from 'lucide-react';
import type { CredentialCard } from '../types';
import { useT } from '../i18n/useT';

interface MigrationNoteBannerProps {
  note: string | null | undefined;
  /** Used only to name the card a `card_incomplete` note points at. */
  cards?: CredentialCard[];
}

export function parseMigrationNote(
  note: string | null | undefined,
): { kind: string; detail: string } | null {
  if (!note) return null;
  const idx = note.indexOf(':');
  if (idx === -1) return { kind: note, detail: '' };
  return { kind: note.slice(0, idx), detail: note.slice(idx + 1) };
}

export default function MigrationNoteBanner({ note, cards }: MigrationNoteBannerProps) {
  const t = useT();
  const parsed = parseMigrationNote(note);
  if (!parsed) return null;

  let text: string;
  switch (parsed.kind) {
    case 'card_incomplete':
      text = t('cards.note.incomplete', { name: parsed.detail });
      break;
    case 'needs_card':
      text = t('cards.note.needsCard', { setup: parsed.detail });
      break;
    case 'card_deleted':
      text = t('cards.note.deleted', { name: parsed.detail });
      break;
    default:
      text = note as string;
  }

  // A `card_incomplete` note names a card that still exists and only needs a
  // model; if it is gone the name is all we have, which the string covers.
  const named = parsed.kind === 'card_incomplete'
    ? cards?.find((c) => c.name === parsed.detail)
    : undefined;

  return (
    <div
      data-testid="migration-note"
      className="flex items-start gap-2 border border-warning/30 bg-warning/5 rounded-lg px-3 py-2 mb-3"
    >
      <AlertTriangle className="w-4 h-4 text-warning shrink-0 mt-0.5" />
      <p className="text-xs text-primary min-w-0">
        {text}
        {named && !named.model && (
          <span className="block text-secondary mt-0.5">{t('cards.note.editHint')}</span>
        )}
      </p>
    </div>
  );
}

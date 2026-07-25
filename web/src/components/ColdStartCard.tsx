// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';

interface ColdStartCardProps {
  folderName: string;
  onScan: () => void;
  onSkip: () => void;
  busy?: boolean;
  /** Translated error from a failed scan (e.g. missing API key). Scan stays
   * enabled so the user can retry after fixing Settings. */
  error?: string | null;
}

/**
 * First-session consent card for imported (non-empty) workspaces. Renders in
 * the empty chat view; spends no tokens until the user clicks Scan. Scan starts
 * the cold-start scan session; Skip dismisses the card for this view.
 */
export function ColdStartCard({ folderName, onScan, onSkip, busy, error }: ColdStartCardProps) {
  const t = useT();
  return (
    <div className="mb-3 rounded-lg border border-border bg-card overflow-hidden">
      <div className="bg-accent/10 px-4 py-2 border-b border-border">
        <span className="text-accent font-semibold text-sm">{t('coldStart.header')}</span>
      </div>
      <div className="px-4 py-3.5 space-y-3.5">
        <p className="text-sm text-secondary leading-relaxed">
          {t('coldStart.body', { folder: folderName })}
        </p>
        {error && (
          <p data-testid="cold-start-error" className="text-sm text-error" role="alert">
            {error}
          </p>
        )}
        <div className="flex gap-2">
          <button
            onClick={onScan}
            disabled={busy}
            className="bg-accent text-white text-sm font-medium rounded-lg px-4 py-1.5 hover:opacity-90 transition-opacity duration-150 max-md:min-h-[44px] disabled:opacity-50"
          >
            {busy ? t('coldStart.scanning') : t('coldStart.scan')}
          </button>
          <button
            onClick={onSkip}
            disabled={busy}
            className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-1.5 hover:bg-card-hover transition-colors duration-150 max-md:min-h-[44px] disabled:opacity-50"
          >
            {t('coldStart.skip')}
          </button>
        </div>
      </div>
    </div>
  );
}

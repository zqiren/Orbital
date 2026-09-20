// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import type { AutosaveStatus } from '../hooks/useAutosave';
import { useT } from '../i18n/useT';

/**
 * Where the Save button's confirmation used to be — except it follows the
 * user down the page, since the edit may be a card tile halfway down. Shows
 * while saving, briefly after, and stays with a Retry while a save failed.
 *
 * Shared by project and global settings. It is `sticky`, so it must be
 * rendered INSIDE the page's scroll container, after the document.
 */
export default function AutosaveStatusPill({
  status,
  error,
  onRetry,
}: {
  status: AutosaveStatus;
  error: string;
  onRetry: () => void;
}) {
  const t = useT();
  const [showSaved, setShowSaved] = useState(false);
  useEffect(() => {
    if (status !== 'saved') return;
    setShowSaved(true);
    const id = setTimeout(() => setShowSaved(false), 2000);
    return () => clearTimeout(id);
  }, [status]);

  if (status === 'idle' || (status === 'saved' && !showSaved)) return null;
  return (
    <div className="sticky bottom-4 flex justify-center pointer-events-none">
      <div
        role="status"
        data-testid="settings-autosave-status"
        className={`pointer-events-auto inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs shadow-sm ${
          status === 'error'
            ? 'border-error/40 bg-background text-error'
            : 'border-border bg-background text-secondary'
        }`}
      >
        {status === 'saving' && t('settings.autosave.saving')}
        {status === 'saved' && t('settings.saved')}
        {status === 'error' && (
          <>
            {t('settings.autosave.error', { message: error })}
            <button
              type="button"
              onClick={onRetry}
              data-testid="settings-autosave-retry"
              className="font-medium underline underline-offset-2"
            >
              {t('settings.autosave.retry')}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * FolderAccessSection — Project Settings → Folder access (spec 117 D8).
 *
 * Lists the folders granted to this project's sandbox (the access card's
 * grants, i.e. the provider's portal paths for the project) with a remove
 * wired to the existing `POST /api/v2/platform/folders/revoke`. A
 * safe-mode-only surface: with the sandbox off there is nothing to grant.
 *
 * Owns no state: SettingsView holds the list and drops a row once the
 * revoke has been accepted by the daemon.
 */

import { useState } from 'react';
import { useT } from '../i18n/useT';
import type { GrantedFolder } from '../types';

interface FolderAccessSectionProps {
  folders: GrantedFolder[];
  /** Resolves once the daemon accepted the revoke; a rejection keeps the row. */
  onRevoke: (folder: GrantedFolder) => Promise<void>;
}

export function FolderAccessSection({ folders, onRevoke }: FolderAccessSectionProps) {
  const t = useT();
  const [busyPath, setBusyPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function revoke(folder: GrantedFolder) {
    setBusyPath(folder.path);
    setError(null);
    try {
      await onRevoke(folder);
    } catch {
      setError(t('settings.folders.revokeError'));
    } finally {
      setBusyPath(null);
    }
  }

  return (
    // Heading + intro come from the enclosing SettingsSection.
    <div data-testid="folder-access-section">
      {folders.length === 0 ? (
        <p className="text-caption text-secondary italic">{t('settings.folders.empty')}</p>
      ) : (
        <div className="flex flex-col gap-1">
          {folders.map((folder) => (
            <div key={folder.path} className="flex items-center gap-2 py-1 max-md:min-h-[44px]">
              <span className="text-body font-mono text-primary flex-1 truncate" title={folder.path}>
                {folder.path}
              </span>
              <span className="text-caption text-secondary shrink-0">
                {folder.mode === 'read_write'
                  ? t('settings.folders.readWrite')
                  : t('settings.folders.readOnly')}
              </span>
              <button
                type="button"
                aria-label={t('settings.folders.removeAria', { path: folder.path })}
                disabled={busyPath === folder.path}
                onClick={() => revoke(folder)}
                className="text-caption font-medium text-error border border-error/40 rounded-lg px-2.5 py-1 hover:bg-error/5 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]"
              >
                {t('settings.folders.remove')}
              </button>
            </div>
          ))}
        </div>
      )}
      {error && <p className="text-caption text-error mt-2">{error}</p>}
    </div>
  );
}

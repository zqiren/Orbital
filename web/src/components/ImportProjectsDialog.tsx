// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ImportProjectsDialog — "Import projects from your agents" for users who are
 * past the wizard (spec 138). The wizard's Projects step as a dialog: the
 * same checklist (ImportProjectsStep), the same scan, the same one POST.
 */

import { useEffect, useId, useState } from 'react';
import { api } from '../config';
import { X } from 'lucide-react';
import { useT } from '../i18n/useT';
import { useOnboardingScan } from '../hooks/useOnboardingScan';
import { importIntro, type ImportResult } from '../utils/onboarding';
import ImportProjectsStep from './ImportProjectsStep';

interface ImportProjectsDialogProps {
  onClose: () => void;
  onImported: (result: ImportResult, landingProjectId: string | null) => void;
}

export default function ImportProjectsDialog({ onClose, onImported }: ImportProjectsDialogProps) {
  const t = useT();
  const titleId = useId();
  const scan = useOnboardingScan();
  // Spec 160: who reads the imported projects once (the sign beside
  // Import): Orbital with a key, else the user's own ready agent; else
  // nobody (round 3: no longer tied to the wizard's Skip).
  const [scanRunner, setScanRunner] = useState<{ key: boolean; agent: string | null } | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await api<{
          llm?: { api_key_set?: boolean };
              import_scan_agent?: { name: string } | null;
        }>('/api/v2/settings');
        if (cancelled || !data) return;
        if (data.llm?.api_key_set) setScanRunner({ key: true, agent: null });
        else if (data.import_scan_agent?.name) {
          setScanRunner({ key: false, agent: data.import_scan_agent.name });
        }
      } catch {
        /* no sign */
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="import-projects-dialog"
        className="flex max-h-[85vh] w-full max-w-xl flex-col surface-float-pop"
      >
        <div className="flex items-start gap-3 border-b border-border pt-5 pb-3 pl-6 pr-4">
          <div className="min-w-0 flex-1">
            <h2 id={titleId} className="text-[15px] font-semibold text-primary">{t('importProjects.entry')}</h2>
            <p className="mt-1 text-xs leading-relaxed text-secondary">
              {importIntro(t, scan.history?.sessions_total, scan.importable)}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label={t('importProjects.close')}
            data-testid="import-dialog-close"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-secondary transition-colors duration-150 hover:bg-surface hover:text-primary"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
        <ImportProjectsStep
          scan={scan}
          onSkip={onClose}
          skipLabel={t('importProjects.cancel')}
          onImported={onImported}
          scrollable
          scanRunner={scanRunner}
        />
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ArrowLeft } from 'lucide-react';
import PricingEditor from './PricingEditor';
import { useT } from '../i18n/useT';

interface PricingEditorPageProps {
  /** Return to the project's settings surface (the editor's only exit). */
  onBack: () => void;
  /**
   * Forwarded to PricingEditor: fired after a successful PUT so the host can
   * trigger the Budget cost surfaces to recompute against the new rates.
   */
  onSaved?: () => void;
}

/**
 * Modal-page shell for the pricing-table editor (P3-I). Mirrors
 * SettingsModalPage's header band + scrollable body so the editor feels like a
 * peer surface of project settings. Pricing is global (one daemon-level
 * overrides file), so this page carries no project context beyond the back nav.
 */
export default function PricingEditorPage({ onBack, onSaved }: PricingEditorPageProps) {
  const t = useT();
  return (
    <div className="flex flex-col flex-1 min-h-0 bg-background">
      {/* Header band */}
      <div className="flex flex-col gap-1 px-6 pt-5 pb-4 border-b border-border max-md:px-4">
        <button
          onClick={onBack}
          data-testid="pricing-back-button"
          className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-colors w-fit"
        >
          <ArrowLeft size={14} />
          {t('pricing.back')}
        </button>
        <h1 className="text-lg font-semibold text-primary mt-1" data-testid="pricing-title">
          {t('pricing.title')}
        </h1>
        <p className="text-sm text-secondary">{t('pricing.subtitle')}</p>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <PricingEditor onSaved={onSaved} />
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import PricingEditor from './PricingEditor';
import { useT } from '../i18n/useT';

interface PricingEditorPageProps {
  /**
   * Forwarded to PricingEditor: fired after a successful PUT so the host can
   * trigger the Budget cost surfaces to recompute against the new rates.
   */
  onSaved?: () => void;
}

/**
 * Page body for the pricing-table editor (P3-I): a page under Settings in the
 * spec 137 registry, so the shared toolbar carries Back (to Settings' Budget
 * section) and the crumb. Pricing is global (one daemon-level overrides
 * file), so this page carries no project context.
 */
export default function PricingEditorPage({ onSaved }: PricingEditorPageProps) {
  const t = useT();
  return (
    <div className="flex flex-col h-full min-h-0 bg-background">
      {/* Spec 137: Back and the title ride the shared page toolbar
          (`Project / Settings / Pricing table`); only the explainer stays. */}
      <p className="shrink-0 px-6 pt-4 pb-3 text-body text-secondary border-b border-border max-md:px-4">
        {t('pricing.subtitle')}
      </p>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <PricingEditor onSaved={onSaved} />
      </div>
    </div>
  );
}

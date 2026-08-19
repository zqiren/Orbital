// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * NetworkAccessSection — the Project Settings → Network access section
 * (TOFU network allowlist, Plan 2 Task 7).
 *
 * Shows the domains this project's sandboxed shell/sub-agents may reach
 * beyond the built-in defaults, plus any TOFU requests awaiting a decision
 * (asked via `request_network_access`, Plan 2 Task 3, and auto-denied into
 * this list under hands-off autonomy, Task 5). All edits are staged locally
 * through `onChange` — like ProjectConnectorToggles, this component owns no
 * state itself; SettingsView holds it and persists both fields together on
 * the form's existing Save (Task 2 applies the resulting grants server-side).
 */

import { useT } from '../i18n/useT';
import type { PendingDomainRequest } from '../types';

interface NetworkAccessSectionProps {
  approvedDomains: string[];
  pendingRequests: PendingDomainRequest[];
  onChange: (next: { approvedDomains: string[]; pendingRequests: PendingDomainRequest[] }) => void;
}

export function NetworkAccessSection({
  approvedDomains,
  pendingRequests,
  onChange,
}: NetworkAccessSectionProps) {
  const t = useT();

  const remove = (domain: string) =>
    onChange({
      approvedDomains: approvedDomains.filter((d) => d !== domain),
      pendingRequests,
    });

  const approve = (req: PendingDomainRequest) =>
    onChange({
      approvedDomains: approvedDomains.includes(req.domain)
        ? approvedDomains
        : [...approvedDomains, req.domain],
      pendingRequests: pendingRequests.filter((r) => r.domain !== req.domain),
    });

  const dismiss = (req: PendingDomainRequest) =>
    onChange({
      approvedDomains,
      pendingRequests: pendingRequests.filter((r) => r.domain !== req.domain),
    });

  return (
    // Heading + intro come from the enclosing SettingsSection.
    <div data-testid="network-access-section">
      {pendingRequests.length > 0 && (
        <div className="mb-3">
          <h4 className="text-xs font-semibold text-primary mb-1.5">
            {t('settings.network.pendingTitle')}
          </h4>
          <div className="flex flex-col gap-1">
            {pendingRequests.map((req) => (
              <div
                key={req.domain}
                className="flex items-center gap-2 py-1 max-md:min-h-[44px]"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-primary truncate">{req.domain}</span>
                  <span className="text-xs text-secondary truncate ml-2">{req.reason}</span>
                </div>
                <button
                  type="button"
                  aria-label={t('settings.network.approveAria', { domain: req.domain })}
                  onClick={() => approve(req)}
                  className="text-xs font-medium text-accent border border-border rounded-lg px-2.5 py-1 hover:bg-accent/5 transition-all duration-150 max-md:min-h-[44px]"
                >
                  {t('settings.network.approve')}
                </button>
                <button
                  type="button"
                  aria-label={t('settings.network.dismissAria', { domain: req.domain })}
                  onClick={() => dismiss(req)}
                  className="text-xs font-medium text-secondary border border-border rounded-lg px-2.5 py-1 hover:bg-secondary/5 transition-all duration-150 max-md:min-h-[44px]"
                >
                  {t('settings.network.dismiss')}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {approvedDomains.length === 0 ? (
        <p className="text-xs text-secondary/60 italic">{t('settings.network.empty')}</p>
      ) : (
        <div className="flex flex-col gap-1">
          {approvedDomains.map((domain) => (
            <div key={domain} className="flex items-center gap-2 py-1 max-md:min-h-[44px]">
              <span className="text-sm text-primary flex-1 truncate">{domain}</span>
              <button
                type="button"
                aria-label={t('settings.network.removeAria', { domain })}
                onClick={() => remove(domain)}
                className="text-xs font-medium text-error border border-error/40 rounded-lg px-2.5 py-1 hover:bg-error/5 transition-all duration-150 max-md:min-h-[44px]"
              >
                {t('settings.network.remove')}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

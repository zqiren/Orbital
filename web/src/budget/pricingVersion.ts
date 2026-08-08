// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Module-level pricing-table version. Bumped after a pricing-table edit
 * (PUT /pricing/overrides) so every mounted useCost re-reads /cost under the
 * NEW rates. A rate change fires no WS event (that's ledger-append only), and
 * the old prop-drilled refreshKey reached only the settings Budget meter —
 * BudgetCorner and QueueTab kept showing costs computed under the old rates
 * until the next spend event (backlog #40). useCost subscribes here via
 * useSyncExternalStore, so all consumers recompute together.
 */

let version = 0;
const listeners = new Set<() => void>();

export function getPricingVersion(): number {
  return version;
}

export function subscribePricingVersion(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Signal that pricing rates changed; all useCost consumers re-fetch. */
export function bumpPricingVersion(): void {
  version += 1;
  listeners.forEach((l) => l());
}

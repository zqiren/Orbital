// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Why an installed agent is not ready, in a few words (2026-11-25, honest
 * status). The daemon decides it (`not_ready: {reason, detail}` on the
 * agent lists, from setup_types.not_ready); this maps the reason to its
 * short label. `detail` is the vendor's or the agent's own words, shown
 * behind a hover sign, never translated. An unknown reason (a manifest may
 * declare new refusal reasons) still says "unavailable", never nothing.
 */

import type { StringKey } from '../i18n/strings';

export interface NotReady {
  reason: string;
  detail?: string | null;
}

const LABEL: Record<string, StringKey> = {
  auth_expired: 'agentIssue.authExpired',
  handshake_failed: 'agentIssue.cantConnect',
  key_missing: 'agentIssue.keyMissing',
  missing_dependency: 'agentIssue.missingDependency',
  region_unavailable: 'agentIssue.regionUnavailable',
  signed_out: 'agentIssue.signedOut',
  update_needed: 'agentIssue.needsUpdate',
};

/** The label key for a reason; anything else reads "unavailable". */
export function issueLabelKey(reason: string | null | undefined): StringKey {
  return (reason && LABEL[reason]) || 'agentIssue.unavailable';
}

/** Reasons a fresh check can change: a refused turn, a failed handshake, a
 *  reason the UI does not know. Offered "Check again". */
export function issueRechecks(reason: string | null | undefined): boolean {
  return !reason || !(reason in LABEL) || reason === 'region_unavailable'
    || reason === 'handshake_failed';
}

/** The daemon's `not_ready` when it is a usable object, else null. */
export function notReadyOf(value: unknown): NotReady | null {
  if (!value || typeof value !== 'object') return null;
  const reason = (value as { reason?: unknown }).reason;
  if (typeof reason !== 'string' || !reason) return null;
  const detail = (value as { detail?: unknown }).detail;
  return { reason, detail: typeof detail === 'string' && detail ? detail : null };
}

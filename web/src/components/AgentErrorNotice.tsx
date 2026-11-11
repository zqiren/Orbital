// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * AgentErrorNotice — in-chat surface for classified agent/provider errors
 * (credential-error surfacing). Rendered in the ChatView timeline when the
 * agent loop errors (agent.status error broadcast) or when hydration finds a
 * last_terminal_event error on run-status. Previously these errors were
 * dropped (red status dot only), so users with a missing/wrong API key saw
 * the agent silently do nothing and blamed Orbital.
 *
 * The headline is the translated, actionable message for the classifier
 * `code`; the raw backend `message` is shown untranslated as secondary detail
 * (dynamic provider strings are not UI chrome — see i18n rules).
 *
 * Spec 141: `internal_error` is Orbital's own crash, so there is nothing to
 * fix in Settings — the notice offers Send feedback (the spec-113 dialog,
 * prefilled with the raw detail) instead of Open settings.
 */

import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { useT } from '../i18n/useT';
import { isInternalError, providerErrorKey, readableProviderDetail } from '../utils/providerError';
import { SendFeedbackDialog } from './SendFeedback';

export interface AgentErrorNoticeProps {
  /** Stable classifier code (missing_api_key, invalid_api_key, …). */
  code?: string;
  /** Raw backend detail message, shown untranslated below the headline. */
  message?: string;
  /** Open the Global Settings surface (where the API key lives). */
  onOpenSettings: () => void;
  onDismiss: () => void;
}

export default function AgentErrorNotice({
  code,
  message,
  onOpenSettings,
  onDismiss,
}: AgentErrorNoticeProps) {
  const t = useT();
  const internal = isInternalError(code);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  // The provider's sentence, not the SDK payload around it (1111 pass 2).
  // Send feedback still carries the raw detail.
  const detail = message ? readableProviderDetail(message) : '';
  return (
    <div
      data-testid="agent-error-notice"
      className="mb-3 surface-float-static px-4 py-3"
      role="alert"
      aria-live="polite"
    >
      <div className="flex items-start gap-2">
        <AlertCircle size={14} className="mt-0.5 shrink-0 text-error" />
        <div className="min-w-0">
          <p className="text-sm text-primary">{t(providerErrorKey(code))}</p>
          {detail && (
            <p className="mt-1 text-xs text-secondary break-words">{detail}</p>
          )}
        </div>
      </div>
      <div className="mt-3 flex gap-2">
        {internal ? (
          <button
            type="button"
            data-testid="agent-error-notice-feedback"
            onClick={() => setFeedbackOpen(true)}
            className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide bg-accent text-white hover:bg-accent/85 transition-colors duration-150"
          >
            {t('feedback.button')}
          </button>
        ) : (
          <button
            type="button"
            data-testid="agent-error-notice-settings"
            onClick={onOpenSettings}
            className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide bg-accent text-white hover:bg-accent/85 transition-colors duration-150"
          >
            {t('agentError.openSettings')}
          </button>
        )}
        <button
          type="button"
          data-testid="agent-error-notice-dismiss"
          onClick={onDismiss}
          className="shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide border border-border bg-background text-secondary hover:bg-card-hover transition-colors duration-150"
        >
          {t('agentError.dismiss')}
        </button>
      </div>
      {feedbackOpen && (
        <SendFeedbackDialog
          context={{ kind: 'agent_error', ...(message ? { detail: message } : {}) }}
          onClose={() => setFeedbackOpen(false)}
        />
      )}
    </div>
  );
}

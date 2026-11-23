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
 *
 * Batch 1123: `agent_signed_out` names the agent that needs a sign-in and
 * offers Sign in (the in-app login job) instead of Open settings; the notice
 * goes once the sign-in completes.
 */

import { useCallback, useState } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { useT } from '../i18n/useT';
import { useAgentLogin } from '../hooks/useAgentLogin';
import { refreshAgentAvailability } from '../hooks/useAgentAvailability';
import { isInternalError, providerErrorKey, readableProviderDetail } from '../utils/providerError';
import LoginCodeField from './LoginCodeField';
import { SendFeedbackDialog } from './SendFeedback';

export interface AgentErrorNoticeProps {
  /** Stable classifier code (missing_api_key, invalid_api_key, …). */
  code?: string;
  /** Raw backend detail message, shown untranslated below the headline. */
  message?: string;
  /** `agent_signed_out`: the agent's slug and display name. */
  agent?: string;
  agentName?: string;
  /** Open the Global Settings surface (where the API key lives). */
  onOpenSettings: () => void;
  onDismiss: () => void;
}

/** Sign in for `agent_signed_out`: the login job, then the notice goes. */
function SignInButton({ agent, onSignedIn }: { agent: string; onSignedIn: () => void }) {
  const t = useT();
  const login = useAgentLogin(agent, onSignedIn);
  return (
    <>
      <button
        type="button"
        data-testid="agent-error-notice-signin"
        disabled={login.busy}
        onClick={() => void login.start()}
        className="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1 rounded-md text-caption font-medium bg-accent text-white hover:bg-accent/85 transition-colors duration-150 disabled:opacity-70"
      >
        {login.busy && <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />}
        {login.busy ? t('subAgentCard.startingLogin') : t('wizard.agents.signIn')}
      </button>
      {login.busy && login.url && (
        <a href={login.url} target="_blank" rel="noreferrer" className="self-center text-caption text-accent-text underline">
          {t('subAgentCard.loginOpenLink')}
        </a>
      )}
      {login.failed && <span className="self-center text-caption text-error-text" role="alert">{login.failed}</span>}
      {/* A login that asks for the code its browser page shows (Antigravity):
          its own line under the buttons; nothing for other agents. */}
      {login.busy && <LoginCodeField slug={agent} className="order-last basis-full" />}
    </>
  );
}

export default function AgentErrorNotice({
  code,
  message,
  agent,
  agentName,
  onOpenSettings,
  onDismiss,
}: AgentErrorNoticeProps) {
  const t = useT();
  const internal = isInternalError(code);
  const signedOut = code === 'agent_signed_out' && !!agent;
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const onSignedIn = useCallback(() => {
    refreshAgentAvailability();
    onDismiss();
  }, [onDismiss]);
  // The provider's sentence, not the SDK payload around it (1111 pass 2).
  // Send feedback still carries the raw detail. The signed-out headline
  // already says everything the daemon's sentence does.
  const detail = message && !signedOut ? readableProviderDetail(message) : '';
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
          <p className="text-body text-primary">
            {signedOut
              ? t('agentError.agent_signed_out', { agent: agentName || agent })
              : t(providerErrorKey(code))}
          </p>
          {detail && (
            <p className="mt-1 text-caption text-secondary break-words">{detail}</p>
          )}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {signedOut ? (
          <SignInButton agent={agent} onSignedIn={onSignedIn} />
        ) : internal ? (
          <button
            type="button"
            data-testid="agent-error-notice-feedback"
            onClick={() => setFeedbackOpen(true)}
            className="shrink-0 px-2.5 py-1 rounded-md text-caption font-medium bg-accent text-white hover:bg-accent/85 transition-colors duration-150"
          >
            {t('feedback.button')}
          </button>
        ) : (
          <button
            type="button"
            data-testid="agent-error-notice-settings"
            onClick={onOpenSettings}
            className="shrink-0 px-2.5 py-1 rounded-md text-caption font-medium bg-accent text-white hover:bg-accent/85 transition-colors duration-150"
          >
            {t('agentError.openSettings')}
          </button>
        )}
        <button
          type="button"
          data-testid="agent-error-notice-dismiss"
          onClick={onDismiss}
          className="shrink-0 px-2.5 py-1 rounded-md text-caption font-medium border border-border bg-background text-secondary hover:bg-card-hover transition-colors duration-150"
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

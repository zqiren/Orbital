// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * LoginCodeField — the one field a paste-the-code login needs (spec 193).
 *
 * Some CLIs finish their OAuth by asking for the code the browser page shows
 * (`agy acp --login` opens Google sign-in itself, then reads the code from
 * stdin). The daemon emits `login.input_needed` when the CLI's manifest-
 * declared prompt appears; this field shows for that job and posts the value
 * to `.../login-jobs/{job_id}/input`, which writes it to the CLI. The job
 * then ends like any login (`login.complete` / `login.failed`), which hides
 * the field. Orbital never opens the sign-in URL itself: a second tab's code
 * fails the CLI's PKCE check.
 *
 * Shared by the Settings agent card and the onboarding agents step, so both
 * login buttons can finish such a login.
 */

import { useEffect, useState } from 'react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useWebSocket } from '../hooks/useWebSocket';
import type { WebSocketEvent } from '../types';

export default function LoginCodeField({ slug, className = '' }: { slug: string; className?: string }) {
  const t = useT();
  const ws = useWebSocket();
  const [jobId, setJobId] = useState<string | null>(null);
  const [value, setValue] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onNeeded = (event: WebSocketEvent) => {
      if (event.type !== 'login.input_needed' || event.slug !== slug || event.kind === 'update') return;
      setJobId(event.job_id);
      setValue('');
      setError(null);
    };
    const onEnded = (event: WebSocketEvent) => {
      if (event.type !== 'login.complete' && event.type !== 'login.failed') return;
      if (event.slug !== slug || event.kind === 'update') return;
      setJobId(null);
      setValue('');
      setError(null);
    };
    ws.on('login.input_needed', onNeeded);
    ws.on('login.complete', onEnded);
    ws.on('login.failed', onEnded);
    return () => {
      ws.off('login.input_needed', onNeeded);
      ws.off('login.complete', onEnded);
      ws.off('login.failed', onEnded);
    };
  }, [ws, slug]);

  if (!jobId) return null;

  const submit = async () => {
    const code = value.trim();
    if (!code || sending) return;
    setSending(true);
    setError(null);
    try {
      await api(
        `/api/v2/settings/sub-agents/${encodeURIComponent(slug)}/login-jobs/${encodeURIComponent(jobId)}/input`,
        { method: 'POST', body: JSON.stringify({ value: code }) },
      );
      // Sent: the CLI finishes the exchange and the job's terminal event
      // reports the outcome. The field has done its job.
      setJobId(null);
      setValue('');
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : t('subAgentCard.loginFailed'));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className={`flex flex-col gap-1.5 ${className}`} data-testid={`login-code-${slug}`}>
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void submit();
            }
          }}
          placeholder={t('subAgentCard.loginCodePlaceholder')}
          aria-label={t('subAgentCard.loginCodePlaceholder')}
          autoComplete="off"
          spellCheck={false}
          autoFocus
          data-testid={`login-code-input-${slug}`}
          className="min-w-0 flex-1 text-sm font-mono bg-sidebar border border-border rounded px-2 py-1.5 text-primary focus:outline-none focus:border-accent"
        />
        <button
          type="button"
          onClick={() => void submit()}
          disabled={sending || !value.trim()}
          data-testid={`login-code-submit-${slug}`}
          className="shrink-0 text-xs bg-accent text-white rounded px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
        >
          {sending ? t('subAgentCard.sending') : t('subAgentCard.submit')}
        </button>
      </div>
      {error && (
        <p className="text-[11px] text-error-text" role="alert">{error}</p>
      )}
    </div>
  );
}

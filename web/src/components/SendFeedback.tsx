// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState, type ReactNode } from 'react';
import { useT } from '../i18n/useT';
import { FIELD_MULTILINE } from './fieldStyles';

/**
 * Send feedback with diagnostics (spec 113 part D).
 *
 * One dialog, two entry points: the Global Settings section (default export)
 * and `SendDiagnosticsLink`, a small text link the login-failed and
 * dispatch-failed notices drop in with their context. Both open the same
 * `SendFeedbackDialog`.
 *
 * Consent is the preview. The dialog fetches the daemon's sanitised bundle
 * and shows it in full with its size; Send is disabled until that has
 * loaded. Send posts only the message and the context — the daemon rebuilds
 * and re-sanitises the bundle itself, so the client can never upload text
 * the user did not see. No retry, no queue: a failure shows one line and
 * keeps the message.
 *
 * Follows GlobalSettings' local raw-fetch convention.
 */

const API_BASE = import.meta.env.VITE_API_BASE || '';

export interface DiagnosticsContext {
  kind: 'login_failed' | 'dispatch_failed' | 'agent_error' | 'general';
  agent?: string;
  detail?: string;
}

interface PreviewResponse {
  text: string;
  bytes: number;
  telemetry_enabled: boolean;
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function previewUrl(context: DiagnosticsContext | undefined): string {
  const url = `${API_BASE}/api/v2/feedback/preview`;
  if (!context || context.kind === 'general') return url;
  const params = new URLSearchParams({ kind: context.kind });
  if (context.agent) params.set('agent', context.agent);
  if (context.detail) params.set('detail', context.detail);
  return `${url}?${params.toString()}`;
}

type SendState =
  | { status: 'idle' }
  | { status: 'sending' }
  | { status: 'sent'; bytes: number }
  | { status: 'failed' };

export function SendFeedbackDialog({
  context,
  onClose,
}: {
  context?: DiagnosticsContext;
  onClose: () => void;
}) {
  const t = useT();
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState(false);
  const [message, setMessage] = useState('');
  const [send, setSend] = useState<SendState>({ status: 'idle' });

  useEffect(() => {
    let cancelled = false;
    fetch(previewUrl(context))
      .then((r) => r.json())
      .then((data: PreviewResponse) => {
        if (!cancelled) setPreview(data);
      })
      .catch(() => {
        if (!cancelled) setPreviewError(true);
      });
    return () => {
      cancelled = true;
    };
    // The context is fixed for the dialog's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  async function handleSend() {
    if (!preview || send.status === 'sending') return;
    setSend({ status: 'sending' });
    try {
      const body: Record<string, unknown> = { message };
      if (context && context.kind !== 'general') {
        body.context = { kind: context.kind, ...(context.agent ? { agent: context.agent } : {}),
          ...(context.detail ? { detail: context.detail } : {}) };
      } else if (context) {
        body.context = { kind: 'general' };
      }
      const r = await fetch(`${API_BASE}/api/v2/feedback/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { ok?: boolean; bytes?: number };
      if (!data.ok) throw new Error('not ok');
      setSend({ status: 'sent', bytes: data.bytes ?? preview.bytes });
    } catch {
      setSend({ status: 'failed' }); // the message stays in the textarea
    }
  }

  let contextLine: ReactNode = null;
  if (context?.kind === 'login_failed') {
    contextLine = t('feedback.context.login_failed', { agent: context.agent ?? '' });
  } else if (context?.kind === 'dispatch_failed') {
    contextLine = t('feedback.context.dispatch_failed', { agent: context.agent ?? '' });
  } else if (context?.kind === 'agent_error') {
    contextLine = t('feedback.context.agent_error');
  }

  const sent = send.status === 'sent';

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
        aria-label={t('feedback.button')}
        data-testid="feedback-dialog"
        className="w-full max-w-2xl max-h-[85vh] flex flex-col surface-float-pop"
      >
        <div className="px-5 pt-4 pb-2">
          <h2 className="text-[15px] font-semibold text-primary">{t('feedback.button')}</h2>
          {contextLine && (
            <p className="mt-1 text-xs text-secondary" data-testid="feedback-context">
              {contextLine}
            </p>
          )}
        </div>

        <div className="px-5 pb-4 overflow-y-auto flex-1 min-h-0 space-y-4">
          {sent ? (
            <p className="text-sm text-primary" data-testid="feedback-sent">
              {t('feedback.sent', { size: formatSize(send.bytes) })}
            </p>
          ) : (
            <>
              <div>
                <label className="block text-sm font-medium text-primary mb-1" htmlFor="feedback-message">
                  {t('feedback.message.label')}
                </label>
                <textarea
                  id="feedback-message"
                  data-testid="feedback-message"
                  rows={3}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder={t('feedback.message.placeholder')}
                  className={`w-full resize-y ${FIELD_MULTILINE}`}
                />
              </div>

              <div>
                <h3 className="text-sm font-medium text-primary mb-1">
                  {t('feedback.preview.heading')}
                </h3>
                {previewError ? (
                  <p className="text-xs text-error">{t('feedback.preview.error')}</p>
                ) : preview === null ? (
                  <p className="text-xs text-secondary">{t('feedback.preview.loading')}</p>
                ) : (
                  <>
                    <pre
                      data-testid="feedback-preview"
                      className="text-[11px] font-mono bg-sidebar border border-border rounded-lg p-3 max-h-64 overflow-auto whitespace-pre-wrap break-all"
                    >
                      {preview.text}
                    </pre>
                    <p className="mt-1.5 text-xs text-secondary" data-testid="feedback-size">
                      {t('feedback.size', { size: formatSize(preview.bytes) })}
                    </p>
                    {!preview.telemetry_enabled && (
                      <p className="mt-1 text-xs text-secondary" data-testid="feedback-telemetry-off">
                        {t('feedback.telemetryOff')}
                      </p>
                    )}
                  </>
                )}
              </div>

              {send.status === 'failed' && (
                <p className="text-xs text-error" role="alert" data-testid="feedback-failed">
                  {t('feedback.failed')}
                </p>
              )}
            </>
          )}
        </div>

        <div className="px-5 py-3 border-t border-border flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="text-sm px-3 py-1.5 rounded-lg border border-border text-primary hover:bg-sidebar transition-colors"
          >
            {sent ? t('feedback.close') : t('feedback.cancel')}
          </button>
          {!sent && (
            <button
              type="button"
              data-testid="feedback-send"
              onClick={() => void handleSend()}
              disabled={preview === null || send.status === 'sending'}
              className="text-sm px-3 py-1.5 rounded-lg bg-accent text-white disabled:opacity-50 transition-colors"
            >
              {send.status === 'sending' ? t('feedback.sending') : t('feedback.send')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * The link the failure notices render (SubAgentSettings' login-failed notice,
 * ChatView's dispatch-failed notice). Text only; it opens the dialog with
 * the failure's context so the bundle header says what the user was doing.
 */
export function SendDiagnosticsLink({
  context,
  className,
}: {
  context: DiagnosticsContext;
  className?: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        data-testid="feedback-diagnostics-link"
        className={className ?? 'text-xs underline underline-offset-2 text-accent hover:opacity-80'}
      >
        {t('feedback.link')}
      </button>
      {open && <SendFeedbackDialog context={context} onClose={() => setOpen(false)} />}
    </>
  );
}

/** Global Settings → Device & privacy → Send feedback. */
export default function SendFeedback() {
  const t = useT();
  const [open, setOpen] = useState(false);
  return (
    <div className="flex items-start justify-between gap-4">
      <p className="text-xs text-secondary min-w-0">{t('feedback.intro')}</p>
      <button
        type="button"
        data-testid="feedback-open"
        onClick={() => setOpen(true)}
        className="shrink-0 text-sm px-3 py-1.5 rounded-lg border border-border text-primary hover:bg-sidebar transition-colors"
      >
        {t('feedback.button')}
      </button>
      {open && <SendFeedbackDialog onClose={() => setOpen(false)} />}
    </div>
  );
}

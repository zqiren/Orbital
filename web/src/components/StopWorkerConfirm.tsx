// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The "stop this agent?" confirm, shared by the worker tiles under the
 * session title (SubAgentStatusBar) and the docked worker panel's Stop
 * (spec 195). Portalled to <body> (spec 169 C): inside an overflow box the
 * packaged app's WKWebView clipped the backdrop to that box.
 */
import { createPortal } from 'react-dom';
import { useT } from '../i18n/useT';

interface Props {
  name: string;
  /** The turn is running (else: only background work is alive). */
  running: boolean;
  backgroundCommands?: string[];
  onCancel: () => void;
  onConfirm: () => void;
}

export default function StopWorkerConfirm({ name, running, backgroundCommands = [], onCancel, onConfirm }: Props) {
  const t = useT();
  return createPortal(
    <div
      data-testid="sub-agent-stop-confirm"
      className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-sm surface-float-pop p-4 text-body text-primary"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="font-medium mb-2">{t('subAgentBar.confirm.title', { name })}</div>
        <p className="text-secondary mb-2">
          {running ? t('subAgentBar.confirm.running') : t('subAgentBar.confirm.background')}
        </p>
        {!running && backgroundCommands.length > 0 && (
          <ul className="mb-2 max-h-24 overflow-y-auto rounded bg-background p-2 font-mono text-micro text-warning-text">
            {backgroundCommands.map((c, i) => (
              <li key={i} className="truncate">{c}</li>
            ))}
          </ul>
        )}
        <p data-testid="sub-agent-stop-warning" className="text-micro text-secondary mb-3">
          {t('subAgentBar.confirm.warning')}
        </p>
        <div className="flex justify-end gap-2">
          <button
            data-testid="sub-agent-stop-cancel"
            onClick={onCancel}
            className="rounded px-3 py-1.5 text-secondary hover:bg-card-hover"
          >
            {t('subAgentBar.confirm.cancel')}
          </button>
          <button
            data-testid="sub-agent-stop-confirm-button"
            onClick={onConfirm}
            className="rounded bg-error px-3 py-1.5 text-white hover:opacity-90"
          >
            {t('subAgentBar.confirm.stop')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

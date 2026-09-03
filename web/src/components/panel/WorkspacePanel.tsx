// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5 — the panel body: a Files | Browser switch, an Annotate button,
 * and the selected view. No tabs beyond the two views, no other chrome (D7).
 * CONTRACT FILE: props are final.
 */
import type { ReactNode } from 'react';
import type { PanelView } from '../../utils/panelSelectors';
import { useT } from '../../i18n/useT';

export interface WorkspacePanelProps {
  view: PanelView;
  onViewChange: (v: PanelView) => void;
  annotating: boolean;
  onToggleAnnotate: () => void;
  browser: ReactNode;
  files: ReactNode;
}

export default function WorkspacePanel({
  view,
  onViewChange,
  annotating,
  onToggleAnnotate,
  browser,
  files,
}: WorkspacePanelProps) {
  const t = useT();

  const tab = (key: PanelView, label: string) => {
    const active = view === key;
    return (
      <button
        key={key}
        type="button"
        role="tab"
        aria-selected={active}
        data-testid={`panel-view-${key}`}
        onClick={() => onViewChange(key)}
        className={`text-[11px] font-medium px-2.5 py-1 rounded-md transition-colors duration-150 ${
          active
            ? 'bg-card text-primary shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
            : 'text-secondary hover:text-primary'
        }`}
      >
        {label}
      </button>
    );
  };

  return (
    <div className="flex flex-col h-full min-h-0" data-testid="workspace-panel">
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-border shrink-0">
        <div
          role="tablist"
          aria-label={t('panel.handle')}
          className="inline-flex gap-0.5 rounded-lg border border-border bg-nav p-0.5"
        >
          {tab('files', t('panel.view.files'))}
          {tab('browser', t('panel.view.browser'))}
        </div>
        <button
          type="button"
          aria-pressed={annotating}
          data-testid="panel-annotate"
          onClick={onToggleAnnotate}
          className={`text-[11px] font-medium px-2.5 py-1 rounded-md border transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
            annotating
              ? 'border-accent bg-accent/10 text-accent'
              : 'border-border text-secondary hover:text-primary hover:bg-card-hover'
          }`}
        >
          {annotating ? t('panel.annotate.done') : t('panel.annotate')}
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {view === 'browser' ? browser : files}
      </div>
    </div>
  );
}

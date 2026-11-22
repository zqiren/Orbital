// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5 — the panel: a Files | Browser switch, an Annotate button, and
 * the selected view. No tabs beyond the two views, no other chrome (D7).
 *
 * Spec 088 splits the bar from the body: `WorkspacePanelBar` is handed to the
 * docked drawer's header slot, so the switch shares one row with collapse
 * instead of stacking a second bar under an empty one.
 *
 * Spec 137 adds two icon controls on the right: Upload (Files view) and
 * Full width / Restore. Maximize is what replaced the Files tab: maximized
 * Files is the tab's old layout, tree on the left and the file on the right.
 */
import type { ReactNode } from 'react';
import { Maximize2, Minimize2, Upload } from 'lucide-react';
import type { PanelView } from '../../utils/panelSelectors';
import { useT } from '../../i18n/useT';

export interface WorkspacePanelBarProps {
  view: PanelView;
  onViewChange: (v: PanelView) => void;
  annotating: boolean;
  onToggleAnnotate: () => void;
  /** Files view: pick a file to upload to the workspace. Absent = no button. */
  onUpload?: () => void;
  uploading?: boolean;
  /** Full width ⇄ restore. Absent = no control (the phone's bottom sheet). */
  onToggleMaximize?: () => void;
  maximized?: boolean;
}

const ICON_BUTTON =
  'flex items-center justify-center w-7 h-7 shrink-0 rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50';

export function WorkspacePanelBar({
  view,
  onViewChange,
  annotating,
  onToggleAnnotate,
  onUpload,
  uploading = false,
  onToggleMaximize,
  maximized = false,
}: WorkspacePanelBarProps) {
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
        className={`text-micro font-medium px-2.5 py-1 rounded-md transition-colors duration-150 ${
          active
            ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
            : 'text-secondary hover:text-primary'
        }`}
      >
        {label}
      </button>
    );
  };

  return (
    <div className="flex items-center justify-between gap-2 min-w-0">
      <div
        role="tablist"
        aria-label={t('panel.handle')}
        className="inline-flex shrink-0 gap-0.5 rounded-lg border border-border bg-nav p-0.5"
      >
        {tab('files', t('panel.view.files'))}
        {tab('browser', t('panel.view.browser'))}
      </div>
      <div className="flex items-center gap-1 shrink-0">
      {/* Annotate is a Browser-only mode: on the live page a drag would
          otherwise be forwarded, so a mode is needed to draw a box. Files
          needs none — selecting text shows Quote, like any document. */}
      {view === 'browser' && (
        <button
          type="button"
          aria-pressed={annotating}
          data-testid="panel-annotate"
          onClick={onToggleAnnotate}
          className={`shrink-0 text-micro font-medium px-2.5 py-1 rounded-md border transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
            annotating
              ? 'border-accent bg-accent/10 text-accent'
              : 'border-border text-secondary hover:text-primary hover:bg-card-hover'
          }`}
        >
          {annotating ? t('panel.annotate.done') : t('panel.annotate')}
        </button>
      )}
      {view === 'files' && onUpload && (
        <button
          type="button"
          onClick={onUpload}
          disabled={uploading}
          aria-label={uploading ? t('panel.uploading') : t('panel.upload')}
          title={uploading ? t('panel.uploading') : t('panel.upload')}
          data-testid="panel-upload"
          className={ICON_BUTTON}
        >
          {uploading ? (
            <span className="w-3.5 h-3.5 border-2 border-secondary border-t-transparent rounded-full animate-spin" aria-hidden />
          ) : (
            <Upload size={14} aria-hidden />
          )}
        </button>
      )}
      {onToggleMaximize &&
        (maximized ? (
          <button
            type="button"
            onClick={onToggleMaximize}
            aria-label={t('panel.restore')}
            title={t('panel.restore')}
            data-testid="panel-restore"
            className={ICON_BUTTON}
          >
            <Minimize2 size={14} aria-hidden />
          </button>
        ) : (
          <button
            type="button"
            onClick={onToggleMaximize}
            aria-label={t('panel.maximize')}
            title={t('panel.maximize')}
            data-testid="panel-maximize"
            data-tour="panel-maximize"
            className={ICON_BUTTON}
          >
            <Maximize2 size={14} aria-hidden />
          </button>
        ))}
      </div>
    </div>
  );
}

export interface WorkspacePanelProps {
  view: PanelView;
  browser: ReactNode;
  files: ReactNode;
}

export default function WorkspacePanel({ view, browser, files }: WorkspacePanelProps) {
  return (
    <div className="flex flex-col flex-1 h-full min-h-0 overflow-hidden" data-testid="workspace-panel" data-tour="workspace-panel">
      {view === 'browser' ? browser : files}
    </div>
  );
}

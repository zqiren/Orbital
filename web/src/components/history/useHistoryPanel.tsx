// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.8 — FilePreview's history mode, as one hook so FilePreview only
 * places three things: the 历史 glyph in its header, `wrap(body)` around its
 * body, and nothing at all while the project has no recorded change (D10:
 * `has_changes` false → no glyph, no column, the pane is byte-identical).
 *
 * Desktop: a ~300 px side column on a raised Float card, inside the file
 * area to the right of the content (overlaying it when the panel is too
 * narrow for both). Phone: a bottom sheet over the file. Closing returns the
 * file to the current version; no version stays selected.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { HistoryView } from '../../types';
import { useT } from '../../i18n/useT';
import { usePanelMobile } from '../../hooks/usePanelState';
import {
  bumpFileHistory,
  clearHistoryFocus,
  historyApi,
  historyVisible,
  SETTLE_MS,
  useFileVersions,
  useHistoryEpoch,
  useHistoryFocus,
  useHistoryStatus,
} from '../../hooks/useFileHistory';
import HistoryColumn from './HistoryColumn';
import VersionView from './VersionView';
import {
  buildRows,
  formatClock,
  legendAgents,
  PALETTE_VARS,
  rowKeyForVersion,
  selectionFor,
} from './historyModel';

export interface HistoryPanelOptions {
  projectId: string;
  /** Told after a restore wrote the file, so the parent re-reads it. */
  onChanged?: () => void;
}

interface Toast {
  text: string;
  undo?: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RemarkPlugin = any;

export function useHistoryPanel(
  opts: HistoryPanelOptions | undefined,
  path: string | null,
  /** The file type renders a document engine (Office): restore only. */
) {
  const t = useT();
  const projectId = opts?.projectId;
  const status = useHistoryStatus(projectId);
  const enabled = !!path && historyVisible(projectId, status);
  const phone = usePanelMobile();
  const epoch = useHistoryEpoch(projectId);

  const [open, setOpen] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [view, setView] = useState<{ key: string; view: HistoryView } | null>(null);
  const [restoring, setRestoring] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const onChangedRef = useRef(opts?.onChanged);
  useEffect(() => {
    onChangedRef.current = opts?.onChanged;
  });

  // Another file: the column closes and forgets its selection.
  useEffect(() => {
    setOpen(false);
    setSelectedKey(null);
    setExpanded(new Set());
    setView(null);
    setToast(null);
  }, [path, projectId]);

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(id);
  }, [toast]);

  const { list, loading, reload } = useFileVersions(projectId, path, enabled && open);
  const rows = useMemo(() => buildRows(list?.versions ?? []), [list]);
  const unavailable = status?.status === 'unavailable' || list?.status === 'unavailable';
  const wholeFileOnly = !!list?.whole_file_only;

  // 查看 from the undo sheet / a chip: open on that turn's version.
  const focus = useHistoryFocus(projectId, path);
  useEffect(() => {
    if (!focus || !enabled || !projectId) return;
    setOpen(true);
    if (focus.v === null) {
      clearHistoryFocus(projectId, focus.nonce);
      return;
    }
    if (!list) return; // wait for the versions
    const key = rowKeyForVersion(rows, focus.v);
    const row = rows.find((r) => r.key === key);
    setSelectedKey(row && !row.current ? key : null);
    clearHistoryFocus(projectId, focus.nonce);
  }, [focus, enabled, projectId, list, rows]);

  const selection = useMemo(() => {
    const sel = selectionFor(rows, selectedKey);
    return sel ?? selectionFor(rows, null);
  }, [rows, selectedKey]);

  // The tinted view of the selection (never for Office: restore only).
  const viewKey =
    open && enabled && selection && !wholeFileOnly && !unavailable && selection.target.sha
      ? `${path}|${selection.target.v}|${selection.fromV ?? ''}|${epoch}`
      : '';
  useEffect(() => {
    if (!viewKey || !projectId || !path || !selection) return;
    let cancelled = false;
    historyApi.view(projectId, path, selection.target.v, selection.fromV).then(
      (v) => {
        if (!cancelled && v && Array.isArray(v.lines)) setView({ key: viewKey, view: v });
      },
      () => {
        /* keep the normal body */
      },
    );
    return () => {
      cancelled = true;
    };
    // `selection` is derived from rows + key, both captured by viewKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey]);

  const close = useCallback(() => {
    setOpen(false);
    setSelectedKey(null);
    setView(null);
  }, []);

  const toggle = useCallback(() => {
    if (open) close();
    else {
      setOpen(true);
      if (projectId) bumpFileHistory(projectId);
    }
  }, [open, close, projectId]);

  const runRestore = useCallback(
    async (
      sha: string | null,
      expectSha: string | null,
      version: number | undefined,
      label: string,
      undoLabel: string | null,
    ) => {
      if (!projectId || !path) return;
      setRestoring(true);
      try {
        const { results } = await historyApi.restore(projectId, [
          { path, sha, expect_sha: expectSha, ...(version !== undefined ? { version } : {}) },
        ]);
        const r = results?.[0];
        if (r?.status === 'ok') {
          setSelectedKey(null);
          const after = r.current_sha ?? null;
          setToast({
            text: label,
            // The restore is itself a version, so it can be undone: put back
            // what it replaced, if nothing changed the file meanwhile.
            undo:
              undoLabel !== null
                ? () => {
                    setToast(null);
                    void runRestore(expectSha, after, undefined, undoLabel, null);
                  }
                : undefined,
          });
          onChangedRef.current?.();
        } else if (r?.status === 'conflict') {
          setToast({ text: t('history.conflict') });
        } else if (r?.status === 'busy') {
          setToast({ text: t('history.busy') });
        } else {
          setToast({ text: t('history.restoreFailed') });
        }
      } catch {
        setToast({ text: t('history.restoreFailed') });
      } finally {
        setRestoring(false);
        bumpFileHistory(projectId, { again: SETTLE_MS });
        reload();
      }
    },
    [projectId, path, reload, t],
  );

  const currentRow = rows.find((r) => r.current) ?? null;
  const canRestore = !!selection && !selection.isCurrent && !!currentRow && !unavailable;
  const restore = useCallback(() => {
    if (!selection || selection.isCurrent || !currentRow) return;
    const target = selection.target;
    void runRestore(
      target.sha,
      currentRow.head.sha,
      target.v,
      t('history.restored', { t: formatClock(target.ts) }),
      t('history.restored', { t: formatClock(currentRow.head.ts) }),
    );
  }, [selection, currentRow, runRestore, t]);

  const name = useCallback(
    async (version: { sha: string | null }, value: string) => {
      if (!projectId || !path || !version.sha) return;
      try {
        await historyApi.name(projectId, path, version.sha, value);
      } finally {
        reload();
      }
    },
    [projectId, path, reload],
  );

  const toggleExpand = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const shownView = view && view.key === viewKey ? view.view : null;
  const legend = useMemo(() => {
    if (shownView) return legendAgents(shownView.editors);
    return [];
  }, [shownView]);

  const glyph: ReactNode = enabled ? (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={open}
      title={t('history.open')}
      data-testid="history-open"
      className={`flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg px-[9px] text-[12.5px] transition-colors hover:bg-accent/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
        open ? 'text-accent' : 'text-secondary hover:text-primary'
      }`}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </svg>
      <span className="@max-[20rem]:hidden">{t('history.open')}</span>
    </button>
  ) : null;

  const column = (variant: 'column' | 'sheet') => (
    <HistoryColumn
      list={list}
      loading={loading}
      unavailable={unavailable}
      rows={rows}
      selectedKey={selectedKey}
      expanded={expanded}
      onSelect={setSelectedKey}
      onToggleExpand={toggleExpand}
      canRestore={canRestore}
      restoring={restoring}
      onRestore={restore}
      onClose={close}
      onName={name}
      legend={legend}
      variant={variant}
    />
  );

  const toastNode = toast && (
    <div
      role="status"
      data-testid="history-toast"
      className="absolute bottom-[22px] left-1/2 z-30 flex -translate-x-1/2 items-center gap-3 whitespace-nowrap rounded-[10px] bg-[#18181B] py-2 pl-3.5 pr-2 text-[12.5px] text-white shadow-[0_10px_30px_rgb(0_0_0/0.25)] dark:bg-[#F4F6F9] dark:text-[#18181B]"
    >
      <span>{toast.text}</span>
      {toast.undo && (
        <button
          type="button"
          onClick={toast.undo}
          className="rounded-md px-2 py-[3px] font-medium text-[#8DBBFB] dark:text-[#2A6FD6]"
        >
          {t('history.undo')}
        </button>
      )}
    </div>
  );

  /**
   * The file body with history around it. `markdown` picks the view's
   * renderer; `plugins` are the body's own remark plugins.
   */
  const wrap = (body: ReactNode, markdown: boolean, plugins?: RemarkPlugin[]): ReactNode => {
    if (!enabled || (!open && !toast)) return body;
    const content =
      open && shownView && !wholeFileOnly ? (
        <VersionView
          view={shownView}
          markdown={markdown}
          remarkPlugins={plugins}
          scrollKey={`${selectedKey ?? 'cur'}`}
        />
      ) : (
        body
      );
    if (phone) {
      return (
        <div className={`relative flex min-h-0 flex-1 flex-col ${PALETTE_VARS}`} data-testid="history-layout">
          {content}
          {open && (
            <div className="absolute inset-x-0 bottom-0 z-20 flex h-[56%] flex-col rounded-t-[18px] bg-elevated shadow-[0_-10px_40px_rgb(0_0_0/0.14)] dark:shadow-[0_-10px_40px_rgb(0_0_0/0.6)]">
              <span aria-hidden className="mx-auto mt-2 h-1 w-9 shrink-0 rounded-sm bg-muted/45" />
              {column('sheet')}
            </div>
          )}
          {toastNode}
        </div>
      );
    }
    return (
      <div className={`@container relative flex min-h-0 flex-1 ${PALETTE_VARS}`} data-testid="history-layout">
        <div className="flex min-w-0 flex-1 flex-col">{content}</div>
        {open && (
          <aside
            aria-label={t('history.open')}
            className="mb-2.5 ml-2.5 mr-2.5 mt-0.5 flex w-[304px] shrink-0 flex-col rounded-[14px] bg-elevated shadow-[0_2px_2px_rgb(0_0_0/0.02),0_4px_14px_rgb(0_0_0/0.04),0_22px_50px_rgb(0_0_0/0.07),inset_0_1px_0_rgb(255_255_255/0.8)] dark:shadow-[0_2px_3px_rgb(0_0_0/0.45),0_16px_48px_rgb(0_0_0/0.5),inset_0_1px_0_rgb(255_255_255/0.06)] @max-[36rem]:absolute @max-[36rem]:inset-y-0 @max-[36rem]:right-0 @max-[36rem]:z-20 @max-[36rem]:w-[min(304px,calc(100%-20px))]"
          >
            {column('column')}
          </aside>
        )}
        {toastNode}
      </div>
    );
  };

  return { enabled, open, glyph, wrap, close };
}

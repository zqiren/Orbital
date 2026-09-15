// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 090 — CSV / XLSX / XLS engine. Parsing runs in a worker (terminated on
 * unmount or after PARSE_TIMEOUT_MS); the owned virtualized grid renders the
 * resulting read-only model. The active sheet is published as SheetNavState
 * (DocumentNav renders a compact selector for headers), and the full tab strip
 * sits in this body's footer.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useT } from '../../i18n/useT';
import SheetGrid from './SheetGrid';
import type { SheetFormat, WorkbookModel } from './sheetModel';
import SheetWorker from './sheetWorker?worker';
import type { SheetWorkerRequest, SheetWorkerResponse } from './sheetWorker';
import type { EngineProps } from './types';

const PARSE_TIMEOUT_MS = 60_000;

function isSheetFormat(format: string): format is SheetFormat {
  return format === 'csv' || format === 'xlsx' || format === 'xls';
}

export default function SheetPreview({ bytes, format, fileName, onNav, onError }: EngineProps) {
  const t = useT();
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const [model, setModel] = useState<WorkbookModel | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (!isSheetFormat(format)) {
      onErrorRef.current('unreadable');
      return;
    }
    const worker = new SheetWorker();
    const fail = (reason: 'too_large' | 'unreadable') => {
      window.clearTimeout(timer);
      worker.terminate();
      onErrorRef.current(reason);
    };
    const timer = window.setTimeout(() => fail('unreadable'), PARSE_TIMEOUT_MS);
    worker.onmessage = (event: MessageEvent<SheetWorkerResponse>) => {
      if (!event.data.ok) {
        fail(event.data.reason);
        return;
      }
      window.clearTimeout(timer);
      worker.terminate();
      setModel(event.data.model);
    };
    worker.onerror = () => fail('unreadable');
    const copy = bytes.slice(0);
    const request: SheetWorkerRequest = { bytes: copy, format, name: fileName };
    worker.postMessage(request, [copy]);
    return () => {
      window.clearTimeout(timer);
      worker.terminate();
    };
  }, [bytes, format, fileName]);

  const sheetNames = useMemo(() => model?.sheets.map((s) => s.name) ?? [], [model]);
  const index = Math.min(active, Math.max(0, sheetNames.length - 1));
  const workbook = format !== 'csv';

  // A CSV has no sheets to navigate, so it publishes nothing.
  useEffect(() => {
    if (workbook && sheetNames.length > 0) {
      onNav({ kind: 'sheet', sheets: sheetNames, active: index, select: setActive });
    }
  }, [onNav, workbook, sheetNames, index]);
  useEffect(() => () => onNav(null), [onNav]);

  if (!model) {
    return (
      <div className="flex items-center justify-center h-full" role="status">
        <p className="text-xs text-secondary animate-pulse">{t('filePreview.doc.loading')}</p>
      </div>
    );
  }

  const sheet = model.sheets[index];
  return (
    <div className="absolute inset-0 flex flex-col min-h-0">
      {sheet?.truncated && (
        <p className="px-3 py-1 text-2xs text-secondary border-b border-border shrink-0">
          {t('filePreview.doc.sheetCapped', {
            rows: sheet.rows.length.toLocaleString(),
            cols: sheet.colCount.toLocaleString(),
          })}
        </p>
      )}
      <div className="relative flex-1 min-h-0">
        {!sheet || sheet.rows.length === 0 || sheet.colCount === 0 ? (
          <p className="p-4 text-xs text-secondary">{t('filePreview.doc.sheetEmpty')}</p>
        ) : (
          <SheetGrid key={index} sheet={sheet} />
        )}
      </div>
      {workbook && sheetNames.length > 1 && (
        <div
          role="tablist"
          aria-label={t('filePreview.doc.sheets')}
          className="flex items-center gap-1 px-2 py-1 border-t border-border bg-sidebar overflow-x-auto shrink-0"
        >
          {sheetNames.map((name, i) => (
            <button
              key={`${i}:${name}`}
              type="button"
              role="tab"
              aria-selected={i === index}
              title={name}
              onClick={() => setActive(i)}
              className={`shrink-0 max-w-[10rem] truncate text-xs px-2 py-0.5 rounded-md border transition-colors ${
                i === index
                  ? 'bg-accent text-white border-accent'
                  : 'border-border text-secondary hover:text-primary'
              }`}
            >
              {name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

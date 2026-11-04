// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.8 / DESIGN-DECISIONS "History of one file" — the column's
 * contents, shared by the desktop side column and the phone bottom sheet:
 * 恢复到此版本 and × on top, the legend of agent colours under it, then the
 * timeline grouped by day. Each row: the time (or the name) · 当前; the agent
 * logo and session name; 改了 N 行. A turn's writes fold into one row with a
 * count and a gutter chevron; a hover pencil names a version inline.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronRight, Pencil, X } from 'lucide-react';
import type { FileHistoryList, HistoryVersion } from '../../types';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { AgentLogo, AuthorLogos } from './AgentLogo';
import {
  agentColor,
  authorOf,
  changeLabel,
  dayLabel,
  formatClock,
  linesLabel,
  memberKey,
  type TimelineRow,
} from './historyModel';

export interface HistoryColumnProps {
  list: FileHistoryList | null;
  loading: boolean;
  /** The project's watcher is down: 历史不可用, no timeline. */
  unavailable: boolean;
  rows: TimelineRow[];
  /** null = the current row. */
  selectedKey: string | null;
  expanded: ReadonlySet<string>;
  onSelect: (key: string | null) => void;
  onToggleExpand: (rowKey: string) => void;
  canRestore: boolean;
  restoring: boolean;
  onRestore: () => void;
  onClose: () => void;
  onName: (version: HistoryVersion, name: string) => void;
  /** Agents whose colours the view shows. */
  legend: string[];
  /** 'sheet' = the phone bottom sheet (grab handle, same rows). */
  variant: 'column' | 'sheet';
}

const ICON_BTN =
  'flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-lg text-secondary transition-colors hover:bg-accent/10 hover:text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50';

function NameInput({
  initial,
  placeholder,
  onDone,
}: {
  initial: string;
  placeholder: string;
  onDone: (value: string | null) => void;
}) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement | null>(null);
  const done = useRef(false);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  const finish = (v: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(v);
  };
  return (
    <input
      ref={ref}
      value={value}
      placeholder={placeholder}
      aria-label={placeholder}
      onChange={(e) => setValue(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Enter') finish(value.trim());
        if (e.key === 'Escape') finish(null);
      }}
      onBlur={() => finish(value.trim())}
      className="h-[22px] min-w-0 flex-1 border-0 bg-transparent p-0 text-[13px] font-medium text-primary outline-none placeholder:font-normal placeholder:text-muted shadow-[inset_0_-1px_0_rgb(83_154_248/0.6)]"
    />
  );
}

export default function HistoryColumn({
  list,
  loading,
  unavailable,
  rows,
  selectedKey,
  expanded,
  onSelect,
  onToggleExpand,
  canRestore,
  restoring,
  onRestore,
  onClose,
  onName,
  legend,
  variant,
}: HistoryColumnProps) {
  const t = useT();
  const { locale } = useLocale();
  const [naming, setNaming] = useState<number | null>(null);
  // A selection made from elsewhere (查看 in the undo sheet) may sit below
  // the fold, especially in the phone sheet: bring it into view.
  const listRef = useRef<HTMLDivElement | null>(null);
  const hasList = rows.length > 0;
  useEffect(() => {
    // Scroll the list only: scrollIntoView would also scroll every clipped
    // ancestor (the project body) and shift the whole screen.
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (!list || !el) return;
    const top = el.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
    const bottom = top + el.offsetHeight;
    // Off screen: bring it to the top, which on the phone is the part of the
    // sheet the fixed composer never covers.
    if (top < list.scrollTop || bottom > list.scrollTop + list.clientHeight) {
      list.scrollTop = Math.max(0, top - 8);
    }
  }, [selectedKey, hasList]);

  const head = (
    <div className="flex shrink-0 items-center gap-1.5 pb-1.5 pl-3 pr-2.5 pt-3">
      <button
        type="button"
        onClick={onRestore}
        disabled={!canRestore || restoring}
        data-testid="history-restore"
        className="h-8 min-w-0 flex-1 truncate rounded-[9px] bg-accent px-2.5 text-center text-[13px] font-medium text-white transition-[filter] hover:brightness-105 disabled:cursor-default disabled:opacity-35 disabled:hover:brightness-100"
      >
        {t('history.restore')}
      </button>
      <button
        type="button"
        onClick={onClose}
        aria-label={t('history.close')}
        title={t('history.close')}
        data-testid="history-close"
        className={ICON_BTN}
      >
        <X size={14} aria-hidden />
      </button>
    </div>
  );

  const legendRow = legend.length > 0 && (
    <div className="flex shrink-0 flex-wrap gap-3.5 px-4 pb-1 pt-1.5" data-testid="history-legend">
      {legend.map((slug) => (
        <span key={slug} className="inline-flex items-center gap-[5px]">
          <AgentLogo slug={slug} />
          <span
            aria-hidden
            className="h-2 w-3.5 rounded-[3px]"
            style={{ backgroundColor: `color-mix(in srgb, ${agentColor(slug)} 45%, transparent)` }}
          />
        </span>
      ))}
    </div>
  );

  let body: ReactNode;
  if (unavailable) {
    body = (
      <p className="px-4 py-6 text-[13px] text-secondary" data-testid="history-unavailable">
        {t('history.unavailable')}
      </p>
    );
  } else if (list && list.versions.length === 0 && list.skipped_reason) {
    const key =
      list.skipped_reason === 'binary'
        ? 'history.skipped.binary'
        : list.skipped_reason === 'size'
          ? 'history.skipped.size'
          : 'history.skipped.ignored';
    body = (
      <p className="px-4 py-6 text-[13px] text-secondary" data-testid="history-skipped">
        {t(key)}
      </p>
    );
  } else if (!list && loading) {
    body = <div className="px-4 py-6" aria-busy="true" />;
  } else {
    const items: ReactNode[] = [];
    let lastDay = '';
    // Building, or a tracked file whose baseline hasn't landed: 当前 only.
    if (rows.length === 0) {
      items.push(
        <div key="cur-only" className="rounded-[10px] bg-accent/10 py-2 pl-7 pr-2.5 text-[13px] font-medium text-primary">
          {t('history.current')}
        </div>,
      );
    }
    rows.forEach((row, ri) => {
      if (row.dayKey !== lastDay) {
        lastDay = row.dayKey;
        items.push(
          <div key={`day-${row.dayKey}-${ri}`} className="pb-1 pl-7 pr-2.5 pt-3.5 text-[11px] font-medium text-muted">
            {dayLabel(row.head.ts, locale, t)}
          </div>,
        );
      }
      const v = row.head;
      const author = authorOf(v, t);
      const folded = row.members.length > 1;
      const open = folded && expanded.has(row.key);
      const selected = selectedKey === null ? row.current : selectedKey === row.key;
      const older = rows[ri + 1]?.head ?? null;
      // A restore row reads 你 · 恢复自 {t}: names belong to the content, so
      // the name stays on the version that was named, not on its restore.
      const isRestore = v.kind === 'restore';
      const name = isRestore ? null : v.name;
      const title = naming === v.v ? (
        <NameInput
          initial={v.name ?? ''}
          placeholder={t('history.name')}
          onDone={(value) => {
            setNaming(null);
            if (value !== null && value !== (v.name ?? '')) onName(v, value);
          }}
        />
      ) : name ? (
        <>
          <span className="min-w-0 truncate">{name}</span>
          <span className="text-xs font-normal tabular-nums text-muted">{formatClock(v.ts)}</span>
        </>
      ) : (
        <span className="tabular-nums">{formatClock(v.ts)}</span>
      );
      items.push(
        <div
          key={row.key}
          role="button"
          tabIndex={0}
          aria-pressed={selected}
          data-testid="history-row"
          data-version={v.v}
          onClick={() => onSelect(row.current ? null : row.key)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelect(row.current ? null : row.key);
            }
          }}
          className={`group/row relative flex cursor-pointer flex-col gap-px rounded-[10px] py-2 pl-7 pr-2.5 outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
            selected ? 'bg-accent/10' : 'hover:bg-accent/5'
          }`}
        >
          {folded && (
            <button
              type="button"
              aria-label={t('history.expand')}
              aria-expanded={open}
              title={t('history.edits', { n: row.members.length })}
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand(row.key);
              }}
              className={`absolute left-1.5 top-[9px] flex h-[18px] w-[18px] items-center justify-center rounded-[5px] text-muted transition-transform hover:text-primary ${
                open ? 'rotate-90' : ''
              }`}
            >
              <ChevronRight size={11} strokeWidth={2.4} aria-hidden />
            </button>
          )}
          <div className="flex min-h-5 items-center gap-[7px] text-[13px] font-medium text-primary">
            {title}
            {row.current && <span className="whitespace-nowrap text-[11.5px] font-normal text-muted">{t('history.current')}</span>}
            {folded && (
              <span className="whitespace-nowrap text-[11.5px] font-normal text-muted">
                {t('history.edits', { n: row.members.length })}
              </span>
            )}
            {naming !== v.v && v.sha && v.kind !== 'delete' && !isRestore && (
              <button
                type="button"
                aria-label={t('history.name')}
                title={t('history.name')}
                data-testid="history-name"
                onClick={(e) => {
                  e.stopPropagation();
                  setNaming(v.v);
                }}
                className="-mr-1 ml-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted opacity-0 transition-opacity hover:bg-accent/10 hover:text-primary focus-visible:opacity-100 group-hover/row:opacity-100 [@media(hover:none)]:opacity-100"
              >
                <Pencil size={13} aria-hidden />
              </button>
            )}
          </div>
          {(author.label || author.agents.length > 0 || author.you) && (
            <div className="flex min-w-0 items-center gap-[5px] text-xs text-muted">
              <AuthorLogos agents={author.agents} you={author.you} title={author.either} />
              <span className="truncate">{author.label}</span>
            </div>
          )}
          <div className="text-xs text-muted">{changeLabel(row, t, older)}</div>
        </div>,
      );
      if (open) {
        items.push(
          <div key={`${row.key}-subs`} className="mb-0.5 ml-2.5">
            {row.members.map((m) => {
              const k = memberKey(m);
              const sel = selectedKey === k;
              return (
                <div
                  key={k}
                  role="button"
                  tabIndex={0}
                  aria-pressed={sel}
                  data-testid="history-subrow"
                  data-version={m.v}
                  onClick={() => onSelect(k)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelect(k);
                    }
                  }}
                  className={`flex cursor-pointer items-center gap-[7px] rounded-[10px] py-1.5 pl-7 pr-2.5 text-[12.5px] outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                    sel ? 'bg-accent/10' : 'hover:bg-accent/5'
                  }`}
                >
                  <span className="tabular-nums text-primary">{formatClock(m.ts)}</span>
                  <span className="ml-0.5 whitespace-nowrap text-[11.5px] text-muted">
                    {m.lines !== null ? linesLabel(m.lines, t) : ''}
                  </span>
                </div>
              );
            })}
          </div>,
        );
      }
    });
    body = <div className="flex flex-col">{items}</div>;
  }

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid={`history-${variant}`}>
      {head}
      {legendRow}
      {list?.whole_file_only && !unavailable && (
        <p className="shrink-0 px-4 pb-1 pt-1.5 text-xs text-muted">{t('history.wholeFileOnly')}</p>
      )}
      {/* The phone keeps its composer fixed over the bottom of the screen:
          the sheet's list scrolls clear of it. */}
      <div ref={listRef} className={`min-h-0 flex-1 overflow-auto px-2 ${variant === 'sheet' ? 'pb-24' : 'pb-3.5'}`}>{body}</div>
    </div>
  );
}

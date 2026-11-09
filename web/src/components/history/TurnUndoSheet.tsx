// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.9 / §7 — "Undo this turn" as a reviewable list, never a
 * one-click rollback. One row per file the turn changed: a checkbox, the
 * path, 改了 N 行 and a bare eye (查看) that opens the file's history on the
 * turn. Untouched files start checked; a file someone changed after the turn
 * starts unchecked with an amber 之后又被 [logo] 改过; a file the turn created
 * starts unchecked (restoring deletes it). The button counts what will
 * happen; each restore carries the hash the user reviewed (`expect_sha`), so
 * a write that lands meanwhile comes back as a conflict, not a clobber.
 *
 * Lives in the docked panel slot (ChatTab), a bottom sheet on the phone.
 */
import { Fragment, useEffect, useMemo, useState } from 'react';
import type { HistoryEditor, TurnFile } from '../../types';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { bumpFileHistory, historyApi, SETTLE_MS, useTurnFiles } from '../../hooks/useFileHistory';
import { AgentLogo, AuthorLogos, YouGlyph } from './AgentLogo';
import { dayLabel, defaultChecked, formatBytes, formatClock, linesLabel, splitOnSlot, splitTurnFiles } from './historyModel';

export interface TurnRef {
  sessionId: string;
  fromTs: string;
  toTs: string | null;
  /**
   * Spec 156 §5.6: the sheet opened by editing a sent message, for the span
   * from that message to now. The primary button restores and then sends
   * (`onProceed`, only once every checked file restored; a conflict keeps the
   * sheet open and sends nothing); a second button sends without restoring.
   * Absent: 065's undo sheet, unchanged.
   */
  edit?: {
    onProceed: () => void;
    onSendOnly: () => void;
  };
}

export interface TurnUndoSheetProps {
  projectId: string;
  turn: TurnRef;
  /** 查看: open this file's history column on the turn's last version. */
  onView: (path: string, v: number) => void;
  /**
   * n files restored, with a way to put them back. `done` is true when every
   * checked file restored (the sheet can close); false leaves the conflicts
   * on screen.
   */
  onRestored: (n: number, undo: () => Promise<void>, done: boolean) => void;
}

const SLOT = '\u0000';

function LaterBy({ editor }: { editor: HistoryEditor }) {
  const t = useT();
  const you = editor.kind === 'user' || editor.kind === 'restore';
  const agents = editor.neutral ? [] : editor.agents;
  if (!you && agents.length === 0) {
    return <span className="text-warning-text">· {t('turn.changedLaterUnknown')}</span>;
  }
  const [before, after] = splitOnSlot(t('turn.changedLater', { x: SLOT }), SLOT);
  return (
    <span className="inline-flex items-center gap-1 text-warning-text" data-testid="turn-later">
      · {before.trim()}
      {you ? <YouGlyph /> : <AuthorLogos agents={agents} />}
      {after.trim()}
    </span>
  );
}

function changeOf(file: TurnFile, t: ReturnType<typeof useT>): string {
  if (file.kind === 'create') return t('history.created');
  if (file.kind === 'delete') return t('history.deleted');
  if (file.lines === null) {
    return t('history.size', { from: formatBytes(file.before_size), to: formatBytes(file.turn_size) });
  }
  return linesLabel(file.lines, t);
}

function CheckGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M5 12.5l4.5 4.5L19 7.5" />
    </svg>
  );
}

function EyeGlyph() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export default function TurnUndoSheet({ projectId, turn, onView, onRestored }: TurnUndoSheetProps) {
  const t = useT();
  const { locale } = useLocale();
  const data = useTurnFiles(projectId, turn.sessionId, turn.fromTs, turn.toTs, true);
  // Work files first; memory files last, under 记忆, never pre-checked.
  const { work, memory } = useMemo(() => splitTurnFiles(data?.files ?? []), [data]);
  const files = useMemo(() => [...work, ...memory], [work, memory]);

  // Re-read when the sheet opens: what changed since the line last looked
  // decides what is checked.
  useEffect(() => {
    bumpFileHistory(projectId);
  }, [projectId, turn.sessionId, turn.fromTs, turn.toTs]);

  // path → checked. A file keeps the user's choice across re-reads; a file
  // that comes back changed (a conflict) is unchecked by the restore itself.
  const [checks, setChecks] = useState<Map<string, boolean>>(() => new Map());
  const [rowNote, setRowNote] = useState<Map<string, 'busy' | 'refused'>>(() => new Map());
  const [working, setWorking] = useState(false);
  const [failed, setFailed] = useState(false);
  const checked = (f: TurnFile) => checks.get(f.path) ?? defaultChecked(f);
  const selected = files.filter(checked);
  const n = selected.length;

  const toggle = (f: TurnFile) =>
    setChecks((prev) => new Map(prev).set(f.path, !checked(f)));

  const restore = async () => {
    if (n === 0 || working) return;
    setWorking(true);
    setFailed(false);
    const items = selected.map((f) => ({ path: f.path, sha: f.before_sha, expect_sha: f.current_sha }));
    try {
      const { results } = await historyApi.restore(projectId, items);
      const ok = results.filter((r) => r.status === 'ok');
      const nextChecks = new Map(checks);
      const notes = new Map<string, 'busy' | 'refused'>();
      for (const r of results) {
        // Restored, or changed meanwhile: either way not something to
        // restore again by default.
        if (r.status === 'ok' || r.status === 'conflict') nextChecks.set(r.path, false);
        if (r.status === 'busy' || r.status === 'refused') notes.set(r.path, r.status);
      }
      setChecks(nextChecks);
      setRowNote(notes);
      bumpFileHistory(projectId, { again: SETTLE_MS });
      if (ok.length > 0) {
        const back = ok.map((r) => {
          const before = items.find((it) => it.path === r.path);
          return { path: r.path, sha: before?.expect_sha ?? null, expect_sha: r.current_sha ?? null };
        });
        onRestored(ok.length, async () => {
          await historyApi.restore(projectId, back).catch(() => undefined);
          bumpFileHistory(projectId, { again: SETTLE_MS });
        }, ok.length === results.length);
      }
      // Spec 156: restore, then fork, then send — only when nothing is left
      // to look at. A conflict (or a busy / refused file) keeps the sheet
      // open with that row amber and sends nothing.
      if (turn.edit && ok.length === results.length) turn.edit.onProceed();
    } catch {
      setFailed(true);
    } finally {
      setWorking(false);
    }
  };

  const editorAgents = useMemo(() => {
    const out: string[] = [];
    for (const f of work.length ? work : files) for (const a of f.editor.agents) if (!out.includes(a)) out.push(a);
    return out;
  }, [work, files]);
  const startedAt = Number.isNaN(Date.parse(turn.fromTs)) ? null : turn.fromTs;

  return (
    <div className="flex h-full min-h-0 flex-col bg-card dark:bg-elevated" data-testid="turn-undo-sheet">
      <div className="flex min-w-0 items-center gap-1.5 px-5 pb-2.5 pt-2.5 text-xs text-muted">
        {editorAgents.map((a) => (
          <AgentLogo key={a} slug={a} />
        ))}
        <span className="truncate">
          {[data?.session_name, startedAt ? `${dayLabel(startedAt, locale, t)} ${formatClock(startedAt)}` : null]
            .filter(Boolean)
            .join(' · ')}
        </span>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-auto px-2.5 py-1">
        {files.map((f, i) => {
          const on = checked(f);
          const note = rowNote.get(f.path);
          return (
            <Fragment key={f.path}>
            {i === work.length && memory.length > 0 && (
              <p data-testid="turn-undo-memory" className="px-2.5 pb-0.5 pt-3 text-[11px] font-medium text-muted">
                {t('turn.memory')}
              </p>
            )}
            <div data-testid="turn-undo-row" data-path={f.path} className="flex items-start gap-3 rounded-[10px] p-2.5 hover:bg-accent/5">
              <button
                type="button"
                role="checkbox"
                aria-checked={on}
                aria-label={f.path}
                onClick={() => toggle(f)}
                className={`mt-px flex h-[17px] w-[17px] shrink-0 items-center justify-center rounded-[5px] text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                  on ? 'bg-accent' : 'shadow-[inset_0_0_0_1.5px_rgb(138_138_148/0.7)]'
                }`}
              >
                {on && <CheckGlyph />}
              </button>
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className="truncate font-mono text-[12.5px] text-primary" title={f.path}>
                  {f.path}
                </span>
                <span className="flex flex-wrap items-center gap-[5px] text-xs text-muted">
                  {changeOf(f, t)}
                  {!f.clean && f.later_editor && <LaterBy editor={f.later_editor} />}
                  {note === 'busy' && <span className="text-warning-text">· {t('history.busy')}</span>}
                  {note === 'refused' && <span className="text-warning-text">· {t('history.restoreFailed')}</span>}
                </span>
              </div>
              <button
                type="button"
                onClick={() => onView(f.path, f.last_v)}
                aria-label={t('turn.view')}
                title={t('turn.view')}
                className="-mt-[5px] flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-accent/10 hover:text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              >
                <EyeGlyph />
              </button>
            </div>
            </Fragment>
          );
        })}
      </div>
      {(data?.no_history ?? 0) > 0 && (
        <p className="px-5 pt-2.5 text-xs text-muted" data-testid="turn-undo-skipped">
          {data!.no_history === 1
            ? t('turn.noHistory.one', { n: 1 })
            : t('turn.noHistory.other', { n: data!.no_history })}
        </p>
      )}
      {failed && <p className="px-5 pt-2.5 text-xs text-warning-text">{t('history.restoreFailed')}</p>}
      <div className="px-4 pb-[18px] pt-3.5">
        <button
          type="button"
          onClick={() => void restore()}
          disabled={n === 0 || working}
          data-testid="turn-undo-restore"
          className="block h-9 w-full truncate rounded-[9px] bg-accent px-2.5 text-center text-[13px] font-medium text-white transition-[filter] hover:brightness-105 disabled:cursor-default disabled:opacity-35"
        >
          {turn.edit
            ? n === 1
              ? t('chat.edit.restoreAndSend.one', { n })
              : t('chat.edit.restoreAndSend.other', { n })
            : n === 1
              ? t('turn.restoreSelected.one', { n })
              : t('turn.restoreSelected.other', { n })}
        </button>
        {turn.edit && (
          <button
            type="button"
            onClick={turn.edit.onSendOnly}
            disabled={working}
            data-testid="turn-undo-send-only"
            className="mt-2 block h-9 w-full truncate rounded-[9px] bg-soft-button px-2.5 text-center text-[13px] font-medium text-primary transition-colors hover:bg-soft-button-hover disabled:cursor-default disabled:opacity-35"
          >
            {t('chat.edit.sendOnly')}
          </button>
        )}
      </div>
    </div>
  );
}

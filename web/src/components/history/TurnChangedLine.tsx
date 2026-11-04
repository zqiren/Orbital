// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 065 §5.9 — the changed-files line inside a turn's last capsule:
 * 改了 N 个文件 plus small path chips, and a bare undo glyph revealed on hover
 * (always visible on a phone) whose tooltip reads 撤销这一轮的修改. N comes
 * from the turn route, so it counts shell and worker writes the tool counts
 * never saw. Renders nothing when the turn changed nothing on record.
 */
import type { TurnFiles } from '../../types';
import { useT } from '../../i18n/useT';
import { splitTurnFiles, undoneCount } from './historyModel';

function FileGlyph() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </svg>
  );
}

function UndoGlyph({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9 14 4 9l5-5" />
      <path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
    </svg>
  );
}

export default function TurnChangedLine({
  turn,
  onUndo,
  onOpenFile,
}: {
  turn: TurnFiles;
  /** Absent: no undo glyph (no panel slot to open the sheet in). */
  onUndo?: () => void;
  onOpenFile?: (path: string, v: number) => void;
}) {
  const t = useT();
  // The line is "what the agent did to my files": memory files stay out.
  const files = splitTurnFiles(turn.files).work;
  const n = files.length;
  const undone = undoneCount(files);
  return (
    <div
      data-testid="turn-changed-line"
      className="group/chg relative mt-1 flex w-fit max-w-full flex-wrap items-center gap-1.5 py-0.5 pr-9 text-xs text-muted"
    >
      <span className="inline-flex">
        <FileGlyph />
      </span>
      <span>{n === 1 ? t('turn.changed.one', { n }) : t('turn.changed.other', { n })}</span>
      {files.map((f) => (
        <button
          key={f.path}
          type="button"
          title={f.path}
          onClick={() => onOpenFile?.(f.path, f.last_v)}
          className="whitespace-nowrap rounded-md bg-accent/10 px-[7px] py-px font-mono text-[11px] text-secondary transition-colors hover:text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
        >
          {f.path.split('/').pop()}
        </button>
      ))}
      {undone > 0 && <span className="text-muted">· {t('turn.undone', { n: undone })}</span>}
      {onUndo && (
        <button
          type="button"
          onClick={onUndo}
          aria-label={t('turn.undo')}
          data-testid="turn-undo"
          className="group/ub absolute right-0 top-1/2 z-30 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-lg text-muted opacity-0 transition-opacity hover:bg-accent/10 hover:text-primary focus:outline-none focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-accent/50 group-hover/chg:opacity-100 [@media(hover:none)]:opacity-100 max-md:opacity-100"
        >
          <UndoGlyph />
          <span
            role="tooltip"
            className="pointer-events-none absolute left-1/2 top-[calc(100%+6px)] z-20 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-[#18181B] px-2 py-1 text-[11.5px] font-medium text-white shadow-[0_6px_18px_rgb(0_0_0/0.18)] group-hover/ub:block dark:bg-[#F4F6F9] dark:text-[#18181B]"
          >
            {t('turn.undo')}
          </span>
        </button>
      )}
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The composer's queue block (2026-11-24, layout 1 of the approved
 * prototype): messages waiting for their turn sit as rows inside the
 * composer's own surface, above the input, and enter the chat only when they
 * are really sent. A row is a lead mark (a clock for this chat's own agent,
 * the outside agent's logo for a message waiting behind it), one line of
 * text, its attachment, and three icon actions: edit, delete, send now. The
 * actions show on hover or focus on a desktop and always on a phone; their
 * names are tooltips only. Styles: `.qblock` in index.css.
 */

import { useCallback, useState, type FocusEvent, type PointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { ArrowUp, Clock, Paperclip, Pencil, Trash2 } from 'lucide-react';
import { useT } from '../i18n/useT';
import { getAgentIcon } from '../utils/agentIcons';
import type { QueueRow } from '../utils/composerQueue';

interface QueueBlockProps {
  rows: readonly QueueRow[];
  /** Display name of an outside agent, for its row's tooltip. */
  agentName: (handle: string) => string;
  onEdit: (row: QueueRow) => void;
  onDelete: (row: QueueRow) => void;
  onSendNow: (row: QueueRow) => void;
  /** A row whose action is in flight: its actions are disabled. */
  busyKey?: string | null;
}

interface Tip {
  text: string;
  left: number;
  top: number;
  align: 'left' | 'right';
}

function WorkerMark({ handle }: { handle: string }) {
  const [failed, setFailed] = useState(false);
  const icon = getAgentIcon(handle);
  if (!icon.src || failed) {
    return (
      <span
        className="qlead-mark"
        style={{ backgroundColor: icon.color, color: '#fff', fontSize: 9, lineHeight: '12px', textAlign: 'center' }}
        aria-hidden="true"
      >
        {icon.monogram.slice(0, 1)}
      </span>
    );
  }
  return (
    <span className="qlead-mark" aria-hidden="true">
      <img src={icon.src} alt="" onError={() => setFailed(true)} />
    </span>
  );
}

export default function QueueBlock({ rows, agentName, onEdit, onDelete, onSendNow, busyKey }: QueueBlockProps) {
  const t = useT();
  const [tip, setTip] = useState<Tip | null>(null);

  // The tooltip is fixed to the viewport (portal): the row list scrolls, and a
  // bubble inside it would be clipped.
  const show = useCallback((el: HTMLElement, text: string, align: 'left' | 'right') => {
    const r = el.getBoundingClientRect();
    setTip({ text, align, top: r.top - 6, left: align === 'right' ? r.right + 4 : r.left - 4 });
  }, []);
  const tipHandlers = (text: string, align: 'left' | 'right' = 'right') => ({
    onPointerEnter: (e: PointerEvent<HTMLElement>) => {
      if (e.pointerType === 'touch') return;
      show(e.currentTarget, text, align);
    },
    onPointerLeave: () => setTip(null),
    onFocus: (e: FocusEvent<HTMLElement>) => {
      if (e.currentTarget.matches(':focus-visible')) show(e.currentTarget, text, align);
    },
    onBlur: () => setTip(null),
  });

  if (rows.length === 0) return null;

  return (
    <div className="qblock" data-testid="queue-block">
      <div className="qrows" onScroll={() => setTip(null)}>
        {rows.map((row) => {
          const busy = busyKey === row.key;
          const behind = row.kind === 'worker' && row.handle
            ? t('chat.queuedBehind', { agent: agentName(row.handle) })
            : null;
          const chip = row.chips[0];
          return (
            <div key={row.key} className="qrow" data-testid="queue-row" data-kind={row.kind}>
              {behind && row.handle ? (
                <span
                  className="qlead"
                  tabIndex={0}
                  role="img"
                  aria-label={behind}
                  data-testid="queue-row-agent"
                  {...tipHandlers(behind, 'left')}
                >
                  <WorkerMark handle={row.handle} />
                </span>
              ) : (
                <span className="qlead" aria-hidden="true">
                  <Clock size={14} strokeWidth={1.75} />
                </span>
              )}
              {row.text && <span className="qtxt" data-testid="queue-row-text">{row.text}</span>}
              {chip && (
                <span className="qchip" data-testid="queue-row-chip">
                  <Paperclip size={12} strokeWidth={2} aria-hidden="true" className="shrink-0" />
                  <span>{row.chips.length > 1 ? `${chip.filename} +${row.chips.length - 1}` : chip.filename}</span>
                </span>
              )}
              <span className="qacts">
                {row.editable && (
                  <button
                    type="button"
                    className="qa"
                    aria-label={t('chat.queued.edit')}
                    data-testid="queue-row-edit"
                    disabled={busy}
                    onClick={() => { setTip(null); onEdit(row); }}
                    {...tipHandlers(t('chat.queued.edit'))}
                  >
                    <Pencil size={14} strokeWidth={1.9} aria-hidden="true" />
                  </button>
                )}
                <button
                  type="button"
                  className="qa"
                  aria-label={t('chat.queued.delete')}
                  data-testid="queue-row-delete"
                  disabled={busy}
                  onClick={() => { setTip(null); onDelete(row); }}
                  {...tipHandlers(t('chat.queued.delete'))}
                >
                  <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="qa"
                  aria-label={t('chat.queued.sendNow')}
                  data-testid="queue-row-send-now"
                  disabled={busy}
                  onClick={() => { setTip(null); onSendNow(row); }}
                  {...tipHandlers(t('chat.queued.sendNow'))}
                >
                  <ArrowUp size={14} strokeWidth={2} aria-hidden="true" />
                </button>
              </span>
            </div>
          );
        })}
      </div>
      <div className="qsep" />
      {tip && createPortal(
        <div
          className="qtip"
          role="tooltip"
          style={tip.align === 'right'
            ? { top: tip.top, left: tip.left, transform: 'translate(-100%, -100%)' }
            : { top: tip.top, left: tip.left, transform: 'translateY(-100%)' }}
        >
          {tip.text}
        </div>,
        document.body,
      )}
    </div>
  );
}

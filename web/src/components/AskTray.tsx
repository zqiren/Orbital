// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { SessionAsk } from '../hooks/useSessionAsks';
import { useT } from '../i18n/useT';

interface AskTrayProps {
  asks: SessionAsk[];
  index: number;
  onIndex: (index: number) => void;
  onReply: (ask: SessionAsk) => void;
  onDone: (askId: string) => void;
  onNotNeeded: (askId: string) => void;
  onLater: () => void;
}

/** Spec 146: an open ask at the top of the composer, one at a time. An ask can
 *  wait, so the composer stays free: only Reply makes a message an answer. */
export default function AskTray({ asks, index, onIndex, onReply, onDone, onNotNeeded, onLater }: AskTrayProps) {
  const t = useT();
  const ask = asks[index];
  if (!ask) return null;
  const actionClass = 'text-[13px] text-secondary hover:text-primary whitespace-nowrap';
  return (
    <div className="grid gap-1 px-1 pb-2 mb-2 border-b border-border" data-testid="ask-tray">
      <div className="flex flex-wrap items-start gap-x-3.5 gap-y-1">
        <p className="m-0 flex-[1_1_280px] min-w-0 text-[13.5px] text-primary break-words" data-testid="ask-tray-text">
          {ask.text}
          {ask.due && <span className="ml-2 text-xs text-muted whitespace-nowrap">{t('chat.ask.due', { date: ask.due })}</span>}
        </p>
        <div className="flex items-baseline gap-x-3.5 ml-auto">
          <button type="button" className="text-[13px] font-medium text-link hover:underline whitespace-nowrap" onClick={() => onReply(ask)}>
            {t('chat.ask.reply')}
          </button>
          <button type="button" className={actionClass} onClick={() => onDone(ask.id)}>{t('chat.ask.done')}</button>
          <button type="button" className={actionClass} onClick={() => onNotNeeded(ask.id)}>{t('chat.ask.notNeeded')}</button>
          <button type="button" className={`${actionClass} text-muted`} onClick={onLater}>{t('chat.ask.later')}</button>
        </div>
      </div>
      {asks.length > 1 && (
        <span className="flex items-center gap-1 text-xs text-muted">
          <button
            type="button"
            aria-label={t('chat.ask.previous')}
            disabled={index === 0}
            onClick={() => onIndex(index - 1)}
            className="hover:text-primary disabled:opacity-40"
          >
            <ChevronLeft size={14} />
          </button>
          {index + 1} / {asks.length}
          <button
            type="button"
            aria-label={t('chat.ask.next')}
            disabled={index === asks.length - 1}
            onClick={() => onIndex(index + 1)}
            className="hover:text-primary disabled:opacity-40"
          >
            <ChevronRight size={14} />
          </button>
        </span>
      )}
    </div>
  );
}

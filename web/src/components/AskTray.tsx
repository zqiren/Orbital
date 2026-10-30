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
  onDone: (askId: string) => void;
  onNotNeeded: (askId: string) => void;
  onLater: () => void;
}

/** Spec 146: the question at the top of the composer. Whatever is typed below
 *  it answers it; Later hides the tray so the composer is free for anything else. */
export default function AskTray({ asks, index, onIndex, onDone, onNotNeeded, onLater }: AskTrayProps) {
  const t = useT();
  const ask = asks[index];
  if (!ask) return null;
  const actionClass = 'text-[13px] text-secondary hover:text-primary';
  return (
    <div className="grid gap-1 px-1 pb-2 mb-1 border-b border-border" data-testid="ask-tray">
      <div className="flex items-center justify-between gap-2 text-xs font-medium text-primary">
        <span>
          {t('chat.ask.title')}
          {ask.due && <span className="ml-2 font-normal text-muted">{t('chat.ask.due', { date: ask.due })}</span>}
        </span>
        {asks.length > 1 && (
          <span className="flex items-center gap-1 font-normal text-muted">
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
      <p className="m-0 text-sm text-primary break-words" data-testid="ask-tray-text">{ask.text}</p>
      <div className="flex flex-wrap items-center gap-x-3.5">
        <button type="button" className={actionClass} onClick={() => onDone(ask.id)}>{t('chat.ask.done')}</button>
        <button type="button" className={actionClass} onClick={() => onNotNeeded(ask.id)}>{t('chat.ask.notNeeded')}</button>
        <button type="button" className={`${actionClass} ml-auto text-muted`} onClick={onLater}>{t('chat.ask.later')}</button>
      </div>
    </div>
  );
}

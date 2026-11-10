// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { SessionAsk } from '../hooks/useSessionAsks';
import { useT } from '../i18n/useT';
import MarkdownContent from './MarkdownContent';
import InlineMarkdown from './InlineMarkdown';

interface AskCardProps {
  ask: SessionAsk;
  /** A tap on an option: send it as the answer (one gesture). */
  onOption: (option: string) => void;
  onDone: (askId: string) => void;
  onNotNeeded: (askId: string) => void;
  onLater: () => void;
  /** Spec 169 D: with both, a workspace link in `detail` (the goals card's
   *  `project_goals.md`) is the chat's file card and opens the right panel. */
  workspace?: string;
  onOpenPath?: (path: string) => void;
}

/** Item 160, Claude Code style (2026-11-05): while this conversation has an
 *  open ask, the composer becomes the ask. The question heads it, the agent's
 *  options sit under it as one-tap answers, and the composer's textarea is
 *  the "in your own words" field — so there is no Reply button and no quote
 *  banner. One ask at a time; the next appears once this one is answered or
 *  closed. 已完成 / 不需要 close it without a reply; 稍后 puts the plain
 *  composer back for this session. */
export default function AskCard({ ask, onOption, onDone, onNotNeeded, onLater, workspace, onOpenPath }: AskCardProps) {
  const t = useT();
  const actionClass = 'text-[12.5px] text-muted hover:text-primary whitespace-nowrap';
  const options = ask.options ?? [];
  const detail = ask.detail ?? '';
  return (
    <div className="flex flex-col gap-2.5 px-1 pt-1 pb-0.5" data-testid="ask-card">
      <div className="flex items-start gap-x-4 gap-y-1.5 max-md:flex-col">
        <div className="flex items-start gap-2 flex-1 min-w-0">
          <span className="mt-[7px] w-1.5 h-1.5 rounded-full bg-warning shrink-0" aria-hidden="true" />
          <p className="m-0 flex-1 min-w-0 text-[13.5px] leading-snug font-medium text-primary break-words" data-testid="ask-card-text">
            <InlineMarkdown text={ask.text} />
          </p>
        </div>
        <div className="flex items-center gap-x-3.5 shrink-0 max-md:pl-3.5">
          {ask.due && <span className="text-xs text-muted whitespace-nowrap">{t('chat.ask.due', { date: ask.due })}</span>}
          <button type="button" className={actionClass} onClick={() => onDone(ask.id)}>{t('chat.ask.done')}</button>
          <button type="button" className={actionClass} onClick={() => onNotNeeded(ask.id)}>{t('chat.ask.notNeeded')}</button>
          <button type="button" className={actionClass} onClick={onLater}>{t('chat.ask.later')}</button>
        </div>
      </div>
      {/* The agent's context, as markdown. Capped with its own scroll so a
          long ask never pushes the options and the text line off-screen. */}
      {detail && (
        <div
          data-testid="ask-card-detail"
          className="pl-3.5 pr-1 -mt-1 max-h-[40vh] overflow-y-auto break-words [&_.markdown-content]:!text-primary/75 [&_.markdown-content_strong]:!text-primary"
        >
          <MarkdownContent content={detail} workspace={workspace} onOpenPath={onOpenPath} />
        </div>
      )}
      {options.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pl-3.5" data-testid="ask-card-options">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              data-testid="ask-card-option"
              onClick={() => onOption(option)}
              className="max-w-full truncate px-3 py-1 max-md:py-1.5 rounded-full border border-border text-[12.5px] text-primary hover:bg-card-hover hover:border-secondary/40 transition-colors duration-150"
            >
              {option}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

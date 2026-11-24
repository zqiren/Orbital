// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 195: one dispatched agent as one row of the conversation — the
 * capsule family FanoutCard and the agent_run capsule use (2px bar, then the
 * row): the agent's mark with a status dot, the first line of its brief, the
 * time it has taken, a chevron. The dot carries the state, so there is no
 * status word: green pulsing = working, amber = waits on you, red = failed,
 * hollow = stopped, none = done (the time freezes). A click opens the
 * agent's steps in the docked panel (ChatTab). No stop here: Stop sits in the
 * panel header, and Stop on the conversation stops every agent.
 *
 * WKWebView: no percent min-heights and no composited layers here — the ring
 * is a small absolutely-placed span inside the 20px mark box.
 */
import { memo, useEffect, useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { getAgentIcon } from '../utils/agentIcons';
import { formatElapsed } from './FanoutCard';
import { cardDot, type CardDot, type DispatchCardItem } from '../utils/dispatchCards';


/** The agent's mark with its status dot (the card row and the panel header). */
export function AgentMarkWithDot({ handle, dot, name }: { handle: string; dot: CardDot; name: string }) {
  const [failed, setFailed] = useState(false);
  const icon = getAgentIcon(handle);
  return (
    <span
      data-testid="dispatch-card-mark"
      data-dot={dot}
      role="img"
      aria-label={name}
      className="relative shrink-0 w-5 h-5 rounded-md border border-border/70 bg-card p-0.5 dark:border-transparent dark:bg-logo-tile"
    >
      {icon.src && !failed ? (
        <img src={icon.src} alt="" onError={() => setFailed(true)} className="block w-full h-full object-contain" />
      ) : (
        <span
          className="flex w-full h-full items-center justify-center rounded-[3px] text-[8px] font-semibold text-white"
          style={{ backgroundColor: icon.color }}
        >
          {icon.monogram.slice(0, 1)}
        </span>
      )}
      {dot !== 'none' && (
        <span aria-hidden className="absolute -right-[3px] -bottom-[3px] grid place-items-center w-[11px] h-[11px] rounded-full bg-card">
          {dot === 'running' && (
            <span className="absolute inset-0 rounded-full border-[1.5px] border-success-text motion-safe:animate-ping [animation-duration:1.6s] motion-reduce:opacity-40" />
          )}
          <span
            className={`relative w-[7px] h-[7px] rounded-full ${
              dot === 'running'
                ? 'bg-success-text'
                : dot === 'needs-you'
                  ? 'bg-warning-dot'
                  : dot === 'error'
                    ? 'bg-error-text'
                    : 'bg-card shadow-[inset_0_0_0_1.5px_var(--color-muted)]'
            }`}
          />
        </span>
      )}
    </span>
  );
}

interface DispatchCardProps {
  card: DispatchCardItem;
  /** The agent's name: the tooltip and the accessible label (logo over word). */
  name: string;
  /** The open turn waits on the user's approval. */
  needsYou?: boolean;
  /** This card's steps are the ones open in the panel. */
  open?: boolean;
  onOpen?: (card: DispatchCardItem) => void;
}

function DispatchCard({ card, name, needsYou = false, open = false, onOpen }: DispatchCardProps) {
  const running = card.status === 'running';
  // Date.now() is impure: ticked in state, once a second, only while running.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  const end = running ? now : (card.ended_at ?? card.started_at);
  const elapsed = formatElapsed(Math.max(0, end - card.started_at));
  const label = card.label || name;
  return (
    <div
      data-testid="dispatch-card"
      data-handle={card.handle}
      data-status={card.status}
      data-card-id={card.card_id}
      className="ml-9 flex gap-[10px]"
    >
      <div className="w-0.5 shrink-0 rounded-sm bg-border" aria-hidden />
      <div className="min-w-0 flex-1">
        <button
          type="button"
          data-testid="dispatch-card-row"
          onClick={() => onOpen?.(card)}
          aria-expanded={open}
          title={name}
          className={`-ml-1 flex w-full max-w-[640px] min-h-[30px] items-center gap-2 rounded-lg py-1 pl-1 pr-2 text-left transition-colors duration-150 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
            open ? 'bg-card-hover' : 'hover:bg-card-hover/55'
          }`}
        >
          <AgentMarkWithDot handle={card.handle} dot={cardDot(card.status, needsYou)} name={name} />
          <span className="min-w-0 flex-1 truncate text-caption font-medium text-primary">{label}</span>
          <span data-testid="dispatch-card-time" className="shrink-0 font-mono text-micro text-secondary tabular-nums">
            {elapsed}
          </span>
          <ChevronRight size={13} className="shrink-0 text-muted" aria-hidden />
        </button>
      </div>
    </div>
  );
}

export default memo(DispatchCard);

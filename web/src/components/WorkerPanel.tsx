// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 195: a dispatched agent's steps in the docked panel, opened from its
 * card (a bottom sheet on a phone — ChatTab hands both parts to the same
 * FilePreviewDrawer the panel and the undo sheet use). Not a third panel tab:
 * it takes the panel's slot while open and Files/Browser come back on close.
 * Read-only: talking to a worker is the composer's pin switcher (spec 091).
 *
 * Header: the agent's mark with its status dot, the first line of its brief,
 * and Stop while it runs (the drawer adds the close button). Body: the same
 * header, capsules and bubbles the conversation used to show inline, live —
 * ChatView keeps publishing the card as its steps grow.
 *
 * A fanout worker (spec 009) opens here too: its body is the drill-in view.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { sessionMessagesKey } from '../utils/sessionMessagesStore';
import { findTargetCard, useDispatchCards, type WorkerPanelTarget } from '../utils/dispatchCardsStore';
import type { DispatchCardItem } from '../utils/dispatchCards';
import { useSubAgentStatus } from '../hooks/useSubAgentStatus';
import { AgentMarkWithDot } from './DispatchCard';
import { cardDot } from '../utils/dispatchCards';
import { WorkerSteps } from './ChatView';
import { StopGlyph } from './StopGlyph';
import StopWorkerConfirm from './StopWorkerConfirm';
import SubAgentDrillIn from './SubAgentDrillIn';
import AgentGlyph from './welcome/AgentGlyph';

interface Common {
  projectId: string;
  sessionId: string | undefined;
  target: WorkerPanelTarget;
  agents: Array<{ slug: string; name: string }>;
}

function useTargetCard(projectId: string, sessionId: string | undefined, target: WorkerPanelTarget): DispatchCardItem | undefined {
  const cards = useDispatchCards(sessionMessagesKey(projectId, sessionId));
  return target.kind === 'card' ? findTargetCard(cards, target) : undefined;
}

function nameOf(agents: Common['agents'], handle: string): string {
  return agents.find((a) => a.slug === handle)?.name ?? handle;
}

export function WorkerPanelHeader({ projectId, sessionId, target, agents }: Common) {
  const t = useT();
  const card = useTargetCard(projectId, sessionId, target);
  const { agents: live } = useSubAgentStatus(projectId, sessionId);
  const [confirming, setConfirming] = useState(false);
  const [stopping, setStopping] = useState(false);
  if (target.kind === 'fanout') {
    return (
      <div className="flex items-center gap-2 min-w-0" data-testid="worker-panel-header">
        <span className="min-w-0 flex-1 truncate text-caption font-medium text-primary">{target.label}</span>
      </div>
    );
  }
  const handle = card?.handle ?? target.handle;
  const name = nameOf(agents, handle);
  const running = card?.status === 'running';
  const needsYou = live.some((a) => a.handle === handle && a.awaiting_approval === true);
  const doStop = async () => {
    setConfirming(false);
    setStopping(true);
    try {
      const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      await api(`/api/v2/agents/${encodeURIComponent(projectId)}/sub-agents/${encodeURIComponent(handle)}/stop${qs}`, {
        method: 'POST',
      });
    } catch {
      /* the stopped row reaches the chat either way */
    } finally {
      setStopping(false);
    }
  };
  return (
    <div className="flex items-center gap-2 min-w-0" data-testid="worker-panel-header" data-handle={handle}>
      {card ? (
        <AgentMarkWithDot handle={handle} dot={cardDot(card.status, needsYou)} name={name} />
      ) : (
        <AgentGlyph slug={handle} size={20} title={name} />
      )}
      <span className="min-w-0 flex-1 truncate text-caption font-medium text-primary" title={card?.label || name}>
        {card?.label || name}
      </span>
      {running && (
        <button
          type="button"
          data-testid="worker-panel-stop"
          onClick={() => setConfirming(true)}
          disabled={stopping}
          aria-label={t('subAgentBar.stopTitle', { name })}
          title={t('subAgentBar.stopTitle', { name })}
          className="group flex items-center justify-center w-7 h-7 shrink-0 rounded-md text-error-text hover:bg-error-surface transition-colors disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
        >
          <StopGlyph size={16} />
        </button>
      )}
      {confirming && (
        <StopWorkerConfirm name={name} running onCancel={() => setConfirming(false)} onConfirm={() => void doStop()} />
      )}
    </div>
  );
}

interface BodyProps extends Common {
  workspace?: string;
  onOpenPath?: (path: string) => void;
}

export function WorkerPanelBody({ projectId, sessionId, target, agents, workspace, onOpenPath }: BodyProps) {
  const t = useT();
  const card = useTargetCard(projectId, sessionId, target);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Follow the newest step while the reader is at the bottom.
  const pinnedRef = useRef(true);
  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);
  const steps = card?.steps;
  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [steps]);

  if (target.kind === 'fanout') {
    return (
      <div className="flex-1 min-h-0 flex flex-col" data-testid="worker-panel-body">
        <SubAgentDrillIn
          projectId={projectId}
          sessionId={sessionId}
          handle={target.handle}
          displayName={target.label}
          embedded
        />
      </div>
    );
  }
  const running = card?.status === 'running';
  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      data-testid="worker-panel-body"
      data-card-id={card?.card_id}
      className="flex-1 min-h-0 overflow-y-auto px-4 py-4 flex flex-col gap-3"
    >
      {card && <WorkerSteps steps={card.steps} workspace={workspace} onOpenPath={onOpenPath} />}
      {running && (card?.steps.length ?? 0) === 0 && (
        <div className="flex items-center gap-2 px-1 py-1 text-secondary text-body">
          <Loader2 size={14} className="animate-spin" />
          <span>{t('chat.thinking')}</span>
        </div>
      )}
      {!card && (
        <div className="flex items-center gap-2 px-1 text-caption text-secondary">
          <AgentGlyph slug={target.handle} size={16} title={nameOf(agents, target.handle)} />
        </div>
      )}
    </div>
  );
}

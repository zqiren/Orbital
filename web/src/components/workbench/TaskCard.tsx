// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * One Workbench card (spec 154 §5.7-5.8): a dot, a title line, a muted line
 * and a time. The title is the way into the session (a slight underline on
 * hover); there is no separate open icon (2026-11-04). On hover (desktop) the
 * time fades, a bare ✓ (确认, the card leaves) appears on the title line, and
 * a borderless reply field opens under the text. ✓ is the card's only icon,
 * and only waiting cards have it. On a phone the reply row is always shown on
 * waiting cards, with send and ✓ on its right.
 *
 * A click anywhere else on the card (not the title, ✓ or the reply field)
 * selects it: the Workbench opens that session's chat in the reading pane;
 * a second click closes it (2026-11-04).
 *
 * An automation run (a session a schedule / file-watch trigger started)
 * reads `<automation name> · <the run's first line>`; the first line is then
 * not repeated in the muted line.
 *
 * The agent is its LOGO, never its name (the name is the tooltip). The card
 * always renders from the feed row it is given.
 */

import { useEffect, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { Check, Send } from 'lucide-react';
import type { WorkbenchFeedRow } from '../../types';
import { useT } from '../../i18n/useT';
import type { StringKey } from '../../i18n/strings';
import { formatRelativeTime } from '../../utils/relativeTime';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';
import { needsAction } from '../../utils/waitingOnYou';
import { classifySessionName } from '../../lib/sessionLabel';
import type { FeedItem } from './feedModel';

const CAUSE_KEY: Record<string, StringKey> = {
  usage_limit: 'leftOff.cause.usageLimit',
  model_unavailable: 'leftOff.cause.modelUnavailable',
  network_auth: 'leftOff.cause.networkAuth',
  interrupted: 'leftOff.cause.interrupted',
  no_reply: 'leftOff.cause.noReply',
};

/** An ask detail's first paragraph as one plain line for the muted row:
 *  the markdown marks dropped, links reduced to their text. */
export function detailLine(detail: string): string {
  const first = detail.split(/\n\s*\n/)[0] ?? '';
  return first
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/(\*\*|__|`|~~)/g, '')
    .replace(/^\s*(?:[-*+]|\d+\.)\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function clockTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export interface TaskCardProps {
  item: FeedItem;
  /** The project's display name (null project = not routed yet). */
  projectName: (projectId: string | null) => string | null;
  /** An agent slug's display name, for the logo's tooltip. */
  agentName: (slug: string) => string;
  /** The running session's live line, if any. */
  activity: string | null;
  phone: boolean;
  fresh?: boolean;
  onOpen: (item: FeedItem) => void;
  onConfirm: (item: FeedItem) => void;
  /** Resolves true when the reply went out (the field then clears). */
  onReply: (item: FeedItem, words: string) => Promise<boolean>;
  /** This card's session is in the reading pane (the selected ring). */
  selected?: boolean;
  /** A click on the card body. Absent: the card cannot be selected. */
  onSelect?: (item: FeedItem) => void;
}

export default function TaskCard({
  item, projectName, agentName, activity, phone, fresh, onOpen, onConfirm, onReply,
  selected = false, onSelect,
}: TaskCardProps) {
  const t = useT();
  const [reply, setReply] = useState('');
  const [sending, setSending] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const row: WorkbenchFeedRow | null = item.row;
  const pending = item.pending;
  const approval = row?.status === 'pending_approval';
  const ask = row?.ask ?? null;
  const canReply = item.waiting && !approval && !pending;
  const canConfirm = item.waiting && !approval && !pending;

  const slug = (pending ? (pending.agent === 'orbital' ? null : pending.agent) : row?.agent) ?? MAIN_AGENT_HANDLE;
  const icon = getAgentIcon(slug);
  const agentLabel = slug === MAIN_AGENT_HANDLE ? 'Orbital' : agentName(slug);
  const pid = pending ? pending.project_id : row?.project_id ?? null;
  const project = projectName(pid);

  // An automation run is named by its automation, not by the trigger header
  // its first message carries.
  const automation = row?.trigger_type && !pending
    ? classifySessionName(row.name).displayName
    : null;
  const runLine = automation && !ask ? row?.result_line ?? null : null;
  const taskTitle = pending?.text
    ?? (automation ? (runLine ? `${automation} · ${runLine}` : automation) : null)
    ?? row?.name ?? t('workbench.card.untitled');
  const title = ask ? `“${ask.text}”` : taskTitle;

  // The muted line's tail: one of activity / result / stop / approval / due.
  let detail: { text: string; warn: boolean } | null = null;
  if (pending) {
    detail = { text: t('workbench.card.starting'), warn: false };
  } else if (row) {
    if (ask) {
      // The ask's own context first (its first paragraph); else the task.
      const name = (ask.detail ? detailLine(ask.detail) : '') || automation || row.name;
      detail = name ? { text: name, warn: false } : null;
    } else if (approval) {
      detail = { text: t('workbench.card.approval'), warn: true };
    } else if (item.dot === 'run') {
      detail = activity ? { text: activity, warn: false } : null;
    } else if (row.result_line && !runLine) {
      detail = { text: row.result_line, warn: false };
    } else if (row.stop) {
      const key = CAUSE_KEY[row.stop.cause] ?? 'leftOff.cause.interrupted';
      const text = row.stop.cause === 'usage_limit' && row.stop.resets_at
        ? t('leftOff.cause.usageLimitAt', { time: clockTime(row.stop.resets_at) })
        : t(key);
      detail = { text, warn: needsAction(row) };
    }
  }
  const overdueDue = ask?.overdue && ask.due ? t('workbench.card.due', { date: ask.due }) : null;

  // The relative time is read at render, so re-render on a clock: every 10 s
  // while under a minute ("0 秒前" must not stand for half a minute), then
  // every minute.
  const [, setTick] = useState(0);
  useEffect(() => {
    const then = item.time ? Date.parse(item.time) : NaN;
    if (Number.isNaN(then)) return;
    let id: number;
    const schedule = () => {
      id = window.setTimeout(() => {
        setTick((n) => n + 1);
        schedule();
      }, Date.now() - then < 60_000 ? 10_000 : 60_000);
    };
    schedule();
    return () => window.clearTimeout(id);
  }, [item.time]);
  const timeText = item.time ? formatRelativeTime(item.time, t) : '';

  function confirm() {
    setLeaving(true);
    window.setTimeout(() => onConfirm(item), 220);
  }

  async function sendReply(option?: string) {
    const words = (option ?? reply).trim();
    if (!words || sending) return;
    setSending(true);
    try {
      // A tapped option leaves whatever was typed in the field alone.
      if ((await onReply(item, words)) && option === undefined) setReply('');
    } finally {
      setSending(false);
    }
  }

  // Item 160: the choices the agent offered, as the same pills the chat's ask
  // card shows; a tap answers with that option.
  const options = ask?.options ?? [];

  function onReplyKey(e: KeyboardEvent<HTMLInputElement>) {
    // While an IME (e.g. Pinyin) is composing, Enter commits the candidate —
    // it belongs to the input method, not to us. Don't submit mid-composition.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter') {
      e.preventDefault();
      void sendReply();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setReply('');
      e.currentTarget.blur();
    }
  }

  // ✓ is the only icon, and only a card that can be confirmed has it.
  const acts = canConfirm ? (
    <div className="wb-acts">
      <button
        type="button"
        className="wb-ab is-ok"
        data-testid="wb-card-confirm"
        aria-label={t('workbench.card.confirm')}
        title={t('workbench.card.confirm')}
        onClick={confirm}
      >
        <Check size={14} strokeWidth={1.75} aria-hidden="true" />
      </button>
    </div>
  ) : null;

  const replyRow = canReply ? (
    <div className="wb-rwrap">
      <div>
        {options.length > 0 && (
          <div className="wb-askopts">
            {options.map((option) => (
              <button
                key={option}
                type="button"
                className="wb-askopt"
                data-testid="wb-card-option"
                onClick={() => void sendReply(option)}
                disabled={sending}
              >
                {option}
              </button>
            ))}
          </div>
        )}
        <div className="wb-rbox">
          <input
            value={reply}
            onChange={(e) => setReply(e.target.value)}
            onKeyDown={onReplyKey}
            placeholder={t('workbench.card.reply')}
            aria-label={t('workbench.card.reply')}
            data-testid="wb-card-reply"
            disabled={sending}
          />
          <button
            type="button"
            className="wb-snd"
            aria-label={t('workbench.composer.send')}
            title={t('workbench.composer.send')}
            onClick={() => void sendReply()}
            disabled={sending}
          >
            <Send size={14} aria-hidden="true" />
          </button>
          {phone && acts}
        </div>
      </div>
    </div>
  ) : null;

  // The body selects; the controls inside it (title, ✓, reply field and its
  // send) keep their own jobs.
  function onCardClick(e: MouseEvent<HTMLDivElement>) {
    if (!onSelect) return;
    if ((e.target as Element).closest('button, input, textarea, a')) return;
    onSelect(item);
  }

  const cls = [
    'wb-card', 'wb-layer',
    onSelect ? 'is-selectable' : '',
    selected ? 'is-selected' : '',
    ask ? 'is-ask' : '',
    item.waiting ? 'is-waiting' : '',
    fresh ? 'is-fresh' : '',
    leaving ? 'is-leaving' : '',
  ].filter(Boolean).join(' ');

  return (
    <div
      className={cls}
      data-testid="wb-card"
      data-dot={item.dot}
      data-key={item.key}
      data-selected={selected ? 'true' : 'false'}
      onClick={onCardClick}
    >
      <div className="wb-head">
        <span className={`wb-dot is-${item.dot}`} aria-hidden="true" />
        <span className="wb-tx">
          <button
            type="button"
            className="wb-title"
            data-testid="wb-card-title"
            title={t('workbench.card.open')}
            onClick={() => onOpen(item)}
          >
            {title}
          </button>
          <span className="wb-meta" data-testid="wb-card-meta">
            <span className={`wb-lg${slug === MAIN_AGENT_HANDLE ? ' is-orb' : ''}`} title={agentLabel}>
              {icon.src ? <img src={icon.src} alt={agentLabel} /> : null}
            </span>
            {project}
            {detail && (
              <>
                {project ? ' · ' : ''}
                {detail.warn ? <span className="wb-warn">{detail.text}</span> : detail.text}
              </>
            )}
            {overdueDue && (
              <>
                {' · '}
                <span className="wb-warn">{overdueDue}</span>
              </>
            )}
          </span>
          {!phone && replyRow}
        </span>
        <span className="wb-rz">
          <span className="wb-time">{timeText}</span>
        </span>
        {!phone && acts}
      </div>
      {phone && replyRow}
    </div>
  );
}

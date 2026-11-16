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
 * An ask card's question renders as inline markdown inside its quotes, and
 * the agent's detail as markdown under it (clamped to six lines); its muted line then names the automation or the
 * session. A result line stays plain: the daemon already flattened it.
 *
 * An automation run (a session a schedule / file-watch trigger started)
 * reads `<automation name> · <the run's first line>`; the first line is then
 * not repeated in the muted line.
 *
 * The agent is its LOGO, never its name (the name is the tooltip). The card
 * always renders from the feed row it is given.
 *
 * Spec 180 — the project name is a button (its chevron always shows, small
 * and muted) that opens the composer's own project picker:
 *   - while 自动 picks, the name's place reads a shimmering 自动 after the
 *     Sparkles glyph; the picker has 自动 ticked on top, and a project there
 *     ends the picking and sends to it. ✕ (撤回) returns the text;
 *   - after routing, ✦ (the composer's Sparkles) sits before the name when
 *     Orbital chose it; opening the card or moving it folds ✦ away;
 *   - on a card with a session, another project moves the session at once
 *     (a running one too): the dot rests, the name changes in place (old
 *     blurs up, new rises), and the work continues there.
 */

import { useEffect, useLayoutEffect, useState, type KeyboardEvent, type MouseEvent, useRef } from 'react';
import { Check, ChevronDown, Send, Sparkles, X } from 'lucide-react';
import type { Project, WorkbenchFeedRow } from '../../types';
import { useT } from '../../i18n/useT';
import { changeOptionPrefill, isChangeOption } from '../../utils/changeOptionPrefill';
import type { StringKey } from '../../i18n/strings';
import { formatRelativeTime } from '../../utils/relativeTime';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';
import { needsAction } from '../../utils/waitingOnYou';
import { classifySessionName } from '../../lib/sessionLabel';
import MarkdownContent from '../MarkdownContent';
import InlineMarkdown from '../InlineMarkdown';
import type { FeedItem } from './feedModel';
import { AnchoredProjectPicker } from './ProjectPicker';
import SourceMark from '../sources/SourceMark';
import { feedSourceName } from '../sources/sourcesText';

/** How long the name swap runs (the CSS animations end inside it). */
const SWAP_MS = 620;

/**
 * The project name on the card, which changes in place: the old name blurs
 * up and out while the new one rises in, and the slot's width eases from
 * one to the other (prototype `setNm`, l.819-836). The first render, and the
 * shimmering 自动 while picking, appear without a swap.
 */
function SlotName({ text, shim, phone }: { text: string; shim: boolean; phone: boolean }) {
  const k = shim ? '\u0000auto' : `n:${text}`;
  const wrapRef = useRef<HTMLSpanElement>(null);
  const [shown, setShown] = useState({ k, text, shim });
  const [out, setOut] = useState<{ k: string; text: string; shim: boolean } | null>(null);
  const fromRef = useRef<number | null>(null);

  // A new name: remember the width the old one had, keep it on show as the
  // leaving copy, and render the new one (before paint).
  useLayoutEffect(() => {
    if (k === shown.k) return;
    fromRef.current = wrapRef.current?.offsetWidth ?? null;
    setOut(shown);
    setShown({ k, text, shim });
  }, [k, text, shim, shown]);

  // Then ease the width from the old name to the new one.
  useLayoutEffect(() => {
    const w = wrapRef.current;
    const from = fromRef.current;
    if (!w || !out || from === null) return;
    fromRef.current = null;
    const nu = w.querySelector<HTMLElement>('.wb-nm-in');
    const slot = w.closest<HTMLElement>('.wb-slot');
    const meta = slot?.parentElement;
    if (!nu || !slot || !meta) return;
    nu.classList.add('is-measure');
    const natural = nu.offsetWidth;
    nu.classList.remove('is-measure');
    const cap = Math.max(40, meta.clientWidth * (phone ? 0.7 : 0.64) - (slot.offsetWidth - from) - 2);
    const to = Math.min(natural, cap);
    w.classList.remove('is-anim');
    w.style.width = `${from}px`;
    void w.offsetWidth;
    w.classList.add('is-anim');
    w.style.width = `${to}px`;
    const id = window.setTimeout(() => {
      w.classList.remove('is-anim');
      w.style.width = '';
      setOut(null);
    }, SWAP_MS);
    return () => window.clearTimeout(id);
  }, [out, phone]);

  const body = (v: { text: string; shim: boolean }) => (v.shim ? <span className="wb-shim">{v.text}</span> : v.text);
  return (
    <span ref={wrapRef} className="wb-nmw">
      {out && <span key={`o${out.k}`} className="wb-nm is-out" aria-hidden="true">{body(out)}</span>}
      <span key={`n${shown.k}`} className={`wb-nm${out ? ' wb-nm-in is-in' : ''}`} data-testid="wb-card-project-name">
        {body(shown)}
      </span>
    </span>
  );
}

const CAUSE_KEY: Record<string, StringKey> = {
  usage_limit: 'leftOff.cause.usageLimit',
  model_unavailable: 'leftOff.cause.modelUnavailable',
  network_auth: 'leftOff.cause.networkAuth',
  interrupted: 'leftOff.cause.interrupted',
  no_reply: 'leftOff.cause.noReply',
};

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
  /** Spec 180: the projects the card's picker offers (the Workbench's). */
  projects?: readonly Project[];
  /** Spec 180: a project picked on the card — ends the picking (pending), or
   *  moves the session. Absent: the name is not a button. */
  onPickProject?: (item: FeedItem, projectId: string) => void;
  /** Spec 180: ✕ on a picking card (after its leave animation). */
  onUnsend?: (item: FeedItem) => void;
  /** Spec 180: the picker opened / closed on a picking card (a route answer
   *  waits for it). */
  onPickerOpen?: (item: FeedItem, open: boolean) => void;
  /** Task sources: the card's source mark opens the panel at its entry. */
  onOpenSource?: (item: FeedItem) => void;
}

export default function TaskCard({
  item, projectName, agentName, activity, phone, fresh, onOpen, onConfirm, onReply,
  selected = false, onSelect, projects = [], onPickProject, onUnsend, onPickerOpen, onOpenSource,
}: TaskCardProps) {
  const t = useT();
  const [reply, setReply] = useState('');
  const replyRef = useRef<HTMLInputElement>(null);
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
  // Spec 180: 自动 is still picking this send's project.
  const picking = !!pending?.routing;
  const slotRef = useRef<HTMLButtonElement>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  // The name opens the picker while picking, and on a card with a session
  // (a move); in between (routed, not minted yet) and mid-move it only shows.
  const slotLive = !!onPickProject && (picking || (!pending && !!row?.session_id && !item.moving));
  function openPicker(open: boolean) {
    setPickerOpen(open);
    if (picking) onPickerOpen?.(item, open);
  }
  function pickProject(projectId: string) {
    const current = picking ? null : pid;
    setPickerOpen(false);
    if (projectId !== current) onPickProject?.(item, projectId);
    if (picking) onPickerOpen?.(item, false);
  }
  // A move or a route answer that ends the picking closes the picker.
  useEffect(() => {
    if (pickerOpen && !slotLive) setPickerOpen(false);
  }, [pickerOpen, slotLive]);

  // An automation run is named by its automation, not by the trigger header
  // its first message carries.
  const automation = row?.trigger_type && !pending
    ? classifySessionName(row.name).displayName
    : null;
  const runLine = automation && !ask ? row?.result_line ?? null : null;
  const taskTitle = pending?.text
    ?? (automation ? (runLine ? `${automation} · ${runLine}` : automation) : null)
    ?? row?.name ?? t('workbench.card.untitled');
  // An ask's question is the agent's own line, often with `**bold**` and
  // `code` in it: inline markdown inside the quotes.
  // The 2-line clamp sits on an inner span: WebKit ignores line-clamp on a
  // <button> (the owner's WKWebView showed a 9-line title).
  const title = ask
    ? <span className="wb-tclamp" data-testid="wb-card-title-clamp">“<InlineMarkdown text={ask.text} links={false} />”</span>
    : taskTitle;

  // The muted line's tail: one of activity / result / stop / approval / due.
  let detail: { text: string; warn: boolean } | null = null;
  if (pending) {
    // While 自动 picks there is no detail: the name's place says it.
    detail = picking ? null : { text: t('workbench.card.starting'), warn: false };
  } else if (item.moving) {
    // Stopped for the move: the dot rests, nothing to say yet.
    detail = null;
  } else if (row && item.woke) {
    // Moved and restarting in the new project, until its run is seen.
    detail = { text: t('workbench.card.starting'), warn: false };
  } else if (row) {
    if (ask) {
      // The ask's context renders as markdown under the title (askDetail);
      // the muted line names where it came from.
      const name = automation || row.name;
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

  function unsend() {
    setPickerOpen(false);
    setLeaving(true);
    window.setTimeout(() => onUnsend?.(item), 220);
  }

  // ✓ is the only icon, and only a card that can be confirmed has it; a card
  // still picking its project has ✕ (撤回) in that place instead.
  const acts = picking && onUnsend ? (
    <div className="wb-acts">
      <button
        type="button"
        className="wb-ab"
        data-testid="wb-card-unsend"
        aria-label={t('workbench.card.unsend')}
        title={t('workbench.card.unsend')}
        onClick={unsend}
      >
        <X size={14} strokeWidth={1.75} aria-hidden="true" />
      </button>
    </div>
  ) : canConfirm ? (
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
                data-change-option={isChangeOption(option, ask?.change_option) ? 'true' : undefined}
                onClick={() => {
                  // Batch 1110 item 2: "needs changes" asks for the
                  // correction instead of sending the bare option.
                  if (isChangeOption(option, ask?.change_option)) {
                    setReply(changeOptionPrefill(option));
                    requestAnimationFrame(() => replyRef.current?.focus());
                    return;
                  }
                  void sendReply(option);
                }}
                disabled={sending}
              >
                {option}
              </button>
            ))}
          </div>
        )}
        <div className="wb-rbox">
          <input
            ref={replyRef}
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
    picking ? 'is-picking' : '',
  ].filter(Boolean).join(' ');

  // The project's place on the meta line: 自动 while picking, else the name.
  const slotText = picking ? t('workbench.composer.auto') : project;
  const marked = item.marked;
  const slotInner = slotText !== null ? (
    <>
      <span className={`wb-mk${marked ? '' : ' is-off'}`} title={marked && !picking ? t('workbench.card.autoPicked') : undefined} data-testid="wb-card-mark" data-on={marked ? 'true' : 'false'}>
        <span className="wb-mk-in"><span className="wb-mk-gl"><Sparkles size={11} strokeWidth={2} aria-hidden="true" /></span></span>
      </span>
      <SlotName text={slotText} shim={picking} phone={phone} />
      {onPickProject && <span className="wb-cv"><ChevronDown size={11} strokeWidth={2.2} aria-hidden="true" /></span>}
    </>
  ) : null;
  const slot = slotInner === null ? null : onPickProject ? (
    <button
      ref={slotRef}
      type="button"
      className="wb-slot"
      data-testid="wb-card-project"
      aria-haspopup="listbox"
      aria-expanded={pickerOpen}
      aria-disabled={slotLive ? undefined : 'true'}
      aria-label={t('workbench.card.projectAria', { name: slotText ?? '' })}
      title={picking ? undefined : slotText ?? undefined}
      onClick={() => { if (slotLive) openPicker(!pickerOpen); }}
    >
      {slotInner}
    </button>
  ) : (
    <span className="wb-slot is-static" data-testid="wb-card-project">{slotInner}</span>
  );
  const dotClass = item.moving ? 'is-idle' : `is-${item.dot}`;

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
        <span className={`wb-dot ${dotClass}`} aria-hidden="true" />
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
          {/* The agent's context as markdown, clamped; the session shows all
              of it. */}
          {ask?.detail && (
            <div className="wb-askdetail" data-testid="wb-card-askdetail">
              <MarkdownContent content={ask.detail} />
            </div>
          )}
          <span className="wb-meta" data-testid="wb-card-meta">
            <span className={`wb-lg${slug === MAIN_AGENT_HANDLE ? ' is-orb' : ''}`} title={agentLabel}>
              {icon.src ? <img src={icon.src} alt={agentLabel} /> : null}
            </span>
            {slot}
            {row?.source && (() => {
              const src = row.source;
              const nm = feedSourceName(t, src.type, src.name);
              const inner = <><SourceMark type={src.type} size="sm" /><span className="t">{nm}</span></>;
              return (
                <>
                  {slot ? <span className="wb-dsep"> · </span> : null}
                  {onOpenSource ? (
                    <button type="button" className="wb-src" title={nm} onClick={() => onOpenSource(item)} data-testid="wb-card-source">{inner}</button>
                  ) : <span className="wb-src" title={nm} data-testid="wb-card-source">{inner}</span>}
                </>
              );
            })()}
            {(detail || overdueDue) && (
              <span className={`wb-dt${detail?.warn ? ' is-warn' : ''}`}>
                {detail && (
                  <>
                    {slot || row?.source ? <span className="wb-dsep"> · </span> : null}
                    {detail.warn ? <span className="wb-warn">{detail.text}</span> : detail.text}
                  </>
                )}
                {overdueDue && (
                  <>
                    {slot || detail ? <span className="wb-dsep"> · </span> : null}
                    <span className="wb-warn">{overdueDue}</span>
                  </>
                )}
              </span>
            )}
          </span>
          {!phone && replyRow}
        </span>
        <span className="wb-rz">
          <span className="wb-time">{timeText}</span>
        </span>
        {(!phone || picking) && acts}
      </div>
      {phone && replyRow}
      {pickerOpen && slotLive && (
        <AnchoredProjectPicker
          anchor={slotRef.current}
          projects={projects}
          selectedId={picking ? null : pid}
          auto={picking ? 'ticked' : 'none'}
          onAuto={() => openPicker(false)}
          onPick={pickProject}
          onClose={() => openPicker(false)}
          testId="wb-pop-card-project"
        />
      )}
    </div>
  );
}

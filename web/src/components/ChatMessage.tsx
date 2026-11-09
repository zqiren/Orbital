// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { parseAskAnswer, stripTurnNote } from '../utils/askAnswer';
import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Pencil, RotateCw } from 'lucide-react';
import type { DisplayItem } from '../utils/chatTransform';
import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import { parseTriggerHeader } from '../lib/trigger-parsing';
import { parseSkillBlock } from '../lib/skill-parsing';
import { splitQuotesBlock } from '../utils/annotations';
import AttachmentChip from './AttachmentChip';
import SkillChip from './SkillChip';
import MarkdownContent from './MarkdownContent';
import MessageAvatar from './MessageAvatar';
import CopyButton from './CopyButton';
import { useT } from '../i18n/useT';
import { agentTint, MAIN_AGENT_HANDLE } from '../utils/agentIcons';

type MessageItem = Extract<
  DisplayItem,
  { type: 'user_message' | 'agent_message' | 'sub_agent_message' }
>;

interface ChatMessageProps {
  message: MessageItem;
  /**
   * The project's configured agent name. Shown as the sender label for the
   * MANAGEMENT agent's messages (in place of the generic "agent"). Sub-agent
   * rows ignore this — they always show their own handle (claude-code, etc.).
   * Falls back to "agent" when empty/undefined.
   */
  agentName?: string;
  /**
   * Spec 002: absolute project workspace + a handler to open a workspace path
   * in the FilePreviewDrawer. Threaded into MarkdownContent so agent-written
   * paths become clickable. Omitted/undefined → no linkification.
   */
  workspace?: string;
  onOpenPath?: (path: string) => void;
  /**
   * Spec 156 §5.1 / 164 §2.1 — the action row under the body. ChatView owns
   * every gate (running, latest turn, optimistic, automation, cross-session)
   * and passes plain booleans plus stable callbacks, so the memo holds.
   */
  /** 修改: the user's own, persisted message, while the session is idle. */
  canEdit?: boolean;
  /** 重试: exactly one row in the view, the latest turn's anchor. */
  canRetry?: boolean;
  /** This bubble is the inline editor (spec 156 §5.2). */
  editing?: boolean;
  /** The edit's send is in flight (sheet lookup / fork): 发送 waits. */
  editBusy?: boolean;
  onEdit?: (message: MessageItem) => void;
  onEditCancel?: () => void;
  onEditSend?: (message: MessageItem, body: string) => void;
  /** Resolves when the resend's request has answered (the button waits). */
  onRetry?: (message: MessageItem) => Promise<unknown> | void;
}

/**
 * The muted row under a message (spec 156 §5.1, placement A): drawn in the
 * timeline's existing 16px gap, out of the layout flow, so revealing it moves
 * nothing. Desktop shows it on hover or keyboard focus of the message; the
 * phone (no hover) always shows it.
 */
function ActionRow({ side, children }: { side: 'left' | 'right'; children: ReactNode }) {
  return (
    <div
      data-testid="message-actions"
      className={`absolute top-full z-[4] flex h-4 items-center gap-0.5 opacity-0 transition-opacity duration-100 group-hover:opacity-100 group-focus-within:opacity-100 max-md:opacity-100 ${
        side === 'right' ? 'right-0 -mr-[5px] max-md:-mr-[9px]' : 'left-0 -ml-[5px] max-md:-ml-[9px]'
      }`}
    >
      {children}
    </div>
  );
}

const ACTION_BTN =
  'inline-flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-[5px] text-muted transition-colors hover:text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 disabled:cursor-default disabled:opacity-40 max-md:h-[30px] max-md:w-[30px]';
// CopyButton's own colour is text-secondary; the row is muted.
const COPY_BTN = `${ACTION_BTN} !text-muted hover:!text-primary`;

function RetryButton({ onRetry }: { onRetry: () => Promise<unknown> | void }) {
  const t = useT();
  // 164 §2.3: a local waiting flag while the send is answered. The ref makes
  // a double click send once even before the disabled state has rendered.
  const [waiting, setWaiting] = useState(false);
  const inFlight = useRef(false);
  return (
    <button
      type="button"
      data-testid="message-retry"
      aria-label={t('chat.message.retry')}
      title={t('chat.message.retry')}
      disabled={waiting}
      onClick={(e) => {
        e.stopPropagation();
        if (inFlight.current) return;
        inFlight.current = true;
        setWaiting(true);
        void Promise.resolve()
          .then(() => onRetry())
          .catch(() => undefined)
          .finally(() => {
            inFlight.current = false;
            setWaiting(false);
          });
      }}
      className={ACTION_BTN}
    >
      <RotateCw size={12} aria-hidden="true" />
    </button>
  );
}

/**
 * Spec 156 §5.2 — the bubble as an editor. The chips stay above the text,
 * read-only; Enter sends, Shift+Enter is a newline, Enter while an IME is
 * composing never sends (the composer's guard), Esc cancels. 取消 / 发送 sit
 * under it because the phone has no Esc. Desktop keeps the bubble's width
 * cap; the phone gets the full column (owner, 2026-11-10).
 */
function MessageEditor({
  initial,
  chips,
  canSendEmpty,
  busy,
  onCancel,
  onSend,
}: {
  initial: string;
  chips: ReactNode;
  canSendEmpty: boolean;
  busy: boolean;
  onCancel: () => void;
  onSend: (body: string) => void;
}) {
  const t = useT();
  const [text, setText] = useState(initial);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };
  // Caret at the end, the box sized to the text, on open.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    fit();
    el.focus({ preventScroll: true });
    el.setSelectionRange(el.value.length, el.value.length);
  }, []);
  const canSend = !busy && (text.trim().length > 0 || canSendEmpty);
  const send = () => {
    if (canSend) onSend(text.trim());
  };
  return (
    <div className="flex w-[min(70%,680px)] flex-col gap-2 max-md:w-full" data-testid="message-editor">
      <div className="rounded-lg rounded-tr-sm bg-bubble px-3 py-2 shadow-[0_0_0_1px_rgb(83_154_248/0.55)]">
        {chips}
        <textarea
          ref={ref}
          rows={1}
          value={text}
          aria-label={t('chat.message.edit')}
          data-testid="message-editor-input"
          onChange={(e) => {
            setText(e.target.value);
            fit();
          }}
          onKeyDown={(e) => {
            // Same guard as the composer: Enter during IME composition picks a
            // candidate, it never sends.
            if (e.nativeEvent.isComposing || e.keyCode === 229) return;
            if (e.key === 'Escape') {
              e.preventDefault();
              onCancel();
              return;
            }
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          className="block max-h-[min(50vh,420px)] w-full resize-none overflow-y-auto border-0 bg-transparent p-0 text-[13px] leading-[1.55] text-bubble-text outline-none"
        />
      </div>
      <div className="flex justify-end gap-1.5">
        <button
          type="button"
          onClick={onCancel}
          title={t('chat.edit.cancelHint')}
          data-testid="message-editor-cancel"
          className="inline-flex h-7 items-center justify-center whitespace-nowrap rounded-[7px] bg-soft-button px-3 text-xs font-medium text-primary transition-colors hover:bg-soft-button-hover max-md:h-9 max-md:px-4 max-md:text-[13px]"
        >
          {t('chat.edit.cancel')}
        </button>
        <button
          type="button"
          onClick={send}
          disabled={!canSend}
          title={t('chat.edit.sendHint')}
          data-testid="message-editor-send"
          className="inline-flex h-7 items-center justify-center whitespace-nowrap rounded-[7px] bg-accent px-3 text-xs font-medium text-white transition-[filter] hover:brightness-105 disabled:cursor-default disabled:opacity-35 disabled:hover:brightness-100 max-md:h-9 max-md:px-4 max-md:text-[13px]"
        >
          {t('chat.edit.send')}
        </button>
      </div>
    </div>
  );
}

function basename(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

function inferSize(label: string): number {
  // Best-effort: parse the same human label format we produce. Used only for
  // chip display in the user message.
  const m = label.match(/^([\d.]+)\s*(B|KB|MB|GB)$/i);
  if (!m) return 0;
  const value = parseFloat(m[1]);
  const unit = m[2].toUpperCase();
  if (unit === 'B') return value;
  if (unit === 'KB') return Math.round(value * 1024);
  if (unit === 'MB') return Math.round(value * 1024 * 1024);
  if (unit === 'GB') return Math.round(value * 1024 * 1024 * 1024);
  return 0;
}

/** 24-hour HH:MM from an ISO timestamp; empty string if unparseable. */
function formatTime(timestamp: string): string {
  const d = new Date(timestamp);
  if (Number.isNaN(d.getTime())) return '';
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

/**
 * One message row. The two speakers are laid out asymmetrically:
 *
 *   agent / sub-agent  a flat avatar-log row (Design §5) — 26×26 avatar box,
 *                      "<sender> · HH:MM" label line, content underneath with
 *                      no bubble background, border, or rounding.
 *   user               right-anchored, no avatar, a width-capped tint block
 *                      under a right-aligned label line.
 *
 * The asymmetry is deliberate and is a reversal of `33be98b` ("chat
 * conversation bubbles → flat avatar-log"), which had put both speakers on one
 * left edge. `user_message` is 1 of 14 DisplayItem variants — the other
 * thirteen (agent/sub-agent messages, reasoning blocks, tool capsules,
 * approval and fanout cards, budget events, activity markers) all stay left,
 * so this is one row type stepping out of a log rather than a two-sided chat.
 * That is the tradeoff being bought: turn-taking becomes scannable at a glance
 * without reading a single label, at the cost of the user's rows no longer
 * sharing a left edge with everything else in the timeline.
 *
 * The user avatar was dropped rather than mirrored: side already encodes
 * identity once the row moves, so a "ME" square would be pure redundancy
 * paid for out of the right margin the alignment exists to create.
 */
export default function ChatMessage({
  message,
  agentName,
  workspace,
  onOpenPath,
  canEdit = false,
  canRetry = false,
  editing = false,
  editBusy = false,
  onEdit,
  onEditCancel,
  onEditSend,
  onRetry,
}: ChatMessageProps) {
  const t = useT();
  const time = formatTime(message.timestamp);
  // Spec 078 §5.4 step 3: the sent message shows the annotation count as a
  // chip and keeps the quotes block folded behind it.
  const [quotesOpen, setQuotesOpen] = useState(false);

  if (message.type === 'user_message') {
    // An automation run's header line: the heading and the list row already
    // name the automation, so the bubble carries the task text only.
    const { strippedContent: afterTrigger } = parseTriggerHeader(stripTurnNote(message.content));
    const { strippedContent: afterAttachments, attachments } = parseAttachmentsBlock(afterTrigger);
    // Spec 132: a `/`-picked skill's block follows the attachments block.
    const { strippedContent, skill } = parseSkillBlock(afterAttachments);
    // Spec 146: a message answering an ask shows the question above the words.
    const { question: askQuestion, rest: afterAsk } = parseAskAnswer(strippedContent);
    const { text: bodyText, quotes, count: annotationCount } = splitQuotesBlock(afterAsk);
    const hasChips = attachments.length > 0 || skill !== null;
    const hasText = bodyText.length > 0;
    const senderLabel = message.target
      ? t('chat.message.youTo', { target: message.target })
      : t('chat.message.you');

    const chips = hasChips && (
      <div className="flex flex-wrap gap-2 mb-2">
        {skill && (
          <SkillChip name={skill.name} title={skill.path} data-testid="message-skill-chip" />
        )}
        {attachments.map((a, i) => (
          <AttachmentChip
            key={`${i}-${a.path}`}
            filename={basename(a.path)}
            mime={a.mime}
            size={inferSize(a.sizeLabel)}
            status="done"
            localPath={a.local ? a.path : undefined}
          />
        ))}
      </div>
    );
    const showEditor = editing && canEdit && onEditSend !== undefined;
    const showEdit = canEdit && onEdit !== undefined;
    const showRetry = canRetry && onRetry !== undefined;

    return (
      <div
        className="group relative flex flex-col items-end"
        title={message.timestamp}
        data-testid="user-message"
      >
        {/* No avatar on this side. Once the row is on its own edge, position
            already says who spoke, and a "ME" square would only eat the right
            margin the alignment exists to create. The label line stays,
            though — it is the only carrier of `you → @{target}` on @mention
            sends, which is real routing information, not chrome. The copy
            icon moved to the action row under the bubble (spec 156 §5.1). */}
        <div className="font-mono text-[11px] mb-1 flex items-center gap-1.5">
          <span className="text-secondary">{senderLabel}</span>
          {time && <span className="text-muted">· {time}</span>}
        </div>
        {askQuestion && (
          <p className="m-0 mb-1 max-w-[84%] text-xs text-muted line-clamp-2" data-testid="user-message-ask">
            {askQuestion}
          </p>
        )}
        {showEditor ? (
          <MessageEditor
            initial={bodyText}
            chips={chips}
            canSendEmpty={hasChips || annotationCount > 0}
            busy={editBusy}
            onCancel={() => onEditCancel?.()}
            onSend={(body) => onEditSend(message, body)}
          />
        ) : (
          /* The tint block is capped rather than allowed to run the pane's full
             width: the chat pane reaches ~1200px on a maximised window, and an
             uncapped right-aligned "ok" would sit that far from the agent reply
             it answers. `min()` keeps it proportional on narrow/mobile panes and
             bounded on wide ones. The corner notch flips tl→tr to point back at
             the label above it, the way it used to point at the avatar. */
          <div className="float-bubble inline-block max-w-[min(70%,680px)] rounded-lg rounded-tr-sm bg-bubble px-3 py-2 text-[13px] leading-[1.55] text-bubble-text whitespace-pre-wrap break-words">
            {chips}
            {hasText && bodyText}
            {annotationCount > 0 && quotes && (
              <div className={hasText ? 'mt-2' : ''}>
                <button
                  type="button"
                  onClick={() => setQuotesOpen((open) => !open)}
                  aria-expanded={quotesOpen}
                  data-testid="message-annotation-chip"
                  className="inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[11px] text-secondary hover:text-primary transition-colors"
                >
                  {annotationCount === 1
                    ? t('composer.annotations.one')
                    : t('composer.annotations.other', { n: annotationCount })}
                </button>
                {quotesOpen && (
                  <pre
                    data-testid="message-annotation-quotes"
                    className="mt-1.5 font-mono text-[11px] leading-[1.5] text-secondary whitespace-pre-wrap break-words"
                  >
                    {quotes}
                  </pre>
                )}
              </div>
            )}
          </div>
        )}
        {!showEditor && (hasText || showRetry || showEdit) && (
          <ActionRow side="right">
            {/* Copy carries the message's own text, with the <attached_files>
                and <use_skill> blocks already stripped — that block is machine
                markup the user never typed and must not be pasted back. An
                attachment-only message has no body to copy. */}
            {hasText && (
              <CopyButton
                text={bodyText}
                ariaLabel={t('chat.message.copyAria')}
                data-testid="user-message-copy"
                className={COPY_BTN}
              />
            )}
            {showRetry && <RetryButton onRetry={() => onRetry(message)} />}
            {showEdit && (
              <button
                type="button"
                data-testid="message-edit"
                aria-label={t('chat.message.edit')}
                title={t('chat.message.edit')}
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(message);
                }}
                className={ACTION_BTN}
              >
                <Pencil size={12} aria-hidden="true" />
              </button>
            )}
          </ActionRow>
        )}
      </div>
    );
  }

  // Determine if this is a sub-agent message and get the label
  const isSubAgent =
    message.type === 'sub_agent_message' ||
    (message.type === 'agent_message' && message.source && message.source !== 'management' && message.source !== 'user');
  const senderLabel = isSubAgent && message.source
    ? message.source
    : (agentName && agentName.trim() ? agentName : t('chat.message.agent'));

  // FE-A3: header-only mode. Emitted for content-null assistant turns so the
  // capsule that follows has a visible agent anchor (avatar + sender · HH:MM)
  // and does not visually attach to the preceding user message.
  const isHeaderOnly =
    message.type === 'agent_message' && message.isHeaderOnly === true;

  const hasBody = !isHeaderOnly && message.content.length > 0;
  const showRetry = !isHeaderOnly && canRetry && onRetry !== undefined;

  return (
    <div
      className="group flex gap-[10px]"
      title={message.timestamp}
      data-testid={isHeaderOnly ? 'agent-header' : undefined}
    >
      <MessageAvatar variant="agent" agentHandle={isSubAgent ? message.source : undefined} />
      <div className="relative flex-1 min-w-0">
        <div className="font-mono text-[11px] mb-1 flex items-center gap-1.5">
          {/* Dark mode names the agent in its brand colour, lifted toward white
              so it reads on Graphite (palette 1a); light keeps text-secondary. */}
          <span
            className="chat-agent-name text-secondary dark:text-[color-mix(in_srgb,var(--agent-color)_55%,white)]"
            style={{ '--agent-color': agentTint(isSubAgent ? message.source : MAIN_AGENT_HANDLE) } as CSSProperties}
          >
            {senderLabel}
          </span>
          {time && <span className="text-muted">· {time}</span>}
        </div>
        {!isHeaderOnly && (
          // overflow-x clips a card's shadow at this box's edge; the bleed
          // (negative margin + equal padding) moves that edge out so file
          // cards keep their soft sides, and the text does not move.
          <div className="-mx-4 px-4 -my-1.5 py-1.5 text-[13px] leading-[1.55] text-primary break-words overflow-x-auto">
            <MarkdownContent content={message.content} workspace={workspace} onOpenPath={onOpenPath} />
          </div>
        )}
        {/* Header-only rows are an anchor for the capsule that follows and
            carry no body: no row (a copy there would copy nothing). */}
        {(hasBody || showRetry) && (
          <ActionRow side="left">
            {hasBody && (
              <CopyButton
                text={message.content}
                ariaLabel={t('chat.message.copyAria')}
                data-testid="agent-message-copy"
                className={COPY_BTN}
              />
            )}
            {showRetry && <RetryButton onRetry={() => onRetry(message)} />}
          </ActionRow>
        )}
      </div>
    </div>
  );
}

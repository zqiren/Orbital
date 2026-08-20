// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { DisplayItem } from '../utils/chatTransform';
import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import AttachmentChip from './AttachmentChip';
import MarkdownContent from './MarkdownContent';
import MessageAvatar from './MessageAvatar';
import CopyButton from './CopyButton';
import { useT } from '../i18n/useT';

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
export default function ChatMessage({ message, agentName, workspace, onOpenPath }: ChatMessageProps) {
  const t = useT();
  const time = formatTime(message.timestamp);

  if (message.type === 'user_message') {
    const { strippedContent, attachments } = parseAttachmentsBlock(message.content);
    const hasChips = attachments.length > 0;
    const hasText = strippedContent.length > 0;
    const senderLabel = message.target
      ? t('chat.message.youTo', { target: message.target })
      : t('chat.message.you');

    return (
      <div
        className="flex flex-col items-end"
        title={message.timestamp}
        data-testid="user-message"
      >
        {/* No avatar on this side. Once the row is on its own edge, position
            already says who spoke, and a "ME" square would only eat the right
            margin the alignment exists to create. The label line stays,
            though — it is the only carrier of `you → @{target}` on @mention
            sends, which is real routing information, not chrome. */}
        <div className="group font-mono text-[11px] mb-1 flex items-center gap-1.5">
          <span className="text-secondary">{senderLabel}</span>
          {time && <span className="text-muted">· {time}</span>}
          {/* Copy carries the message's own text, with the <attached_files>
              block already stripped — that block is machine markup the user
              never typed and must not be pasted back. */}
          <CopyButton
            text={strippedContent}
            ariaLabel={t('chat.message.copyAria')}
            data-testid="user-message-copy"
            className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
          />
        </div>
        {/* The tint block is capped rather than allowed to run the pane's full
            width: the chat pane reaches ~1200px on a maximised window, and an
            uncapped right-aligned "ok" would sit that far from the agent reply
            it answers. `min()` keeps it proportional on narrow/mobile panes and
            bounded on wide ones. The corner notch flips tl→tr to point back at
            the label above it, the way it used to point at the avatar. */}
        <div className="inline-block max-w-[min(70%,680px)] rounded-lg rounded-tr-sm bg-sidebar px-3 py-2 text-[13px] leading-[1.55] text-primary whitespace-pre-wrap break-words">
          {hasChips && (
            <div className="flex flex-wrap gap-2 mb-2">
              {attachments.map((a, i) => (
                <AttachmentChip
                  key={`${i}-${a.path}`}
                  filename={basename(a.path)}
                  mime={a.mime}
                  size={inferSize(a.sizeLabel)}
                  status="done"
                />
              ))}
            </div>
          )}
          {hasText && strippedContent}
        </div>
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

  return (
    <div
      className="flex gap-[10px]"
      title={message.timestamp}
      data-testid={isHeaderOnly ? 'agent-header' : undefined}
    >
      <MessageAvatar variant="agent" agentHandle={isSubAgent ? message.source : undefined} />
      <div className="flex-1 min-w-0">
        <div className="group font-mono text-[11px] mb-1 flex items-center gap-1.5">
          <span className="text-secondary">{senderLabel}</span>
          {time && <span className="text-muted">· {time}</span>}
          {/* Header-only rows are an anchor for the capsule that follows and
              carry no body — a copy button there would copy nothing. */}
          {!isHeaderOnly && (
            <CopyButton
              text={message.content}
              ariaLabel={t('chat.message.copyAria')}
              data-testid="agent-message-copy"
              className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
            />
          )}
        </div>
        {!isHeaderOnly && (
          <div className="text-[13px] leading-[1.55] text-primary break-words overflow-x-auto">
            <MarkdownContent content={message.content} workspace={workspace} onOpenPath={onOpenPath} />
          </div>
        )}
      </div>
    </div>
  );
}

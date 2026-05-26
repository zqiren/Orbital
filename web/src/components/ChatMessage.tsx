// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { DisplayItem } from '../utils/chatTransform';
import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import AttachmentChip from './AttachmentChip';
import MarkdownContent from './MarkdownContent';
import MessageAvatar from './MessageAvatar';

type MessageItem = Extract<
  DisplayItem,
  { type: 'user_message' | 'agent_message' | 'sub_agent_message' }
>;

interface ChatMessageProps {
  message: MessageItem;
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
 * Up-to-2 uppercase chars for the user avatar. There is no first-class
 * "current user" identity in this app, so derive from a meaningful source
 * label when present, else fall back to the static "ME" placeholder.
 */
function userInitials(source?: string): string {
  const s = (source ?? '').trim();
  if (!s || s === 'user' || s === 'management') return 'ME';
  return s.slice(0, 2).toUpperCase();
}

/**
 * One message rendered as a flat avatar-log row (Design §5): a 26×26 avatar
 * box, a "<sender> · HH:MM" label line, and the content underneath — no
 * bubble background, border, or rounding. User and agent rows share the same
 * left-aligned layout.
 */
export default function ChatMessage({ message }: ChatMessageProps) {
  const time = formatTime(message.timestamp);

  if (message.type === 'user_message') {
    const { strippedContent, attachments } = parseAttachmentsBlock(message.content);
    const hasChips = attachments.length > 0;
    const hasText = strippedContent.length > 0;
    const senderLabel = message.target ? `you → @${message.target}` : 'you';

    return (
      <div className="flex gap-[10px]" title={message.timestamp}>
        <MessageAvatar variant="user" label={userInitials()} />
        <div className="flex-1 min-w-0">
          <div className="font-mono text-[11px] mb-1">
            <span className="text-secondary">{senderLabel}</span>
            {time && <span className="text-muted"> · {time}</span>}
          </div>
          <div className="text-[13px] leading-[1.55] text-primary whitespace-pre-wrap break-words">
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
      </div>
    );
  }

  // Determine if this is a sub-agent message and get the label
  const isSubAgent =
    message.type === 'sub_agent_message' ||
    (message.type === 'agent_message' && message.source && message.source !== 'management' && message.source !== 'user');
  const senderLabel = isSubAgent && message.source ? message.source : 'agent';

  return (
    <div className="flex gap-[10px]" title={message.timestamp}>
      <MessageAvatar variant="agent" />
      <div className="flex-1 min-w-0">
        <div className="font-mono text-[11px] mb-1">
          <span className="text-secondary">{senderLabel}</span>
          {time && <span className="text-muted"> · {time}</span>}
        </div>
        <div className="text-[13px] leading-[1.55] text-primary break-words overflow-x-auto">
          <MarkdownContent content={message.content} />
        </div>
      </div>
    </div>
  );
}

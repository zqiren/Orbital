// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { DisplayItem } from '../utils/chatTransform';
import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import AttachmentChip from './AttachmentChip';
import MarkdownContent from './MarkdownContent';

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
  // chip display in the user bubble.
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

export default function ChatMessage({ message }: ChatMessageProps) {
  if (message.type === 'user_message') {
    const { strippedContent, attachments } = parseAttachmentsBlock(message.content);
    const hasChips = attachments.length > 0;
    const hasText = strippedContent.length > 0;

    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[75%] max-md:max-w-[85%]">
          {message.target && (
            <div className="text-right mb-0.5">
              <span className="text-xs text-accent font-medium">@{message.target}</span>
            </div>
          )}
          <div
            className="bg-card-hover rounded-lg px-4 py-2 whitespace-pre-wrap break-words overflow-x-auto text-sm"
            title={message.timestamp}
          >
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
  const senderLabel = isSubAgent && message.source ? message.source : 'Agent';

  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[75%] max-md:max-w-[85%]">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-sm font-medium text-secondary">{senderLabel}</span>
        </div>
        <div
          className={`bg-background border border-border rounded-lg px-4 py-2 break-words overflow-x-auto${isSubAgent ? ' border-l-2 border-l-accent' : ''}`}
          title={message.timestamp}
        >
          <MarkdownContent content={message.content} />
        </div>
      </div>
    </div>
  );
}

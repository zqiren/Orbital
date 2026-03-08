// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { DisplayItem } from '../utils/chatTransform';
import MarkdownContent from './MarkdownContent';

type MessageItem = Extract<
  DisplayItem,
  { type: 'user_message' | 'agent_message' | 'sub_agent_message' }
>;

interface ChatMessageProps {
  message: MessageItem;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  if (message.type === 'user_message') {
    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[75%] max-md:max-w-[85%]">
          {message.target && (
            <div className="text-right mb-0.5">
              <span className="text-xs text-blue-500 font-medium">@{message.target}</span>
            </div>
          )}
          <div
            className="bg-card-hover rounded-lg px-4 py-2 whitespace-pre-wrap break-words overflow-x-auto text-sm"
            title={message.timestamp}
          >
            {message.content}
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
          className={`bg-background border border-border rounded-lg px-4 py-2 break-words overflow-x-auto${isSubAgent ? ' border-l-2 border-l-blue-400' : ''}`}
          title={message.timestamp}
        >
          <MarkdownContent content={message.content} />
        </div>
      </div>
    </div>
  );
}

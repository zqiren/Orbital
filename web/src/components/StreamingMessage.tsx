// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import MarkdownContent from './MarkdownContent';
import MessageAvatar from './MessageAvatar';

interface StreamingMessageProps {
  text: string;
  source: string;
  isComplete: boolean;
}

/** 24-hour HH:MM for "now" — the streaming message is always live. */
function nowTime(): string {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

export default function StreamingMessage({
  text,
  source,
  isComplete,
}: StreamingMessageProps) {
  const isSubAgent = source !== 'management' && source !== 'user';
  const senderLabel = isSubAgent ? source : 'agent';

  return (
    <div className="flex gap-[10px]">
      <MessageAvatar variant="agent" />
      <div className="flex-1 min-w-0">
        <div className="font-mono text-[11px] mb-1">
          <span className="text-secondary">{senderLabel}</span>
          <span className="text-muted"> · {nowTime()}</span>
        </div>
        <div className="text-[13px] leading-[1.55] text-primary break-words overflow-x-auto">
          <MarkdownContent content={text} />
          {!isComplete && <span className="streaming-cursor">|</span>}
        </div>
      </div>
      <style>{`
        .streaming-cursor {
          display: inline;
          animation: blink 0.8s step-end infinite;
          color: var(--color-accent);
          font-weight: 600;
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}

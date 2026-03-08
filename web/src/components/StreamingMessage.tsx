// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import MarkdownContent from './MarkdownContent';

interface StreamingMessageProps {
  text: string;
  source: string;
  isComplete: boolean;
}

export default function StreamingMessage({
  text,
  source,
  isComplete,
}: StreamingMessageProps) {
  const isSubAgent = source !== 'management' && source !== 'user';

  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[75%] max-md:max-w-[85%]">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-sm font-medium text-secondary">Agent</span>
          {isSubAgent && (
            <span className="text-sm font-mono bg-sidebar px-1.5 py-0.5 rounded">
              @{source}
            </span>
          )}
        </div>
        <div className="bg-background border border-border rounded-lg px-4 py-2 break-words overflow-x-auto">
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

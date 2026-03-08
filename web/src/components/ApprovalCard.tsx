// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { useAgent } from '../hooks/useAgent';

interface ApprovalCardProps {
  approval: {
    what: string;
    tool_name: string;
    tool_call_id: string;
    tool_args: Record<string, unknown>;
    recent_activity: Array<{ role: string; content: string | null }>;
    reasoning?: string;
  };
  projectId: string;
  resolved?: 'approved' | 'denied';
  onResolve?: (toolCallId: string, resolution: 'approved' | 'denied') => void;
}

function renderToolArgs(toolName: string, toolArgs: Record<string, unknown>) {
  switch (toolName) {
    case 'shell':
      return (
        <code className="block text-xs font-mono bg-[#1a1a2e] text-green-400 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-40">
          {String(toolArgs.command ?? '')}
        </code>
      );
    case 'read':
      return (
        <span className="text-xs font-mono text-secondary">
          {String(toolArgs.file_path ?? toolArgs.path ?? '')}
        </span>
      );
    case 'write':
    case 'edit': {
      const filePath = String(toolArgs.file_path ?? toolArgs.path ?? '');
      const content = String(toolArgs.content ?? toolArgs.new_content ?? '');
      const truncated = content.length > 500 ? content.slice(0, 500) + '...' : content;
      return (
        <div className="space-y-1">
          <span className="text-xs font-mono text-secondary">{filePath}</span>
          {content && (
            <details className="text-xs">
              <summary className="cursor-pointer text-secondary hover:text-primary">
                Show content
              </summary>
              <pre className="font-mono bg-sidebar rounded-lg p-2 mt-1 overflow-x-auto whitespace-pre-wrap max-h-40">
                {truncated}
              </pre>
            </details>
          )}
        </div>
      );
    }
    case 'browser':
      return (
        <span className="text-xs text-secondary">
          <span className="font-semibold">{String(toolArgs.action ?? '')}</span>
          {toolArgs.target ? ` \u2014 ${String(toolArgs.target)}` : ''}
        </span>
      );
    default:
      return (
        <pre className="text-xs font-mono bg-sidebar rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-40">
          {JSON.stringify(toolArgs, null, 2)}
        </pre>
      );
  }
}

export default function ApprovalCard({
  approval,
  projectId,
  resolved,
  onResolve,
}: ApprovalCardProps) {
  const [replyText, setReplyText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [localResolution, setLocalResolution] = useState<'approved' | 'denied' | null>(null);
  const [showDenyInput, setShowDenyInput] = useState(false);
  const [denyFeedback, setDenyFeedback] = useState('');
  const [reasoningExpanded, setReasoningExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMobile] = useState(() => window.innerWidth < 768);
  const { approveToolCall, denyToolCall } = useAgent();

  const resolution = resolved ?? localResolution;

  if (resolution) {
    const isApproved = resolution === 'approved';
    return (
      <div className="mb-3 px-4 py-2 rounded-lg bg-sidebar text-sm">
        <span className={isApproved ? 'text-success' : 'text-error'}>
          {isApproved ? '\u2713 Approved' : '\u2717 Denied'}:
        </span>{' '}
        <span className="text-secondary">{approval.what}</span>
      </div>
    );
  }

  const recentLines = approval.recent_activity
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && m.content)
    .slice(-4)
    .map((m) => ({
      role: m.role as 'user' | 'assistant',
      content: String(m.content),
    }));

  async function handleApprove(approveAll = false) {
    setSubmitting(true);
    setError(null);
    try {
      await approveToolCall(projectId, approval.tool_call_id, replyText || undefined, approveAll || undefined);
      setLocalResolution('approved');
      onResolve?.(approval.tool_call_id, 'approved');
    } catch {
      setSubmitting(false);
      setError('Failed to submit. Please try again.');
    }
  }

  function handleDenyClick() {
    setShowDenyInput(true);
  }

  async function handleDenyConfirm() {
    setSubmitting(true);
    setError(null);
    try {
      await denyToolCall(projectId, approval.tool_call_id, denyFeedback.trim() || replyText || 'Denied by user');
      setLocalResolution('denied');
      onResolve?.(approval.tool_call_id, 'denied');
    } catch {
      setSubmitting(false);
      setError('Failed to submit. Please try again.');
    }
  }

  return (
    <div className="mb-3 rounded-lg border border-border overflow-hidden">
      <div className="bg-warning/10 px-4 py-1.5 border-b border-border">
        <span className="text-warning font-semibold text-sm">APPROVAL NEEDED</span>
      </div>

      <div className="px-4 py-3 space-y-3">
        <p className="text-sm font-medium text-primary">
          {approval.what}
        </p>

        {approval.reasoning && (
          <div className="text-sm">
            <span className="text-xs text-secondary font-medium">Why: </span>
            <span className={`text-primary whitespace-pre-wrap break-words ${!reasoningExpanded ? 'line-clamp-4 md:line-clamp-none' : ''}`}>
              {approval.reasoning}
            </span>
            {approval.reasoning.length > 200 && (
              <button
                type="button"
                onClick={() => setReasoningExpanded(!reasoningExpanded)}
                className="text-xs text-accent hover:underline ml-1 md:hidden cursor-pointer"
              >
                {reasoningExpanded ? 'Show less' : 'Show more'}
              </button>
            )}
          </div>
        )}

        <details open={!isMobile}>
          <summary className="text-xs text-secondary cursor-pointer select-none mb-1">
            Action details
          </summary>
          {renderToolArgs(approval.tool_name, approval.tool_args)}
        </details>

        {recentLines.length > 0 && (
          <details open={!isMobile} className="text-xs text-secondary">
            <summary className="font-medium cursor-pointer select-none mb-0.5">
              Recent context
            </summary>
            <div className="space-y-0.5">
              {recentLines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  <span className="font-semibold text-secondary">
                    {line.role === 'user' ? 'You:' : 'Agent:'}
                  </span>{' '}
                  {line.content.length > 300 ? line.content.slice(0, 300) + '...' : line.content}
                </div>
              ))}
            </div>
          </details>
        )}

        <input
          type="text"
          placeholder="Optional guidance for the agent..."
          value={replyText}
          onChange={(e) => setReplyText(e.target.value)}
          className="w-full text-sm px-3 py-1.5 rounded-lg border border-border bg-sidebar focus:outline-none focus:border-accent max-md:min-h-[44px]"
        />

        {showDenyInput && (
          <div className="space-y-2">
            <input
              type="text"
              placeholder="Why are you denying? (optional)"
              value={denyFeedback}
              onChange={(e) => setDenyFeedback(e.target.value)}
              autoFocus
              className="w-full text-sm px-3 py-1.5 rounded-lg border border-error/30 bg-sidebar focus:outline-none focus:border-error max-md:min-h-[44px]"
            />
            <div className="flex flex-col md:flex-row gap-2">
              <button
                type="button"
                onClick={() => setShowDenyInput(false)}
                onTouchEnd={(e) => { e.preventDefault(); setShowDenyInput(false); }}
                className="px-4 py-1.5 text-sm font-medium rounded-lg border border-border text-secondary hover:bg-card-hover transition-colors duration-150 cursor-pointer w-full md:w-auto min-h-[44px]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDenyConfirm}
                onTouchEnd={(e) => { e.preventDefault(); handleDenyConfirm(); }}
                disabled={submitting}
                className="px-4 py-1.5 text-sm font-medium rounded-lg border border-error/40 text-error hover:bg-error/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
              >
                Confirm Deny
              </button>
            </div>
          </div>
        )}

        {!showDenyInput && (
          <div className="flex flex-col md:flex-row gap-2 md:justify-end sticky bottom-0 md:static bg-bg pt-2 md:pt-0 -mx-4 px-4 md:mx-0 md:px-0 pb-1 md:pb-0">
            <button
              type="button"
              onClick={handleDenyClick}
              onTouchEnd={(e) => { e.preventDefault(); handleDenyClick(); }}
              disabled={submitting}
              className="px-4 py-1.5 text-sm font-medium rounded-lg border border-border text-secondary hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
            >
              Deny
            </button>
            <button
              type="button"
              onClick={() => handleApprove(true)}
              onTouchEnd={(e) => { e.preventDefault(); handleApprove(true); }}
              disabled={submitting}
              title="Approve and auto-approve all actions for 10 minutes"
              className="px-4 py-1.5 text-sm font-medium rounded-lg border border-accent text-accent hover:bg-accent/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
            >
              Auto-approve 10 min
            </button>
            <button
              type="button"
              onClick={() => handleApprove()}
              onTouchEnd={(e) => { e.preventDefault(); handleApprove(); }}
              disabled={submitting}
              className="px-4 py-1.5 text-sm font-medium rounded-lg bg-accent text-white hover:opacity-90 transition-opacity duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
            >
              Approve
            </button>
          </div>
        )}

        {error && (
          <p className="text-xs text-error mt-1">{error}</p>
        )}
      </div>
    </div>
  );
}

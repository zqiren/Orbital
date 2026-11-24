// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo, useState } from 'react';
import { useAgent } from '../hooks/useAgent';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import { describeLiveActivity } from '../utils/chatTransform';
import { allowlistExecutable } from './UnsandboxedRunsSection';
import type { UnsandboxedRunDecision } from '../types';
import AgentGlyph from './welcome/AgentGlyph';
import { MAIN_AGENT_HANDLE } from '../utils/agentIcons';

type TFunc = ReturnType<typeof useT>;

interface ApprovalCardProps {
  approval: {
    what: string;
    tool_name: string;
    tool_call_id: string;
    tool_args: Record<string, unknown>;
    recent_activity: Array<{ role: string; content: string | null }>;
    reasoning?: string;
    /** Spec 117 D4/D5: `"unsandboxed_run"` — the sandbox blocked `command`
     *  for `reason`; the card offers once / always / turn off safe mode. */
    kind?: string;
    command?: string;
    /** The program "Always allow" allowlists (daemon-computed); `""` = the
     *  command cannot be allowlisted, so the answer is not offered. */
    executable?: string;
    reason?: string;
  };
  projectId: string;
  /** F1 session_id this approval belongs to (the viewed/holder session).
   *  Threaded into approve/deny so the backend resolves the SPECIFIC
   *  session's pending tool call, not just the project's default one. */
  sessionId?: string;
  resolved?: 'approved' | 'denied';
  onResolve?: (toolCallId: string, resolution: 'approved' | 'denied') => void;
  /** Spec 195: who asks — a worker's handle; absent = Orbital. Its mark sits
   *  in the header, so two agents asking at once are told apart. */
  asker?: string;
  /** The asker's name: the mark's tooltip and accessible label. */
  askerName?: string;
}

function renderToolArgs(toolName: string, toolArgs: Record<string, unknown>, t: TFunc) {
  switch (toolName) {
    case 'shell':
      return (
        <code className="block text-caption font-mono bg-[#1a1a2e] text-green-400 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-40">
          {String(toolArgs.command ?? '')}
        </code>
      );
    case 'read':
      return (
        <span className="text-caption font-mono text-secondary">
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
          <span className="text-caption font-mono text-secondary">{filePath}</span>
          {content && (
            <details className="text-caption">
              <summary className="cursor-pointer text-secondary hover:text-primary">
                {t('approval.showContent')}
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
        <span className="text-caption text-secondary">
          <span className="font-medium">{String(toolArgs.action ?? '')}</span>
          {toolArgs.target ? ` \u2014 ${String(toolArgs.target)}` : ''}
        </span>
      );
    case 'request_network_access': {
      const domain = String(toolArgs.domain ?? '');
      const reason = String(toolArgs.reason ?? '');
      const intent = String(toolArgs.intent ?? '');
      return (
        <div className="space-y-1">
          <div className="text-caption font-mono font-medium">{domain}</div>
          {intent && <div className="text-caption text-secondary">{intent}</div>}
          {reason && <div className="text-caption text-secondary">{reason}</div>}
          <div className="text-caption text-secondary opacity-50">{t('approval.network.permanentNote')}</div>
        </div>
      );
    }
    default:
      return (
        <pre className="text-caption font-mono bg-sidebar rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-40">
          {JSON.stringify(toolArgs, null, 2)}
        </pre>
      );
  }
}

export default function ApprovalCard({
  approval,
  projectId,
  sessionId,
  resolved,
  onResolve,
  asker,
  askerName,
}: ApprovalCardProps) {
  const t = useT();
  const { locale } = useLocale();
  // Spec 117: a sandbox denial the agent may run outside the sandbox. Not a
  // separate card — same chrome, same deny, different answers.
  const isUnsandboxedRun = approval.kind === 'unsandboxed_run';
  // Spec 195: a worker's question the user answers. Same card, same routes:
  // the answer is the approve route's reply text (or an option id as its
  // response payload), Deny declines it.
  const isQuestion = approval.kind === 'question';
  const unsandboxedCommand = approval.command ?? String(approval.tool_args?.command ?? '');
  const alwaysAllowExe = allowlistExecutable(unsandboxedCommand, approval.executable);
  // approval.what is backend-authored English; the payload carries the
  // structured tool call, so derive a localized sentence from it and keep
  // the wire prose only as the unknown-tool fallback.
  const localizedWhat = useMemo(
    () =>
      isUnsandboxedRun
        ? translate(locale, 'approval.unsandboxed.what')
        : describeLiveActivity(
          approval.tool_name,
          approval.tool_args,
          undefined,
          (k, v) => translate(locale, k, v),
          approval.what,
        ),
    [locale, approval, isUnsandboxedRun],
  );
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

  if (isQuestion) {
    return (
      <QuestionCard
        approval={approval}
        projectId={projectId}
        sessionId={sessionId}
        resolution={resolution}
        onResolved={(r) => {
          setLocalResolution(r);
          onResolve?.(approval.tool_call_id, r);
        }}
        asker={asker}
        askerName={askerName}
      />
    );
  }

  if (resolution) {
    const isApproved = resolution === 'approved';
    return (
      <div className="mb-3 shrink-0 px-4 py-2 rounded-lg bg-sidebar text-body">
        <span className={isApproved ? 'text-success' : 'text-error'}>
          {isApproved ? t('approval.approved') : t('approval.denied')}:
        </span>{' '}
        <span className="text-secondary">{localizedWhat}</span>
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

  async function handleApprove(approveAll = false, decision?: UnsandboxedRunDecision) {
    setSubmitting(true);
    setError(null);
    try {
      await approveToolCall(projectId, approval.tool_call_id, replyText || undefined, approveAll || undefined, sessionId, decision);
      setLocalResolution('approved');
      onResolve?.(approval.tool_call_id, 'approved');
    } catch {
      setSubmitting(false);
      setError(t('approval.submitError'));
    }
  }

  function handleDenyClick() {
    setShowDenyInput(true);
  }

  async function handleDenyConfirm() {
    setSubmitting(true);
    setError(null);
    try {
      await denyToolCall(projectId, approval.tool_call_id, denyFeedback.trim() || replyText || t('approval.deniedByUser'), sessionId);
      setLocalResolution('denied');
      onResolve?.(approval.tool_call_id, 'denied');
    } catch {
      setSubmitting(false);
      setError(t('approval.submitError'));
    }
  }

  // Codex approvals advertise their valid decisions; "cancel" ends the turn
  // immediately (no final answer) vs plain deny which lets the agent adapt.
  const canDenyAndStop = Array.isArray(approval.tool_args?.availableDecisions)
    && (approval.tool_args.availableDecisions as unknown[]).includes('cancel');

  async function handleDenyAndStop() {
    setSubmitting(true);
    setError(null);
    try {
      await denyToolCall(projectId, approval.tool_call_id, denyFeedback.trim() || replyText || t('approval.deniedByUser'), sessionId, true);
      setLocalResolution('denied');
      onResolve?.(approval.tool_call_id, 'denied');
    } catch {
      setSubmitting(false);
      setError(t('approval.submitError'));
    }
  }

  return (
    <div className="mb-3 shrink-0 rounded-lg border border-warning/50 border-l-[3px] border-l-warning bg-warning-surface overflow-hidden">
      <div className="px-4 py-1.5 border-b border-warning/20 flex items-center gap-2">
        <span data-testid="approval-asker" data-agent={asker ?? MAIN_AGENT_HANDLE} className="inline-flex">
          <AgentGlyph slug={asker ?? MAIN_AGENT_HANDLE} size={16} title={askerName ?? asker ?? 'Orbital'} />
        </span>
        <span className="type-label text-warning-text">{t('approval.header')}</span>
      </div>

      <div className="px-4 py-3 space-y-3">
        <p className="text-body font-medium text-primary">
          {localizedWhat}
        </p>

        {approval.reasoning && (
          <div className="text-body">
            <span className="text-caption text-secondary font-medium">{t('approval.why')} </span>
            <span className={`text-primary whitespace-pre-wrap break-words ${!reasoningExpanded ? 'line-clamp-4 md:line-clamp-none' : ''}`}>
              {approval.reasoning}
            </span>
            {approval.reasoning.length > 200 && (
              <button
                type="button"
                onClick={() => setReasoningExpanded(!reasoningExpanded)}
                className="text-caption text-accent hover:underline ml-1 md:hidden cursor-pointer"
              >
                {reasoningExpanded ? t('approval.showLess') : t('approval.showMore')}
              </button>
            )}
          </div>
        )}

        <details open={!isMobile}>
          <summary className="text-caption text-secondary cursor-pointer select-none mb-1">
            {t('approval.actionDetails')}
          </summary>
          {isUnsandboxedRun ? (
            <div className="space-y-1.5">
              <code className="block text-caption font-mono bg-[#1a1a2e] text-green-400 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-40">
                {unsandboxedCommand}
              </code>
              {approval.reason && (
                <p className="text-caption text-secondary whitespace-pre-wrap break-words">
                  {approval.reason}
                </p>
              )}
            </div>
          ) : (
            renderToolArgs(approval.tool_name, approval.tool_args, t)
          )}
        </details>

        {recentLines.length > 0 && (
          <details open={!isMobile} className="text-caption text-secondary">
            <summary className="font-medium cursor-pointer select-none mb-0.5">
              {t('approval.recentContext')}
            </summary>
            <div className="space-y-0.5">
              {recentLines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  <span className="font-medium text-secondary">
                    {line.role === 'user' ? t('approval.you') : t('approval.agent')}
                  </span>{' '}
                  {line.content.length > 300 ? line.content.slice(0, 300) + '...' : line.content}
                </div>
              ))}
            </div>
          </details>
        )}

        <input
          type="text"
          placeholder={t('approval.guidance.placeholder')}
          value={replyText}
          onChange={(e) => setReplyText(e.target.value)}
          className="w-full text-body px-3 py-1.5 rounded-lg border border-border bg-sidebar focus:outline-none focus:border-accent max-md:min-h-[44px]"
        />

        {showDenyInput && (
          <div className="space-y-2">
            <input
              type="text"
              placeholder={t('approval.denyReason.placeholder')}
              value={denyFeedback}
              onChange={(e) => setDenyFeedback(e.target.value)}
              autoFocus
              className="w-full text-body px-3 py-1.5 rounded-lg border border-error/30 bg-sidebar focus:outline-none focus:border-error max-md:min-h-[44px]"
            />
            <div className="flex flex-col md:flex-row gap-2">
              <button
                type="button"
                onClick={() => setShowDenyInput(false)}
                onTouchEnd={(e) => { e.preventDefault(); setShowDenyInput(false); }}
                className="px-4 py-1.5 text-body font-medium rounded-lg border border-border text-secondary hover:bg-card-hover transition-colors duration-150 cursor-pointer w-full md:w-auto min-h-[44px]"
              >
                {t('approval.cancel')}
              </button>
              <button
                type="button"
                onClick={handleDenyConfirm}
                onTouchEnd={(e) => { e.preventDefault(); handleDenyConfirm(); }}
                disabled={submitting}
                className="px-4 py-1.5 text-body font-medium rounded-lg border border-error/40 text-error hover:bg-error/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
              >
                {t('approval.confirmDeny')}
              </button>
              {canDenyAndStop && (
                <button
                  type="button"
                  onClick={handleDenyAndStop}
                  onTouchEnd={(e) => { e.preventDefault(); handleDenyAndStop(); }}
                  disabled={submitting}
                  title={t('approval.denyAndStop.title')}
                  className="px-4 py-1.5 text-body font-medium rounded-lg border border-error/40 text-error hover:bg-error/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                >
                  {t('approval.denyAndStop')}
                </button>
              )}
            </div>
          </div>
        )}

        {!showDenyInput && (
          <div className="flex flex-col md:flex-row gap-2 md:justify-end pt-2 md:pt-0 pb-1 md:pb-0">
            <button
              type="button"
              onClick={handleDenyClick}
              onTouchEnd={(e) => { e.preventDefault(); handleDenyClick(); }}
              disabled={submitting}
              className="border border-error/40 text-error bg-background rounded-sm text-micro px-2.5 py-1 font-medium hover:bg-error/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
            >
              {t('credCard.deny')}
            </button>
            {isUnsandboxedRun ? (
              // Spec 117 D5: the three answers. "Turn off Safe mode" is where
              // full access lives — this card is the one place every user who
              // hits the sandbox wall passes through.
              <>
                <button
                  type="button"
                  onClick={() => handleApprove(false, 'disable_safe_mode')}
                  onTouchEnd={(e) => { e.preventDefault(); handleApprove(false, 'disable_safe_mode'); }}
                  disabled={submitting}
                  className="border border-border bg-background text-secondary rounded-sm text-micro px-2.5 py-1 font-medium hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                >
                  {t('approval.unsandboxed.disableSafeMode')}
                </button>
                {alwaysAllowExe && (
                  <button
                    type="button"
                    onClick={() => handleApprove(false, 'always')}
                    onTouchEnd={(e) => { e.preventDefault(); handleApprove(false, 'always'); }}
                    disabled={submitting}
                    className="border border-border bg-background text-secondary rounded-sm text-micro px-2.5 py-1 font-medium hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                  >
                    {t('approval.unsandboxed.alwaysAllow', { exe: alwaysAllowExe })}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => handleApprove(false, 'once')}
                  onTouchEnd={(e) => { e.preventDefault(); handleApprove(false, 'once'); }}
                  disabled={submitting}
                  className="bg-primary text-card rounded-sm text-caption font-medium px-3.5 py-1.5 hover:opacity-90 transition-opacity duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                >
                  {t('approval.unsandboxed.allowOnce')}
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => handleApprove(true)}
                  onTouchEnd={(e) => { e.preventDefault(); handleApprove(true); }}
                  disabled={submitting}
                  title={t('approval.autoApprove.title')}
                  className="border border-border bg-background text-secondary rounded-sm text-micro px-2.5 py-1 font-medium hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                >
                  {t('approval.autoApprove')}
                </button>
                <button
                  type="button"
                  onClick={() => handleApprove()}
                  onTouchEnd={(e) => { e.preventDefault(); handleApprove(); }}
                  disabled={submitting}
                  className="bg-primary text-card rounded-sm text-caption font-medium px-3.5 py-1.5 hover:opacity-90 transition-opacity duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
                >
                  {t('approval.approve')}
                </button>
              </>
            )}
          </div>
        )}

        {error && (
          <p className="text-caption text-error mt-1">{error}</p>
        )}
      </div>
    </div>
  );
}

type QuestionOption = { id: string; label: string };

/** The questions a worker's interaction carries (`tool_args.options`): a list
 *  of questions, each with its own prompt and options (ACP ask_question), or
 *  one flat list of options. Option buttons are offered only for a single
 *  question, since a chosen id answers the first question. */
function questionParts(toolArgs: Record<string, unknown>): { prompts: string[]; options: QuestionOption[] } {
  const raw = Array.isArray(toolArgs.options) ? (toolArgs.options as unknown[]) : [];
  const asRecord = (v: unknown) => (v && typeof v === 'object' ? (v as Record<string, unknown>) : null);
  const optionOf = (v: unknown): QuestionOption | null => {
    const o = asRecord(v);
    if (!o || typeof o.id !== 'string' || !o.id) return null;
    const label = [o.label, o.name, o.title].find((x) => typeof x === 'string' && x) as string | undefined;
    return { id: o.id, label: label ?? o.id };
  };
  const nested = raw.map(asRecord).filter((q): q is Record<string, unknown> => !!q && Array.isArray(q.options));
  if (nested.length > 0) {
    const prompts = nested
      .map((q) => [q.prompt, q.question, q.title].find((x) => typeof x === 'string' && x) as string | undefined)
      .filter((p): p is string => !!p);
    const options = nested.length === 1
      ? (nested[0].options as unknown[]).map(optionOf).filter((o): o is QuestionOption => !!o)
      : [];
    return { prompts, options };
  }
  return { prompts: [], options: raw.map(optionOf).filter((o): o is QuestionOption => !!o) };
}

function QuestionCard({
  approval,
  projectId,
  sessionId,
  resolution,
  onResolved,
  asker,
  askerName,
}: {
  approval: ApprovalCardProps['approval'];
  projectId: string;
  sessionId?: string;
  resolution: 'approved' | 'denied' | null | undefined;
  onResolved: (r: 'approved' | 'denied') => void;
  asker?: string;
  askerName?: string;
}) {
  const t = useT();
  const { approveToolCall, denyToolCall } = useAgent();
  const [answer, setAnswer] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prompt = typeof approval.tool_args?.prompt === 'string' ? approval.tool_args.prompt : '';
  const { prompts, options } = questionParts(approval.tool_args ?? {});
  const title = prompt || prompts[0] || '';

  if (resolution) {
    const answered = resolution === 'approved';
    return (
      <div className="mb-3 shrink-0 px-4 py-2 rounded-lg bg-sidebar text-body">
        <span className={answered ? 'text-success' : 'text-error'}>
          {answered ? t('approval.question.answered') : t('approval.denied')}:
        </span>{' '}
        <span className="text-secondary">{title}</span>
      </div>
    );
  }

  async function send(text?: string, optionId?: string) {
    setSubmitting(true);
    setError(null);
    try {
      await approveToolCall(projectId, approval.tool_call_id, text, undefined, sessionId, optionId);
      onResolved('approved');
    } catch {
      setSubmitting(false);
      setError(t('approval.submitError'));
    }
  }

  async function decline() {
    setSubmitting(true);
    setError(null);
    try {
      await denyToolCall(projectId, approval.tool_call_id, t('approval.deniedByUser'), sessionId);
      onResolved('denied');
    } catch {
      setSubmitting(false);
      setError(t('approval.submitError'));
    }
  }

  const typed = answer.trim();
  return (
    <div className="mb-3 shrink-0 rounded-lg border border-warning/50 border-l-[3px] border-l-warning bg-warning-surface overflow-hidden">
      <div className="px-4 py-1.5 border-b border-warning/20 flex items-center gap-2">
        <span data-testid="approval-asker" data-agent={asker ?? MAIN_AGENT_HANDLE} className="inline-flex">
          <AgentGlyph slug={asker ?? MAIN_AGENT_HANDLE} size={16} title={askerName ?? asker ?? 'Orbital'} />
        </span>
        <span className="type-label text-warning-text">{t('approval.question.header')}</span>
      </div>
      <div className="px-4 py-3 space-y-3">
        {prompt && (
          <p className="text-body font-medium text-primary whitespace-pre-wrap break-words">{prompt}</p>
        )}
        {prompts.map((p, i) => (
          <p key={i} className="text-body text-primary whitespace-pre-wrap break-words">{p}</p>
        ))}
        {options.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {options.map((o) => (
              <button
                key={o.id}
                type="button"
                onClick={() => send(undefined, o.id)}
                disabled={submitting}
                className="border border-border bg-background text-primary rounded-sm text-caption px-3 py-1.5 hover:bg-card-hover transition-colors duration-150 disabled:opacity-50 cursor-pointer min-h-[44px] md:min-h-0"
              >
                {o.label}
              </button>
            ))}
          </div>
        )}
        <input
          type="text"
          placeholder={t('approval.question.placeholder')}
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          onKeyDown={(e) => {
            // IME guard: Enter that commits a composition is not a send.
            if (e.key === 'Enter' && !e.nativeEvent.isComposing && typed && !submitting) {
              e.preventDefault();
              void send(typed);
            }
          }}
          className="w-full text-body px-3 py-1.5 rounded-lg border border-border bg-sidebar focus:outline-none focus:border-accent max-md:min-h-[44px]"
        />
        <div className="flex flex-col md:flex-row gap-2 md:justify-end pt-2 md:pt-0 pb-1 md:pb-0">
          <button
            type="button"
            onClick={decline}
            disabled={submitting}
            className="border border-error/40 text-error bg-background rounded-sm text-micro px-2.5 py-1 font-medium hover:bg-error/5 transition-colors duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
          >
            {t('credCard.deny')}
          </button>
          <button
            type="button"
            onClick={() => send(typed)}
            disabled={submitting || !typed}
            className="bg-primary text-card rounded-sm text-caption font-medium px-3.5 py-1.5 hover:opacity-90 transition-opacity duration-150 disabled:opacity-50 cursor-pointer w-full md:w-auto min-h-[44px]"
          >
            {t('chat.send')}
          </button>
        </div>
        {error && <p className="text-caption text-error mt-1">{error}</p>}
      </div>
    </div>
  );
}

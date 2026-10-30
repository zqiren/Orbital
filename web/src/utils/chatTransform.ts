// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { ChatMessage, ToolCall, ActivityCategory, FanoutTaskStatus, JobRef, SubAgentToolRow } from '../types';
import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

/**
 * Locale-aware translator for activity descriptions. Defaults to English so
 * callers (and unit tests) that omit it get the original English strings,
 * keeping output byte-identical when no locale is threaded in.
 */
export type ActivityTranslate = (
  key: StringKey,
  vars?: Record<string, string | number>,
) => string;
const EN_ACTIVITY: ActivityTranslate = (key, vars) => translate('en', key, vars);

export interface ResultTotals {
  chars: number;
  lines: number;
}

export interface ToolCallDispatch {
  agent: string;
  message: string;
}

export interface Activity {
  id: string;
  category: ActivityCategory;
  description: string;
  toolName: string;
  timestamp: string;
}

export type DisplayItem =
  | { type: 'user_message'; content: string; timestamp: string; target?: string; isHistorical?: boolean }
  | {
      type: 'agent_message';
      content: string;
      source: string;
      timestamp: string;
      isHistorical?: boolean;
      /**
       * When true, the renderer shows only the avatar + "agent · HH:MM" header
       * row with no body. Used to give a content-null (tool-only) assistant
       * turn a visible agent anchor above its capsule so the capsule does not
       * visually attach to the preceding user message. See FE-A3.
       */
      isHeaderOnly?: boolean;
    }
  | { type: 'sub_agent_message'; content: string; source: string; timestamp: string; isHistorical?: boolean }
  | { type: 'reasoning_block'; content: string; timestamp: string; turn_id: string; isHistorical?: boolean }
  | {
      type: 'tool_call_row';
      tool_name: string;
      target_description: string;
      tool_call_id: string;
      category: ActivityCategory;
      timestamp: string;
      result_content: string | null;
      result_status: 'pending' | 'received' | 'error';
      /**
       * Set only when result_content is a pre-cut preview (the live
       * tool_result event carries just the part the capsule shows): the
       * full result's size, for the truncation footer.
       */
      result_totals?: ResultTotals;
      /**
       * An `agent_message` send/respond: the agent it went to and the full
       * message Orbital sent, straight from the tool call's arguments.
       */
      dispatch?: ToolCallDispatch;
      isHistorical?: boolean;
    }
  | {
      type: 'agent_run';
      capsule_id: string;
      status: 'running' | 'completed' | 'error' | 'stopped';
      items: DisplayItem[];
      tool_call_count_by_name: Record<string, number>;
      has_thinking: boolean;
      started_at: number;
      ended_at: number | null;
      isHistorical?: boolean;
      /**
       * Spec 100: the sub-agent handle whose run this capsule shows. Unset
       * for the management agent's own capsules. Live events join a running
       * capsule only when this matches their source, so a worker's rows and
       * the manager's never land in each other's capsule.
       */
      source?: string;
      /**
       * When true, the renderer starts this capsule expanded so the reasoning
       * is visible without the user having to click the chevron. Set by the
       * transform when a content-null turn with reasoning opens the capsule.
       * See FE-A3 / FE-2.
       */
      defaultExpanded?: boolean;
      /**
       * Spec 112 A6: the capsule holds the tool calls of the daemon's
       * end-of-run memory record turn (rows tagged `_kind: "memory_record"`).
       * Rendered as a collapsed grey activity line labelled as a memory
       * update; it never carries a bubble or an agent header of its own.
       */
      memory_record?: boolean;
      /**
       * Spec 124 §3.7 follow-up: the capsule is the one row a memory tidy
       * pass leaves in the session that triggered it (`_kind: "memory_tidy"`).
       * Its children are the files the pass changed; the counts come from
       * the row's `_meta`. Collapsed, no bubble, no agent header.
       */
      memory_tidy?: { outcome: string; archived: number; restored: number; error?: string };
    }
  | { type: 'session_separator'; timestamp: string }
  /**
   * A thin daemon divider (keyless onboarding / session import). Rendered and
   * translated by ChatView from these fields, never from the row's English
   * `content`: `imported_transcript` opens an imported conversation (agent +
   * last-active date, `truncated` when older turns were cut);
   * `worker_onboarding` marks a worker running the project setup because
   * Orbital's own agent has no key.
   */
  | {
      /** Spec 146: an ask opened here. The tray in the composer acts on it; the
       *  chat keeps this line. */
      type: 'ask_line';
      text: string;
      timestamp: string;
    }
  | {
      type: 'daemon_row';
      event: 'imported_transcript' | 'worker_onboarding';
      /** Sub-agent slug ('codex', 'claude-code', …). */
      handle: string;
      timestamp: string;
      lastActivity?: string;
      truncated?: boolean;
    }
  | {
      /**
       * The agent hit its context ceiling and summarized everything older
       * than the last ~30% of the conversation. The daemon has always
       * persisted this marker (`_compaction` on a system row) and this
       * transform always dropped it, so the event was invisible despite
       * costing a real LLM call and discarding conversation the user can
       * still see above it.
       *
       * Deliberately NOT a `session_separator`, though it renders in the same
       * divider shape: that type drives the isHistorical pass, and a
       * compaction does not start a new session.
       *
       * Carries no content — the marker's body is the summary written FOR the
       * model, not for the reader.
       */
      type: 'compaction_marker';
      timestamp: string;
    }
  | {
      /**
       * A compact, one-line marker rendered in the chat flow for the daemon's
       * [Sub-agent] lifecycle system messages — start, message-sent, completion
       * (with summary), failure. Surfaces the persisted record that a
       * sub-agent ran; without this, the messages are silently dropped. See
       * FE-A2.
       */
      type: 'sub_agent_activity';
      action:
        | 'started'
        | 'sent'
        | 'completed'
        /**
         * Spec 129: a turn the worker ran on its own (a background command
         * it was waiting on finished) ended. Never a reply to the preceding
         * user row and never a dispatch boundary.
         */
        | 'continued'
        | 'failed'
        | 'error'
        | 'stopped'
        | 'interrupted'
        | 'background_lost'
        | 'interaction_required'
        | 'queue_dropped';
      handle: string;
      timestamp: string;
      /** Present for action='completed' | 'continued'. Trimmed; may be empty. */
      summary?: string;
      /** Present for action='sent'. The first chunk of the sent message. */
      preview?: string;
      /** Present for action='failed' | 'error'. */
      error?: string;
      /**
       * Number of background processes killed. Present for
       * action='background_lost', and for action='stopped' only when the
       * user's stop actually terminated tracked background work.
       */
      count?: number;
      /**
       * The `;`-joined command list behind `count`, with the producer's
       * "This background work did NOT complete." warning sentence stripped —
       * the row's own copy already carries that meaning, so repeating it
       * inside the command list would read twice.
       */
      detail?: string;
      /**
       * Present for action='queue_dropped'. Why the queued message never
       * dispatched, in the producer's own words (backlog #35a) — a stop, a
       * dead transport, an earlier send or dispatch that failed.
       */
      reason?: string;
      /** Present for action='interaction_required'. The interaction kind. */
      kind?: string;
      /**
       * Present for action='interaction_required'. The sub-agent's question,
       * with the agent-facing "Respond on the same in-flight request …"
       * instructions stripped (see SUB_AGENT_INTERACTION_RE).
       */
      prompt?: string;
      isHistorical?: boolean;
    }
  | {
      type: 'approval_card';
      what: string;
      tool_name: string;
      tool_call_id: string;
      tool_args: Record<string, unknown>;
      recent_activity: ChatMessage[];
      reasoning?: string;
      resolved?: 'approved' | 'denied';
    }
  | {
      type: 'agent_notify';
      title: string;
      body: string;
      urgency: 'high' | 'normal' | 'low';
      timestamp: string;
    }
  | {
      type: 'refresh_status';
      status: 'in_progress' | 'done' | 'failed' | 'skipped';
      trigger: 'turn_count' | 'agent_decided' | 'agent_decided_coalesced' | 'token_pressure';
      timestamp: string;
    }
  | {
      // Budget enforcement trip (P3-G). Built from a session row written by the
      // loop: {role:'system', source:'budget', event:'budget_blocked',
      // payload:{action,window,spend,limit,currency}}. Codes/numbers only — the
      // client localizes the message. (budget_threshold is WS-only; it never
      // lands in a session row, so it isn't represented here.)
      type: 'budget_event';
      event: 'budget_blocked';
      action: 'pause' | 'stop';
      window: 'daily' | 'weekly' | 'monthly' | 'total';
      spend: number;
      limit: number | null;
      currency: string;
      timestamp: string;
    }
  | {
      /**
       * Spec 009 §0.5: the management agent's `fanout` tool dispatches N
       * parallel worker sessions. Rendered as a standalone card (never a
       * regular tool_call_row inside an agent_run capsule) so the rows stay
       * clickable and independently status-able. See the fanout parsing
       * block in transformChatHistory for how fanout_id/handles are derived.
       */
      type: 'fanout_card';
      fanout_id: string;
      tasks: Array<{ handle: string; label: string; job?: JobRef }>;
      timestamp: string;
      isHistorical?: boolean;
      /**
       * Backfilled when a later join-summary system message (see
       * `parseFanoutSummary`) names this fanout_id: terminal per-task status
       * + the join's epoch ms. Without this, a fanout that already finished
       * before this history was (re)loaded would show every row defaulting
       * to "running" forever — there are no live WS events to correct it,
       * since those only fire for a fanout that's still in flight in THIS
       * tab. ChatView merges this with the live overlay at render time (live
       * wins per-key when present).
       */
      statuses?: Record<string, FanoutTaskStatus>;
      completedAtMs?: number;
    }
  | {
      /**
       * Spec 009 §0.5 item 4: the join-summary system message the daemon
       * writes when a fanout batch completes. Rendered as a plain system row
       * — no dedicated component in v1, the persisted text is shown verbatim
       * (mirrors how sub_agent_activity has no separate component file).
       * Also drives the `fanout_card.statuses`/`completedAtMs` backfill above.
       */
      type: 'fanout_summary';
      fanout_id: string;
      content: string;
      timestamp: string;
      isHistorical?: boolean;
    };

const WINDOWS_PATH_RE = /[A-Za-z]:\\(?:Users|Windows|Program)[^\s"';&|>]*/gi;
const UNIX_PATH_RE = /\/(?:home|Users|etc|var|root)\/[^\s"';&|>]*/g;
const ENV_VAR_RE = /(?:\$HOME|\$USERPROFILE|%USERPROFILE%|%APPDATA%|%LOCALAPPDATA%)/gi;

export function containsExternalPaths(command: string, workspace: string): boolean {
  const normalizedWs = workspace.replace(/\\/g, '/').toLowerCase();
  const allPaths: string[] = [];

  for (const re of [WINDOWS_PATH_RE, UNIX_PATH_RE]) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(command)) !== null) {
      allPaths.push(m[0]);
    }
  }

  ENV_VAR_RE.lastIndex = 0;
  if (ENV_VAR_RE.test(command)) {
    return true;
  }

  for (const p of allPaths) {
    const normalized = p.replace(/\\/g, '/').toLowerCase();
    if (!normalized.startsWith(normalizedWs)) {
      return true;
    }
  }

  return false;
}

const TOOL_NAME_TO_CATEGORY: Record<string, ActivityCategory> = {
  read: 'file_read',
  write: 'file_write',
  edit: 'file_edit',
  glob: 'file_search',
  grep: 'content_search',
  shell: 'command_exec',
  web_search: 'web_search',
  web_fetch: 'web_fetch',
  request_access: 'request_access',
  agent_message: 'agent_message',
  browser: 'browser_automation',
};

function toolCallToActivity(
  tc: ToolCall,
  timestamp: string,
  message?: ChatMessage,
  workspace?: string,
  tr: ActivityTranslate = EN_ACTIVITY,
): Activity {
  // NOTE: message._activity_descriptions (backend-authored English prose,
  // still written to JSONL for rollback safety) is deliberately IGNORED —
  // rendering from the structured tool call keeps this surface localized,
  // retroactively for old sessions too.
  void message;

  const name = tc.function.name;
  const category = TOOL_NAME_TO_CATEGORY[name] ?? 'tool_use';
  let args: Record<string, unknown> = {};
  try {
    args = JSON.parse(tc.function.arguments);
  } catch {
    // ignore
  }

  let description: string;
  switch (category) {
    case 'file_read':
      description = tr('activity.read', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_write':
      description = tr('activity.created', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_edit':
      description = tr('activity.edited', { path: String(args.path ?? args.file_path ?? name) });
      break;
    case 'file_search':
      description = tr('activity.searchingFiles', { pattern: String(args.pattern ?? '?') });
      break;
    case 'content_search':
      description = args.path
        ? tr('activity.searchingForIn', { pattern: String(args.pattern ?? '?'), path: String(args.path) })
        : tr('activity.searchingFor', { pattern: String(args.pattern ?? '?') });
      break;
    case 'command_exec':
      if (workspace && typeof args.command === 'string' && containsExternalPaths(args.command, workspace)) {
        description = tr('activity.ranRestricted');
      } else {
        description = tr('activity.ran', { command: String(args.command ?? name) });
      }
      break;
    case 'web_search':
      description = tr('activity.searched', { query: String(args.query ?? name) });
      break;
    case 'web_fetch':
      description = tr('activity.fetched', { url: String(args.url ?? name) });
      break;
    case 'request_access':
      description = tr('activity.requestedAccess', { path: String(args.path ?? name) });
      break;
    case 'agent_message':
      description = tr('activity.messaged', { handle: String(args.agent ?? args.handle ?? args.target ?? name) });
      break;
    case 'browser_automation': {
      const action = args.action as string | undefined;
      switch (action) {
        case 'navigate': description = tr('activity.browser.navigate', { url: String(args.url ?? 'page') }); break;
        case 'search': description = tr('activity.browser.search', { query: String(args.query ?? '?').slice(0, 50) }); break;
        case 'click': description = tr('activity.browser.click', { ref: String(args.ref ?? args.selector ?? '?') }); break;
        case 'screenshot': description = tr('activity.browser.screenshot'); break;
        case 'scroll': description = tr('activity.browser.scroll', { direction: String(args.direction ?? 'down') }); break;
        case 'snapshot': description = tr('activity.browser.snapshot'); break;
        case 'type': description = tr('activity.browser.type', { ref: String(args.ref ?? '?') }); break;
        case 'fill': description = tr('activity.browser.fill'); break;
        case 'search_page': description = tr('activity.browser.searchPage', { text: String(args.text ?? '?').slice(0, 30) }); break;
        case 'fetch': description = tr('activity.browser.fetch', { url: String(args.url ?? '?').slice(0, 60) }); break;
        case 'press': description = tr('activity.browser.press', { key: String(args.key ?? '?') }); break;
        case 'hover': description = tr('activity.browser.hover', { ref: String(args.ref ?? '?') }); break;
        case 'select': description = tr('activity.browser.select', { value: String(args.value ?? '?'), ref: String(args.ref ?? '?') }); break;
        case 'drag': description = tr('activity.browser.drag'); break;
        case 'upload_file': description = tr('activity.browser.upload'); break;
        case 'extract': description = tr('activity.browser.extract', { text: String(args.text ?? '?').slice(0, 50) }); break;
        case 'evaluate': description = tr('activity.browser.evaluate'); break;
        case 'go_back': description = tr('activity.browser.back'); break;
        case 'go_forward': description = tr('activity.browser.forward'); break;
        case 'reload': description = tr('activity.browser.reload'); break;
        case 'wait': description = tr('activity.browser.wait'); break;
        case 'pdf': description = tr('activity.browser.pdf'); break;
        case 'tab_new': description = tr('activity.browser.tabNew'); break;
        case 'tab_switch': description = tr('activity.browser.tabSwitch'); break;
        case 'tab_close': description = tr('activity.browser.tabClose'); break;
        case 'done': description = tr('activity.browser.done'); break;
        default: description = tr('activity.browser.unknown', { action: String(action ?? 'unknown') }); break;
      }
      break;
    }
    default:
      description = name === 'request_credential'
        ? tr('activity.requestedCredential', { domain: String(args.domain ?? 'website') })
        : tr('activity.usedTool', { name });
  }

  return {
    id: tc.id,
    category,
    description,
    toolName: name,
    timestamp,
  };
}

/** Localized description for a live agent.activity event. The event's own
 * `description` is backend-authored English; when the daemon ships the
 * parsed `arguments` alongside it, derive the description locally with the
 * same renderer the persisted path uses. Falls back to the wire description
 * for older daemons that don't send arguments. */
export function describeLiveActivity(
  toolName: string,
  args: Record<string, unknown> | undefined,
  workspace: string | undefined,
  tr: ActivityTranslate,
  fallback: string,
): string {
  if (!args) return fallback;
  const tc: ToolCall = {
    id: '',
    type: 'function',
    function: { name: toolName, arguments: JSON.stringify(args) },
  };
  return toolCallToActivity(tc, '', undefined, workspace, tr).description;
}

// Spec 100: a sub-agent's own tool names (claude-code, codex, ACP agents)
// mapped onto the Orbital tool whose description renders the same arguments.
const WORKER_TOOL_ALIASES: Record<string, string> = {
  bash: 'shell',
  commandexecution: 'shell',
  execute: 'shell',
  read: 'read',
  read_file: 'read',
  write: 'write',
  edit: 'edit',
  multiedit: 'edit',
  glob: 'glob',
  grep: 'grep',
  websearch: 'web_search',
  web_search: 'web_search',
  webfetch: 'web_fetch',
  web_fetch: 'web_fetch',
};

// For any other tool: the first of these arguments that holds text.
const WORKER_SUMMARY_ARGS = ['command', 'file_path', 'path', 'query', 'pattern', 'url', 'prompt', 'description', 'input'];

/** Spec 100: the row detail for a sub-agent's tool call, from its arguments.
 * Known tools render like the manager's own; codex's fileChange lists its
 * paths; any other tool shows its main argument, or "Used X" when it has
 * none. Shared by the live capsule and the reloaded one, so both agree. */
export function describeWorkerTool(
  toolName: string,
  args: Record<string, unknown> | undefined,
  workspace: string | undefined,
  tr: ActivityTranslate = EN_ACTIVITY,
): string {
  const key = toolName.toLowerCase();
  if (key === 'filechange' && Array.isArray(args?.changes)) {
    const paths = (args.changes as unknown[])
      .map((c) => (c && typeof c === 'object' ? (c as { path?: unknown }).path : undefined))
      .filter((p): p is string => typeof p === 'string' && p !== '');
    if (paths.length > 0) return tr('activity.edited', { path: paths.join(', ') });
  }
  const canonical = WORKER_TOOL_ALIASES[key];
  if (canonical && args) {
    return describeLiveActivity(canonical, args, workspace, tr, '');
  }
  if (args) {
    for (const k of WORKER_SUMMARY_ARGS) {
      const v = args[k];
      if (typeof v === 'string' && v.trim()) {
        const line = v.trim().split('\n')[0];
        return line.length > 120 ? `${line.slice(0, 120)}…` : line;
      }
    }
  }
  return tr('activity.usedTool', { name: toolName });
}

/** Spec 100: a sub-agent tool row's category (drives nothing but the row's
 * data), by the same alias map. */
export function workerToolCategory(toolName: string): ActivityCategory {
  const key = toolName.toLowerCase();
  return TOOL_NAME_TO_CATEGORY[WORKER_TOOL_ALIASES[key] ?? key] ?? 'tool_use';
}

/** The agent and full message of an `agent_message` send/respond, read from
 * the tool call's arguments (the persisted row's or the live event's).
 * Undefined for any other tool, and for actions that send no message. */
export function dispatchFromToolCall(
  toolName: string,
  args: Record<string, unknown> | undefined,
): ToolCallDispatch | undefined {
  if (toolName !== 'agent_message' || !args) return undefined;
  const message = args.message;
  if (typeof message !== 'string' || message === '') return undefined;
  return { agent: String(args.agent ?? args.handle ?? args.target ?? ''), message };
}

function parseToolArgs(raw: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

// Reasoning, tool_call_row, and empty-content agent_message markers
// may appear inside an agent_run.items array. Tool result content is
// carried on tool_call_row.result_content, paired by tool_call_id.
type CapsuleChild =
  | Extract<DisplayItem, { type: 'reasoning_block' }>
  | Extract<DisplayItem, { type: 'tool_call_row' }>
  | Extract<DisplayItem, { type: 'agent_message' }>;

interface OpenCapsule {
  items: CapsuleChild[];
  startedAtMs: number;
  endedAtMs: number;
  defaultExpanded: boolean;
  /** Spec 112: opened by a `_kind: "memory_record"` tool turn. */
  memoryRecord?: boolean;
}

// [Sub-agent] lifecycle markers persisted to JSONL by the lifecycle observer.
// Parsed and surfaced as sub_agent_activity items; without this they are
// silently dropped along with every other non-approval system message.
const SUB_AGENT_STARTED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+started\b/;
const SUB_AGENT_SENT_RE = /^\[Sub-agent\]\s+Message sent to\s+([\w.-]+):\s*(.*)$/;
const SUB_AGENT_COMPLETED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+completed\.\s*Summary:\s*([\s\S]*)$/;
// Spec 129: the worker continued on its own (a background command it was
// waiting on finished and the CLI ran a turn with no message from anyone).
const SUB_AGENT_CONTINUED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+continued\.\s*Summary:\s*([\s\S]*)$/;
const SUB_AGENT_FAILED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+failed:\s*([\s\S]*)$/;
// b17edb8 (v0.16.0): a pinned send whose spawn failed writes a durable
// "could not start" row; spec 116 gives it a rule (it rendered as nothing)
// and, when the row's _meta names a reason, a translated sentence.
const SUB_AGENT_COULD_NOT_START_RE =
  /^\[Sub-agent\]\s+([\w.-]+)\s+could not start:\s*([\s\S]*)$/;
// backlog #23 D2: on_error's marker ("stopped with error:") had no rule here
// at all, so the durable error row rendered as nothing — the contract
// drifted when on_error was added alongside on_failed's "failed:" shape.
const SUB_AGENT_ERROR_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+stopped with error:\s*([\s\S]*)$/;
// backlog #35: a queued message discarded before it ever dispatched. #26 wrote
// these through on_failed, so sessions recorded by builds before #35 still
// read "[Sub-agent] h failed: queued message dropped: …" and still match the
// failed rule above — correct for them, since that IS what that build wrote.
const SUB_AGENT_QUEUE_DROPPED_RE =
  /^\[Sub-agent\]\s+([\w.-]+)\s+queued message dropped:\s*([\s\S]*)$/;
// backlog #27: the same drift, four more times. on_user_stopped,
// on_turn_interrupted, on_background_work_lost and on_interaction_required
// each write a durable [Sub-agent] marker that no rule below matched, so all
// four rendered as NOTHING on history reload — including the one that says
// the sub-agent is BLOCKED waiting on the user.
//
// Each head pattern is deliberately loose (handle + the distinguishing
// phrase, nothing more) and the payload is extracted separately: if the
// producer's trailing prose is ever reworded, the row degrades to a
// less-detailed row instead of silently vanishing again — which is the whole
// failure mode these rules exist to end.
const SUB_AGENT_USER_STOPPED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+stopped by user\.\s*([\s\S]*)$/;
const SUB_AGENT_INTERRUPTED_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+was stopped before completing its current task\b/;
const SUB_AGENT_BACKGROUND_LOST_RE =
  /^\[Sub-agent\]\s+([\w.-]+)\s+teardown terminated\s+(\d+)\s+live background process\(es\):\s*([\s\S]*)$/;
const SUB_AGENT_INTERACTION_RE = /^\[Sub-agent\]\s+([\w.-]+)\s+requires input\s*\(([^)]*)\):\s*([\s\S]*)$/;
// on_user_stopped's optional tail, appended only when the kill actually
// terminated tracked background work.
const TERMINATED_BACKGROUND_RE = /Terminated\s+(\d+)\s+background process\(es\):\s*([\s\S]*)$/;
// Trailing warning shared by on_user_stopped's tail and on_background_work_lost.
const WORK_INCOMPLETE_RE = /\.?\s*This background work did NOT complete\.?\s*$/;
// on_interaction_required has NO _meta.display_content split (unlike
// on_completed), so the persisted content carries the agent-facing "how to
// answer" instructions inline. Cut them off the prompt — they address the
// LLM, not the reader. If this sentence is ever reworded the prompt keeps
// the extra prose (ugly, still visible); the paired parity tests catch it.
const INTERACTION_GUIDANCE_RE = /\s*Respond on the same in-flight request with[\s\S]*$/;

// Fanout tool call (spec 009 §0.5). The tool call's OWN arguments only carry
// the requested task labels — handles don't exist yet, the backend assigns
// them at dispatch time. The fanout_id instead comes from the tool RESULT's
// ack text, so a fanout call can't be turned into a fanout_card until BOTH
// halves have been seen (handled via `pendingFanoutCalls` in
// transformChatHistory). Once we have the fanout_id, per-task handles are
// synthesized as `worker:<fanout_id>-<index>` — this mirrors the wire format
// Task 2's `fanout.started` WS event uses for real, backend-assigned handles,
// so a card built from persisted history keys identically to one built live.
const FANOUT_ACK_RE = /Fanout ([a-z0-9]{8}) dispatched/;

function parseFanoutTaskLabels(argsJson: string): string[] {
  let args: unknown;
  try {
    args = JSON.parse(argsJson);
  } catch {
    return [];
  }
  const raw = (args as Record<string, unknown> | null)?.tasks;
  if (!Array.isArray(raw)) return [];
  return raw.map((t, i) => {
    if (typeof t === 'string') return t;
    if (t && typeof t === 'object' && typeof (t as Record<string, unknown>).label === 'string') {
      return (t as Record<string, unknown>).label as string;
    }
    return `Task ${i + 1}`;
  });
}

// Fanout join-summary system message (spec 009 §0.5 item 4). Format is
// FROZEN by the backend:
//   Line 1: `[Fanout <id8>] <k>/<N> succeeded.`
//   Per task, in order: `- [<status>] <label> (<handle>): <text> | transcript: <path>`
//   Optionally a trailing guidance paragraph — ignored (it won't match the
//   per-task line pattern, so it's naturally skipped rather than parsed).
const FANOUT_SUMMARY_HEADER_RE = /^\[Fanout ([a-z0-9]{8})\]\s+\d+\/\d+\s+succeeded\.$/;
const FANOUT_SUMMARY_TASK_LINE_RE = /^-\s*\[(completed|error|stalled|interrupted)\]\s+.+?\s+\(([^)]+)\):/;

function parseFanoutSummary(
  content: string,
): { fanoutId: string; statuses: Record<string, FanoutTaskStatus> } | null {
  const lines = content.split('\n');
  const header = lines[0]?.match(FANOUT_SUMMARY_HEADER_RE);
  if (!header) return null;
  const statuses: Record<string, FanoutTaskStatus> = {};
  for (const line of lines.slice(1)) {
    const m = line.match(FANOUT_SUMMARY_TASK_LINE_RE);
    if (!m) continue;
    statuses[m[2]] = m[1] as FanoutTaskStatus;
  }
  return { fanoutId: header[1], statuses };
}

// Spec 134 Part B: background job groups reuse the fanout card. A
// `shell(background=true)` result carries `_meta.background_job`; the
// daemon's job rows carry `_meta.job_group` ({group_id, jobs: [...]}) and an
// `event` from this set. `job_started` (a Codex/Pi launcher job, which has no
// tool row here) only builds the card; the others render their
// display_content as a system row and backfill the card's row statuses.
const JOB_ROW_EVENTS = new Set(['job_terminal', 'job_checkin', 'job_note', 'job_started']);
const FANOUT_STATUSES = new Set<FanoutTaskStatus>(['running', 'completed', 'error', 'stalled', 'interrupted']);

interface JobGroupMeta {
  group_id: string;
  jobs: Array<{
    job_id: string;
    command: string;
    status: FanoutTaskStatus;
    project_id?: string;
    agent?: string | null;
    ended_at_ms?: number | null;
  }>;
}

function parseBackgroundJob(
  meta: Record<string, unknown> | undefined,
): { groupId: string; task: { handle: string; label: string; job: JobRef } } | null {
  const raw = meta?.background_job as Record<string, unknown> | undefined;
  if (!raw || typeof raw.job_id !== 'string' || typeof raw.group_id !== 'string') return null;
  return {
    groupId: raw.group_id,
    task: {
      handle: raw.job_id,
      label: typeof raw.command === 'string' ? raw.command : raw.job_id,
      job: {
        project_id: typeof raw.project_id === 'string' ? raw.project_id : '',
        job_id: raw.job_id,
        agent: null,
      },
    },
  };
}

function parseJobGroup(meta: Record<string, unknown> | undefined): JobGroupMeta | null {
  const raw = meta?.job_group as Record<string, unknown> | undefined;
  if (!raw || typeof raw.group_id !== 'string' || !Array.isArray(raw.jobs)) return null;
  const jobs: JobGroupMeta['jobs'] = [];
  for (const j of raw.jobs as Array<Record<string, unknown>>) {
    if (!j || typeof j.job_id !== 'string') continue;
    const status = FANOUT_STATUSES.has(j.status as FanoutTaskStatus)
      ? (j.status as FanoutTaskStatus)
      : 'error';
    jobs.push({
      job_id: j.job_id,
      command: typeof j.command === 'string' ? j.command : j.job_id,
      status,
      project_id: typeof j.project_id === 'string' ? j.project_id : undefined,
      agent: typeof j.agent === 'string' ? j.agent : null,
      ended_at_ms: typeof j.ended_at_ms === 'number' ? j.ended_at_ms : null,
    });
  }
  return { group_id: raw.group_id, jobs };
}

type SubAgentActivity = Extract<DisplayItem, { type: 'sub_agent_activity' }>;

/**
 * Spec 116 D5: a `sub_agent_dispatch_failed` row whose `_meta.reason` is
 * `update_needed` renders the translated update sentence instead of the
 * backend's English error (backend text is never translated; a reason enum
 * is data the UI may map). Any other reason, or no meta, keeps the content.
 */
function applyDispatchFailureReason(
  activity: SubAgentActivity,
  meta: Record<string, unknown> | undefined,
  tr: ActivityTranslate,
): SubAgentActivity {
  if (!meta || meta.event !== 'sub_agent_dispatch_failed') return activity;
  if (meta.reason !== 'update_needed') return activity;
  const name = typeof meta.name === 'string' ? meta.name : activity.handle;
  return {
    ...activity,
    error: tr('chat.subAgentNeedsUpdate', {
      name,
      version: typeof meta.version === 'string' ? meta.version : '?',
      min: typeof meta.min_version === 'string' ? meta.min_version : '?',
    }),
  };
}

/** The keyless-onboarding / import dividers, keyed on `_meta.event`. */
function parseDaemonRow(msg: ChatMessage): DisplayItem | null {
  const meta = msg._meta;
  if (!meta) return null;
  if (meta.event === 'imported_transcript') {
    return {
      type: 'daemon_row',
      event: 'imported_transcript',
      handle: typeof meta.agent === 'string' ? meta.agent : '',
      lastActivity: typeof meta.last_activity === 'string' ? meta.last_activity : undefined,
      truncated: meta.truncated === true,
      timestamp: msg.timestamp,
    };
  }
  if (meta.event === 'worker_onboarding') {
    return {
      type: 'daemon_row',
      event: 'worker_onboarding',
      handle: typeof meta.handle === 'string' ? meta.handle : '',
      timestamp: msg.timestamp,
    };
  }
  return null;
}

/**
 * True for a dispatch marker (`[Sub-agent] Message sent to <slug>: …` with a
 * `dispatch_id`) whose previous row is the `worker_onboarding` row for the
 * same worker: the marker carrying the onboarding instruction.
 */
function isOnboardingInstructionMarker(msg: ChatMessage, prev: ChatMessage | undefined): boolean {
  const prevMeta = prev?.role === 'system' ? prev._meta : undefined;
  if (!prevMeta || prevMeta.event !== 'worker_onboarding') return false;
  const meta = msg._meta;
  if (!meta || !meta.dispatch_id) return false;
  const content = typeof meta.display_content === 'string' ? meta.display_content : (msg.content ?? '');
  const sent = content.match(/^\[Sub-agent\]\s+Message sent to\s+([\w.-]+):/);
  if (!sent) return false;
  const handle = typeof meta.handle === 'string' && meta.handle ? meta.handle : sent[1];
  return handle === prevMeta.handle;
}

/**
 * The worker bubble for a `sub_agent_terminal` completed row carrying
 * `_meta.reply`, or null when the row has no reply or the worker's bubble for
 * this turn is already drawn (searching back to the user's last message).
 *
 * A `continued` row (spec 129) is the worker's own turn, started by a
 * background command finishing, not by anyone's message: its reply is always
 * its own bubble. It is never paired with the preceding user row, and no
 * transcript join can draw it (the turn has no dispatch marker), so the
 * "already drawn since the last user message" check does not apply.
 */
function terminalReplyBubble(msg: ChatMessage, items: DisplayItem[]): DisplayItem | null {
  const meta = msg._meta;
  if (!meta || meta.event !== 'sub_agent_terminal') return null;
  if (meta.kind !== 'completed' && meta.kind !== 'continued') return null;
  const reply = typeof meta.reply === 'string' ? meta.reply : '';
  const handle = typeof meta.handle === 'string' ? meta.handle : '';
  if (!reply.trim() || !handle) return null;
  if (meta.kind === 'continued') {
    if (reply.trim() === '(no output)') return null;
    return { type: 'agent_message', content: reply, source: handle, timestamp: msg.timestamp };
  }
  for (let k = items.length - 1; k >= 0; k--) {
    const it = items[k];
    if (it.type === 'user_message') break;
    if ((it.type === 'agent_message' || it.type === 'agent_run') && it.source === handle) {
      return null;
    }
  }
  return { type: 'agent_message', content: reply, source: handle, timestamp: msg.timestamp };
}

function parseSubAgentSystemMessage(
  content: string,
  timestamp: string,
): SubAgentActivity | null {
  let m: RegExpMatchArray | null;
  if ((m = content.match(SUB_AGENT_STARTED_RE))) {
    return { type: 'sub_agent_activity', action: 'started', handle: m[1], timestamp };
  }
  if ((m = content.match(SUB_AGENT_SENT_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'sent',
      handle: m[1],
      preview: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_COMPLETED_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'completed',
      handle: m[1],
      summary: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_CONTINUED_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'continued',
      handle: m[1],
      summary: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_FAILED_RE)) || (m = content.match(SUB_AGENT_COULD_NOT_START_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'failed',
      handle: m[1],
      error: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_ERROR_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'error',
      handle: m[1],
      error: m[2].trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_QUEUE_DROPPED_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'queue_dropped',
      handle: m[1],
      reason: m[2].replace(/\.\s*$/, '').trim(),
      timestamp,
    };
  }
  // backlog #27, in producer order. 'stopped' is a neutral system event (the
  // user asked for it) — only the background work it destroyed is bad news,
  // which is what count/detail carry. 'interrupted' and 'background_lost'
  // are warnings: work was expected and none arrived.
  if ((m = content.match(SUB_AGENT_USER_STOPPED_RE))) {
    const terminated = m[2].match(TERMINATED_BACKGROUND_RE);
    return {
      type: 'sub_agent_activity',
      action: 'stopped',
      handle: m[1],
      timestamp,
      ...(terminated
        ? {
            count: Number(terminated[1]),
            detail: terminated[2].replace(WORK_INCOMPLETE_RE, '').trim(),
          }
        : {}),
    };
  }
  if ((m = content.match(SUB_AGENT_INTERRUPTED_RE))) {
    return { type: 'sub_agent_activity', action: 'interrupted', handle: m[1], timestamp };
  }
  if ((m = content.match(SUB_AGENT_BACKGROUND_LOST_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'background_lost',
      handle: m[1],
      count: Number(m[2]),
      detail: m[3].replace(WORK_INCOMPLETE_RE, '').trim(),
      timestamp,
    };
  }
  if ((m = content.match(SUB_AGENT_INTERACTION_RE))) {
    return {
      type: 'sub_agent_activity',
      action: 'interaction_required',
      handle: m[1],
      kind: m[2].trim(),
      prompt: m[3].replace(INTERACTION_GUIDANCE_RE, '').trim(),
      timestamp,
    };
  }
  return null;
}

type BudgetEventItem = Extract<DisplayItem, { type: 'budget_event' }>;

/**
 * Parse a budget-trip session row into a timeline item, or null if the message
 * isn't a budget event. The row shape (written by the agent loop) is
 * `{role:'system', source:'budget', event:'budget_blocked', payload:{...}}`.
 * `event`/`payload` aren't on the ChatMessage type (they survive as raw JSONL
 * passthrough), so we read them off a narrow record cast. Codes/numbers only —
 * never trusts any daemon-side display string.
 */
function parseBudgetSystemMessage(msg: ChatMessage): BudgetEventItem | null {
  if (msg.source !== 'budget') return null;
  const raw = msg as unknown as {
    event?: unknown;
    payload?: Record<string, unknown>;
  };
  if (raw.event !== 'budget_blocked') return null;
  const p = raw.payload ?? {};
  const action = p.action === 'stop' ? 'stop' : 'pause';
  const window =
    p.window === 'weekly' || p.window === 'monthly' || p.window === 'total'
      ? p.window
      : 'daily';
  const spend = typeof p.spend === 'number' ? p.spend : 0;
  const limit = typeof p.limit === 'number' ? p.limit : null;
  const currency = typeof p.currency === 'string' ? p.currency : 'USD';
  return {
    type: 'budget_event',
    event: 'budget_blocked',
    action,
    window,
    spend,
    limit,
    currency,
    timestamp: msg.timestamp ?? '',
  };
}

export function transformChatHistory(
  messages: ChatMessage[],
  workspace?: string,
  tr: ActivityTranslate = EN_ACTIVITY,
): DisplayItem[] {
  const items: DisplayItem[] = [];
  let i = 0;
  let currentSessionId: string | undefined;
  let capsuleCounter = 0;
  let currentCapsule: OpenCapsule | null = null;
  // Fanout calls awaiting their tool result (which carries the fanout_id —
  // see the FANOUT_ACK_RE comment above). Keyed by tool_call_id.
  const pendingFanoutCalls = new Map<string, { labels: string[]; timestamp: string }>();
  // fanout_card(s) whose ack arrived while ANOTHER tool call in the SAME
  // capsule still had a 'pending' result (e.g. tool_calls: [fanout, read] with
  // the fanout ack processed before read's result). Finalizing the capsule
  // right then would flush `read` still pending and permanently orphan its
  // result once it does arrive (the pairing loop only searches the OPEN
  // capsule). So the card waits here instead, and is drained by
  // finalizeCapsule() the moment the capsule actually finalizes — whether
  // that's because every row settled (see the `role === 'tool'` pairing
  // branch below) or an unrelated later boundary (next assistant/user
  // message, session boundary, end of stream) finalizes it first.
  let deferredFanoutCards: Extract<DisplayItem, { type: 'fanout_card' }>[] = [];

  // Spec 134 Part B: the card of background job group `groupId`, wherever it
  // is (still deferred behind its capsule, or already emitted).
  function findJobCard(groupId: string): Extract<DisplayItem, { type: 'fanout_card' }> | undefined {
    const deferred = deferredFanoutCards.find((c) => c.fanout_id === groupId);
    if (deferred) return deferred;
    for (let k = items.length - 1; k >= 0; k--) {
      const it = items[k];
      if (it.type === 'fanout_card' && it.fanout_id === groupId) return it;
    }
    return undefined;
  }

  function addJobTask(groupId: string, task: { handle: string; label: string; job: JobRef },
                      timestamp: string, deferred: boolean): void {
    const card = findJobCard(groupId);
    if (card) {
      if (!card.tasks.some((t) => t.handle === task.handle)) card.tasks.push(task);
      return;
    }
    const fresh: Extract<DisplayItem, { type: 'fanout_card' }> = {
      type: 'fanout_card', fanout_id: groupId, tasks: [task], timestamp,
    };
    if (deferred) deferredFanoutCards.push(fresh);
    else items.push(fresh);
  }

  function tsToMs(ts: string): number {
    const n = Date.parse(ts);
    return Number.isFinite(n) ? n : 0;
  }

  function openCapsuleAt(ts: string, defaultExpanded = false): OpenCapsule {
    const ms = tsToMs(ts);
    return { items: [], startedAtMs: ms, endedAtMs: ms, defaultExpanded };
  }

  function finalizeCapsule(status: 'running' | 'completed' | 'error' | 'stopped' = 'completed'): void {
    if (currentCapsule && currentCapsule.items.length > 0) {
      const counts: Record<string, number> = {};
      let hasThinking = false;
      for (const it of currentCapsule.items) {
        if (it.type === 'tool_call_row') {
          counts[it.tool_name] = (counts[it.tool_name] ?? 0) + 1;
        } else if (it.type === 'reasoning_block') {
          hasThinking = true;
        }
      }
      items.push({
        type: 'agent_run',
        capsule_id: `cap:${currentCapsule.startedAtMs}:${capsuleCounter++}`,
        status,
        items: currentCapsule.items,
        tool_call_count_by_name: counts,
        has_thinking: hasThinking,
        started_at: currentCapsule.startedAtMs,
        ended_at: status === 'running' ? null : currentCapsule.endedAtMs,
        ...(currentCapsule.defaultExpanded ? { defaultExpanded: true } : {}),
        ...(currentCapsule.memoryRecord ? { memory_record: true } : {}),
      });
    }
    currentCapsule = null;
    // Every finalizeCapsule() call site is a safe point to drain: whatever
    // capsule content existed just got flushed above (or there was none), so
    // a deferred card can never land ahead of the capsule it was waiting on.
    if (deferredFanoutCards.length > 0) {
      for (const card of deferredFanoutCards) items.push(card);
      deferredFanoutCards = [];
    }
  }

  while (i < messages.length) {
    const msg = messages[i];

    if (msg._compaction) {
      // Surface the boundary, drop the body: the content is the summary the
      // model reads, not something to paste into the transcript.
      finalizeCapsule();
      items.push({ type: 'compaction_marker', timestamp: msg.timestamp });
      i++;
      continue;
    }

    // Spec 124 §3.7 follow-up: the tidy pass's one row. The Memory session
    // itself is hidden; this daemon system row is the only trace a pass
    // leaves, rendered as ONE collapsed capsule whose children are the
    // files the pass changed. It closes any open capsule first so a tidy
    // that landed mid-run never merges into the agent's own step.
    if (msg._kind === 'memory_tidy') {
      finalizeCapsule();
      const meta = (msg._meta ?? {}) as Record<string, unknown>;
      const files = Array.isArray(meta.files)
        ? (meta.files as unknown[]).filter((f): f is string => typeof f === 'string')
        : [];
      const outcome = typeof meta.outcome === 'string' ? meta.outcome : '';
      const error = typeof meta.error === 'string' && meta.error ? meta.error : undefined;
      const ms = tsToMs(msg.timestamp);
      const children: CapsuleChild[] = files.map((path, f) => ({
        type: 'tool_call_row',
        tool_name: 'edit',
        target_description: path,
        tool_call_id: `tidy:${msg.timestamp}:${f}`,
        category: 'file_edit',
        timestamp: msg.timestamp,
        result_content: '',
        result_status: 'received',
      }));
      items.push({
        type: 'agent_run',
        capsule_id: `tidy:${ms}:${capsuleCounter++}`,
        status: outcome === 'failed' ? 'error' : 'completed',
        items: children,
        tool_call_count_by_name: children.length > 0 ? { edit: children.length } : {},
        has_thinking: false,
        started_at: ms,
        ended_at: ms,
        memory_tidy: {
          outcome,
          archived: typeof meta.archived === 'number' ? meta.archived : 0,
          restored: typeof meta.restored === 'number' ? meta.restored : 0,
          ...(error ? { error } : {}),
        },
      });
      i++;
      continue;
    }

    // Spec 112 A6: the end-of-run memory record turn. The reply the user
    // already saw is the reply; what follows it is bookkeeping. The [MEMORY]
    // ask (system), a pinned worker's reminder (user) and its lifecycle rows
    // (system) render as nothing. The model's tool calls render as ONE
    // collapsed capsule flagged `memory_record` — no bubble, no agent
    // header — and its text ("nothing to record", "Done.") is dropped, so a
    // record turn that edited nothing leaves no trace in the chat.
    if (msg._kind === 'memory_record') {
      if (msg.role === 'system' || msg.role === 'user') {
        i++;
        continue;
      }
      if (msg.source === 'sub_agent') {
        // A pinned worker's record turn (the /chat endpoint's synthetic run
        // summary): only its tool rows, as a memory capsule under the worker.
        finalizeCapsule();
        const handle = msg.sub_agent_handle ?? 'sub-agent';
        const streamTools = (msg.sub_agent_stream ?? []).filter(
          (r): r is SubAgentToolRow & { kind: 'tool' } => r.kind === 'tool',
        );
        const toolRows = streamTools.length > 0 ? streamTools : (msg.sub_agent_tool_rows ?? []);
        if (toolRows.length > 0) {
          const counts: Record<string, number> = {};
          const children: CapsuleChild[] = toolRows.map((row, r) => {
            const name = row.name || 'tool';
            counts[name] = (counts[name] ?? 0) + 1;
            const hasResult = row.result_preview !== undefined;
            return {
              type: 'tool_call_row',
              tool_name: name,
              target_description: row.arguments
                ? describeWorkerTool(name, row.arguments, workspace, tr)
                : `${(row.duration_seconds ?? 0).toFixed(1)}s`,
              tool_call_id: row.tool_call_id || `sub:${handle}:${msg.timestamp}:${r}`,
              category: workerToolCategory(name),
              timestamp: row.timestamp || msg.timestamp,
              result_content: hasResult ? (row.result_preview ?? '') : null,
              result_status: 'received',
            };
          });
          const startedAtMs = tsToMs(msg.timestamp);
          const durationMs = Math.round((msg.sub_agent_duration ?? 0) * 1000);
          items.push({
            type: 'agent_run',
            capsule_id: `sub_agent:${handle}:${msg.timestamp}:${capsuleCounter++}`,
            status: 'completed',
            items: children,
            tool_call_count_by_name: counts,
            has_thinking: false,
            started_at: startedAtMs,
            ended_at: startedAtMs + durationMs,
            source: handle,
            memory_record: true,
          });
        }
        i++;
        continue;
      }
      if (msg.role === 'assistant') {
        const hasTools = !!(msg.tool_calls && msg.tool_calls.length > 0);
        if (!hasTools) {
          i++;
          continue;
        }
        if (!currentCapsule || !currentCapsule.memoryRecord) {
          finalizeCapsule();
          currentCapsule = openCapsuleAt(msg.timestamp, false);
          currentCapsule.memoryRecord = true;
        }
        const msTime = tsToMs(msg.timestamp);
        for (const tc of msg.tool_calls!) {
          const activity = toolCallToActivity(tc, msg.timestamp, msg, workspace, tr);
          currentCapsule.items.push({
            type: 'tool_call_row',
            tool_name: activity.toolName,
            target_description: activity.description,
            tool_call_id: tc.id,
            category: activity.category,
            timestamp: msg.timestamp,
            result_content: null,
            result_status: 'pending',
          });
          currentCapsule.endedAtMs = msTime;
        }
        i++;
        continue;
      }
      // role === 'tool': pairs into the open memory capsule below.
    }

    // Synthetic sub-agent run injected by the /chat endpoint after a dispatch
    // marker (source === "sub_agent"). A sub-agent is a first-class agent, so
    // it renders with the SAME display items as the management agent: an
    // agent header, a collapsible `agent_run` tool capsule, and a response
    // bubble. The handle rides on `source` (drives both the name and the icon
    // in ChatMessage). Spec 100: rows carry the call's arguments and result
    // when the transcript has them (older transcripts: name + duration only),
    // the turn's thinking sits between them, and a dispatch still in flight
    // renders its capsule so far as running — the same capsule the live
    // events are extending. Spec 104: when the row carries the turn's ordered
    // stream, the worker's intermediate messages render as bubbles between
    // its tool capsules — what the live view showed — instead of one capsule
    // of every row followed by the final text only.
    if (msg.source === 'sub_agent') {
      finalizeCapsule();
      const handle = msg.sub_agent_handle ?? 'sub-agent';
      const startedAtMs = tsToMs(msg.timestamp);
      const toolRows = msg.sub_agent_tool_rows ?? [];
      const thinking = (msg.sub_agent_thinking ?? []).filter((b) => b.content.trim());
      const inFlight = !!msg.sub_agent_in_flight;
      const stream = msg.sub_agent_stream ?? [];

      // Shared row builders (spec 104 §3.5): the stream path and the legacy
      // fallback build the same tool_call_row / reasoning_block for a given
      // row, so a turn looks the same whichever path rendered it. `r` is the
      // row's index among the turn's tool rows (its synthetic id).
      const toolCallRow = (row: SubAgentToolRow, r: number): CapsuleChild => {
        const name = row.name || 'tool';
        const hasResult = row.result_preview !== undefined;
        return {
          type: 'tool_call_row',
          tool_name: name,
          // With arguments: the same detail the live row showed. Without
          // (older transcripts): the per-tool duration ("Write · 1.2s").
          target_description: row.arguments
            ? describeWorkerTool(name, row.arguments, workspace, tr)
            : `${(row.duration_seconds ?? 0).toFixed(1)}s`,
          tool_call_id: row.tool_call_id || `sub:${handle}:${msg.timestamp}:${r}`,
          category: workerToolCategory(name),
          timestamp: row.timestamp || msg.timestamp,
          result_content: hasResult ? (row.result_preview ?? '') : null,
          // A call of a finished turn with no result recorded still
          // expands ("no result"); in flight it is still pending.
          result_status: hasResult || !inFlight ? 'received' : 'pending',
          ...(row.result_total_chars !== undefined && row.result_total_lines !== undefined
            ? { result_totals: { chars: row.result_total_chars, lines: row.result_total_lines } }
            : {}),
        };
      };
      const reasoningBlock = (content: string, afterTool: number): CapsuleChild => ({
        type: 'reasoning_block',
        content,
        timestamp: msg.timestamp,
        turn_id: `sub:${handle}:${msg.timestamp}:${afterTool}`,
      });
      const headerAnchor = (): DisplayItem => ({
        type: 'agent_message',
        content: '',
        source: handle,
        timestamp: msg.timestamp,
        isHeaderOnly: true,
      });

      if (stream.length > 0) {
        // Spec 104 stream path. Contiguous tool / thinking rows buffer into
        // one capsule; a message row flushes the buffer and becomes a bubble
        // (D1: an `agent_message` under the handle, like the final bubble).
        // Rows after the last message are a trailing capsule (D2), running
        // when the dispatch is still in flight. The stream's last message IS
        // msg.content (last wins), so no separate final bubble is pushed.
        let buf: CapsuleChild[] = [];
        let counts: Record<string, number> = {};
        let bufHasThinking = false;
        let bufStartTs = '';
        let bufStartMs = 0;
        let bufEndMs = 0;
        let toolIdx = 0;
        const note = (ts: string, endMs: number) => {
          if (buf.length === 0) {
            bufStartTs = ts;
            bufStartMs = tsToMs(ts);
            bufEndMs = bufStartMs;
          }
          bufEndMs = Math.max(bufEndMs, endMs);
        };
        const flushTools = (running: boolean) => {
          if (buf.length === 0) return;
          // D1 anchor rule: the capsule needs the worker's header unless the
          // item before it is already this worker's bubble or capsule.
          const last = items[items.length - 1];
          const anchored =
            !!last &&
            (last.type === 'agent_message' ||
              last.type === 'sub_agent_message' ||
              last.type === 'agent_run') &&
            last.source === handle;
          if (!anchored) items.push(headerAnchor());
          items.push({
            type: 'agent_run',
            capsule_id: `sub_agent:${handle}:${bufStartTs}:${capsuleCounter++}`,
            status: running ? 'running' : 'completed',
            items: buf,
            tool_call_count_by_name: counts,
            has_thinking: bufHasThinking,
            started_at: bufStartMs,
            ended_at: running ? null : bufEndMs,
            defaultExpanded: false,
            source: handle,
          });
          buf = [];
          counts = {};
          bufHasThinking = false;
        };
        for (const row of stream) {
          if (row.kind === 'tool') {
            const ts = row.timestamp || msg.timestamp;
            note(ts, tsToMs(ts) + Math.round((row.duration_seconds ?? 0) * 1000));
            const name = row.name || 'tool';
            counts[name] = (counts[name] ?? 0) + 1;
            buf.push(toolCallRow(row, toolIdx++));
          } else if (row.kind === 'thinking') {
            if (!row.content.trim()) continue;
            const ts = row.timestamp || msg.timestamp;
            note(ts, tsToMs(ts));
            bufHasThinking = true;
            buf.push(reasoningBlock(row.content, toolIdx));
          } else if (row.kind === 'message') {
            flushTools(false);
            const cleaned = (row.content ?? '').replace(/\x1b\[[0-9;]*m/g, '').trim();
            if (!cleaned || cleaned === '(no response)') continue;
            items.push({
              type: 'agent_message',
              content: cleaned,
              source: handle,
              timestamp: row.timestamp || msg.timestamp,
            });
          }
          // Any other kind (a newer daemon): ignored.
        }
        flushTools(inFlight);
        i++;
        continue;
      }

      // Legacy / version-skew fallback (no stream: an older daemon): header +
      // ONE capsule of every row + the final bubble.
      // 1 & 2. Header + tool capsule, emitted together: the header only
      // exists to anchor the capsule that follows, so no capsule means no
      // header either (D1 — an empty header-only anchor with the response
      // bubble's own identical header right after it was rendering as two
      // adjacent headers for the same turn). The response bubble (step 3)
      // always carries its own header regardless.
      if (toolRows.length > 0 || thinking.length > 0) {
        items.push(headerAnchor());

        const counts: Record<string, number> = {};
        const capsuleItems: CapsuleChild[] = [];
        const pushThinking = (afterTool: number) => {
          for (const block of thinking) {
            if (block.after_tool !== afterTool) continue;
            capsuleItems.push(reasoningBlock(block.content, afterTool));
          }
        };
        for (let r = 0; r < toolRows.length; r++) {
          pushThinking(r);
          const row = toolRows[r];
          const name = row.name || 'tool';
          counts[name] = (counts[name] ?? 0) + 1;
          capsuleItems.push(toolCallRow(row, r));
        }
        pushThinking(toolRows.length);
        // A block placed past the last row (a mismatched count) still shows.
        for (const block of thinking) {
          if (block.after_tool > toolRows.length) {
            capsuleItems.push(reasoningBlock(block.content, block.after_tool));
          }
        }
        const durationMs = Math.round((msg.sub_agent_duration ?? 0) * 1000);
        items.push({
          type: 'agent_run',
          capsule_id: `sub_agent:${handle}:${msg.timestamp}:${capsuleCounter++}`,
          status: inFlight ? 'running' : 'completed',
          items: capsuleItems,
          tool_call_count_by_name: counts,
          has_thinking: thinking.length > 0,
          started_at: startedAtMs,
          ended_at: inFlight ? null : startedAtMs + durationMs,
          defaultExpanded: false,
          source: handle,
        });
      }

      // 3. Response bubble (same as a management agent_message), if non-empty.
      const respText = (msg.content ?? '').trim();
      if (respText) {
        items.push({
          type: 'agent_message',
          content: msg.content ?? '',
          source: handle,
          timestamp: msg.timestamp,
        });
      }

      i++;
      continue;
    }

    if (msg.role === 'system') {
      // Budget trip row (P3-G): {source:'budget', event:'budget_blocked',
      // payload:{action,window,spend,limit,currency}}. The session endpoint
      // passes the raw JSONL through, so `event`/`payload` ride as extra fields
      // not declared on ChatMessage — read them via a narrow cast.
      const budgetItem = parseBudgetSystemMessage(msg);
      if (budgetItem) {
        finalizeCapsule();
        items.push(budgetItem);
        i++;
        continue;
      }
      if (msg._meta?.approval_request) {
        finalizeCapsule();
        items.push({
          type: 'approval_card',
          what: msg.content ?? '',
          tool_name: (msg._meta.tool_name as string) ?? '',
          tool_call_id: (msg._meta.tool_call_id as string) ?? '',
          tool_args: (msg._meta.tool_args as Record<string, unknown>) ?? {},
          recent_activity: [],
          reasoning: msg._meta.reasoning as string | undefined,
          resolved: msg._meta.resolution as 'approved' | 'denied' | undefined,
        });
        i++;
        continue;
      }
      // Spec 134 Part B: background job rows (join, check-in, notes) and a
      // worker's job start. See JOB_ROW_EVENTS.
      const jobEvent = msg._meta?.event;
      if (typeof jobEvent === 'string' && JOB_ROW_EVENTS.has(jobEvent)) {
        finalizeCapsule();
        const group = parseJobGroup(msg._meta);
        if (group && jobEvent === 'job_started') {
          for (const j of group.jobs) {
            addJobTask(group.group_id, {
              handle: j.job_id, label: j.command,
              job: { project_id: j.project_id ?? '', job_id: j.job_id, agent: j.agent ?? null },
            }, msg.timestamp, false);
          }
          i++;
          continue;
        }
        const card = group ? findJobCard(group.group_id) : undefined;
        if (group && card) {
          const statuses: Record<string, FanoutTaskStatus> = { ...(card.statuses ?? {}) };
          for (const j of group.jobs) {
            statuses[j.job_id] = j.status;
            const task = card.tasks.find((t) => t.handle === j.job_id);
            if (task?.job && j.ended_at_ms) task.job = { ...task.job, ended_at_ms: j.ended_at_ms };
          }
          card.statuses = statuses;
          const allEnded = card.tasks.every((t) => statuses[t.handle] && statuses[t.handle] !== 'running');
          if (allEnded) card.completedAtMs = tsToMs(msg.timestamp);
        }
        const displayContent =
          typeof msg._meta?.display_content === 'string'
            ? (msg._meta.display_content as string)
            : (msg.content ?? '');
        items.push({
          type: 'fanout_summary',
          fanout_id: group?.group_id ?? '',
          content: displayContent,
          timestamp: msg.timestamp,
        });
        i++;
        continue;
      }
      // Fanout join-summary (spec 009 §0.5 item 4): renders as a plain system
      // row AND backfills the matching fanout_card's terminal statuses so a
      // reload of a COMPLETED fanout doesn't show every row stuck "running"
      // forever (see the `statuses`/`completedAtMs` doc comment on
      // fanout_card). Checked before the [Sub-agent] markers below since the
      // two prefixes are disjoint (`[Fanout ` vs `[Sub-agent] `).
      const fanoutSummary = parseFanoutSummary(msg.content ?? '');
      if (fanoutSummary) {
        finalizeCapsule();
        for (let k = items.length - 1; k >= 0; k--) {
          const it = items[k];
          if (it.type === 'fanout_card' && it.fanout_id === fanoutSummary.fanoutId) {
            items[k] = {
              ...it,
              statuses: fanoutSummary.statuses,
              completedAtMs: tsToMs(msg.timestamp),
            };
            break;
          }
        }
        // Display split (backend _meta contract, spec 009 join summary):
        // _meta.display_content is the user-facing header + per-task lines;
        // the full content keeps the agent-facing guidance paragraph for the
        // LLM. Legacy messages (pre-split) have no _meta and render in full.
        const displayContent =
          typeof msg._meta?.display_content === 'string'
            ? (msg._meta.display_content as string)
            : (msg.content ?? '');
        items.push({
          type: 'fanout_summary',
          fanout_id: fanoutSummary.fanoutId,
          content: displayContent,
          timestamp: msg.timestamp,
        });
        i++;
        continue;
      }
      if (msg._kind === 'ask_card') {
        finalizeCapsule();
        items.push({ type: 'ask_line', text: typeof msg.content === 'string' ? msg.content : '', timestamp: msg.timestamp });
        i++;
        continue;
      }
      // Keyless onboarding / session import dividers (contract §5, §6).
      const daemonRow = parseDaemonRow(msg);
      if (daemonRow) {
        finalizeCapsule();
        items.push(daemonRow);
        i++;
        continue;
      }
      // Worker onboarding: the send() marker right after the onboarding row
      // previews the whole setup instruction. The onboarding divider already
      // says what happened, so that one marker is dropped. Only the row
      // directly after it, and only for the same worker.
      if (isOnboardingInstructionMarker(msg, messages[i - 1])) {
        i++;
        continue;
      }
      // A completed terminal row that carries the worker's whole reply, when
      // no bubble from that worker has been drawn since the user's last
      // message: the reply IS the bubble. This is every turn of an imported
      // transcript (no dispatch marker, so the /chat endpoint joins no
      // transcript), and a live turn whose transcript join failed. When the
      // join did draw the bubble, today's one-line marker below stands.
      const replyBubble = terminalReplyBubble(msg, items);
      if (replyBubble) {
        finalizeCapsule();
        items.push(replyBubble);
        i++;
        continue;
      }
      // [Sub-agent] lifecycle markers — surface them as compact timeline rows.
      // Finalize the open capsule first so the marker appears AFTER the
      // capsule that contains the originating dispatch tool call (chronologic
      // JSONL order), not inside or before it.
      // Display split (backend _meta contract, same mechanism as the fanout
      // join summary above): on_completed's full content carries agent-facing
      // steering guidance for the LLM; _meta.display_content, when present,
      // is the clean marker text and is what SUB_AGENT_COMPLETED_RE parses
      // (its capture group is greedy to end-of-string, so parsing the full
      // content would leak the guidance into the rendered summary).
      const activityContent =
        typeof msg._meta?.display_content === 'string'
          ? (msg._meta.display_content as string)
          : (msg.content ?? '');
      const activity = parseSubAgentSystemMessage(activityContent, msg.timestamp);
      if (activity) {
        finalizeCapsule();
        items.push(applyDispatchFailureReason(activity, msg._meta, tr));
      }
      // Other system messages (ping-pong guard, etc.) remain dropped — they
      // are not user-facing.
      i++;
      continue;
    }

    // Detect session boundary changes — close any open capsule first.
    if (msg.session_id && currentSessionId && msg.session_id !== currentSessionId) {
      finalizeCapsule();
      items.push({ type: 'session_separator', timestamp: msg.timestamp });
    }
    if (msg.session_id) {
      currentSessionId = msg.session_id;
    }

    if (msg.role === 'user') {
      finalizeCapsule();
      items.push({
        type: 'user_message',
        content: msg.content ?? '',
        timestamp: msg.timestamp,
        ...(msg.target && { target: msg.target }),
      });
      i++;
      continue;
    }

    if (msg.role === 'agent') {
      finalizeCapsule();
      if (msg.chunk_type === 'approval_request') {
        items.push({
          type: 'approval_card',
          what: msg.content ?? '',
          tool_name: (msg._meta?.tool_name as string) ?? '',
          tool_call_id: (msg._meta?.tool_call_id as string) ?? '',
          tool_args: (msg._meta?.tool_args as Record<string, unknown>) ?? {},
          recent_activity: [],
        });
        i++;
        continue;
      }
      const cleaned = (msg.content ?? '').replace(/\x1b\[[0-9;]*m/g, '').trim();
      if (cleaned && cleaned !== '(no response)') {
        items.push({
          type: 'sub_agent_message',
          content: cleaned,
          source: msg.source,
          timestamp: msg.timestamp,
        });
      }
      i++;
      continue;
    }

    if (msg.role === 'assistant') {
      const text = msg.content && msg.content.trim() ? msg.content : null;
      const reasoning = (msg.reasoning_content ?? '').trim();
      const hasTools = !!(msg.tool_calls && msg.tool_calls.length > 0);
      const msTime = tsToMs(msg.timestamp);

      if (text) {
        // Visible text closes any open capsule, then emits inline.
        finalizeCapsule();
        items.push({
          type: 'agent_message',
          content: text,
          source: msg.source,
          timestamp: msg.timestamp,
        });
      } else if (reasoning || hasTools) {
        // Silent assistant turn. If we're already inside a capsule with
        // prior items, insert an empty agent_message marker so the UI
        // can delimit between silent tool batches.
        if (currentCapsule && currentCapsule.items.length > 0) {
          currentCapsule.items.push({
            type: 'agent_message',
            content: '',
            source: msg.source,
            timestamp: msg.timestamp,
          });
          currentCapsule.endedAtMs = msTime;
        }
      } else {
        // Never-vanish guard: an assistant message with empty content AND no
        // reasoning AND no tool_calls would otherwise hit neither branch and
        // emit NOTHING — a silent permanent disappearance. Emit a minimal
        // header-only marker so the turn is always represented in the UI.
        finalizeCapsule();
        items.push({
          type: 'agent_message',
          content: '',
          source: msg.source,
          timestamp: msg.timestamp,
          isHeaderOnly: true,
        });
      }

      // Machinery (reasoning + tool_calls) flows into the capsule that
      // follows the just-emitted visible text, or extends the current one.
      if (reasoning || hasTools) {
        if (!currentCapsule) {
          // FE-A3: a content-null assistant turn would otherwise produce a
          // bare capsule with no agent identity above it, visually attaching
          // to the preceding user message. Emit an `agent_message` header
          // marker (rendered as avatar + "agent · HH:MM" only) so the capsule
          // is anchored to a visible agent turn. Only emitted when we have no
          // visible text to anchor on — `text` already covered that case.
          if (!text) {
            items.push({
              type: 'agent_message',
              content: '',
              source: msg.source,
              timestamp: msg.timestamp,
              isHeaderOnly: true,
            });
          }
          // transformChatHistory only ever runs over PERSISTED (completed)
          // history, so a capsule produced here represents a COMPLETED turn.
          // Per the locked product decision a completed reasoning turn renders
          // COLLAPSED (clean summary); it expands only while actively RUNNING,
          // which is handled at render time in ChatView (running ⇒ force-expand).
          // Therefore never force-expand here — open collapsed.
          currentCapsule = openCapsuleAt(msg.timestamp, false);
        }
        // (No force-expand when appending reasoning to an existing capsule
        // either — see the comment above; completed turns stay collapsed.)
        if (reasoning) {
          currentCapsule.items.push({
            type: 'reasoning_block',
            content: reasoning,
            timestamp: msg.timestamp,
            turn_id: msg.timestamp,
          });
          currentCapsule.endedAtMs = msTime;
        }
        if (hasTools) {
          for (const tc of msg.tool_calls!) {
            if (tc.function.name === 'fanout') {
              // Standalone card, never a capsule row — but do NOT finalize
              // the capsule here. An earlier tool call in this SAME turn
              // (e.g. `read`) may still be sitting in it awaiting its own
              // result; the result-pairing branch below only searches the
              // OPEN capsule, so finalizing early would flush it before that
              // result arrives and the pairing would silently miss (a fixed
              // bug — the capsule is left open on purpose). It gets
              // finalized instead at ack-resolution time (the `role ===
              // 'tool'` branch below), right before the fanout_card is
              // pushed, which flushes it in the correct chronological spot.
              pendingFanoutCalls.set(tc.id, {
                labels: parseFanoutTaskLabels(tc.function.arguments),
                timestamp: msg.timestamp,
              });
              continue;
            }
            const activity = toolCallToActivity(tc, msg.timestamp, msg, workspace, tr);
            const dispatch = tc.function.name === 'agent_message'
              ? dispatchFromToolCall(tc.function.name, parseToolArgs(tc.function.arguments))
              : undefined;
            currentCapsule.items.push({
              type: 'tool_call_row',
              tool_name: activity.toolName,
              target_description: activity.description,
              tool_call_id: tc.id,
              category: activity.category,
              timestamp: msg.timestamp,
              result_content: null,
              result_status: 'pending',
              ...(dispatch ? { dispatch } : {}),
            });
            currentCapsule.endedAtMs = msTime;
          }
        }
      }

      i++;
      continue;
    }

    if (msg.role === 'tool') {
      // Spec 134 Part B: a background job's start joins its group's card,
      // which waits (deferred) until the capsule holding the shell call
      // finalizes. The row itself still pairs normally below.
      const bgJob = parseBackgroundJob(msg._meta);
      if (bgJob) addJobTask(bgJob.groupId, bgJob.task, msg.timestamp, true);
      const pendingFanout = pendingFanoutCalls.get(msg.tool_call_id ?? '');
      if (pendingFanout) {
        pendingFanoutCalls.delete(msg.tool_call_id ?? '');
        const content = typeof msg.content === 'string' ? msg.content : '';
        const match = content.match(FANOUT_ACK_RE);
        if (match) {
          const fanoutId = match[1];
          const card: Extract<DisplayItem, { type: 'fanout_card' }> = {
            type: 'fanout_card',
            fanout_id: fanoutId,
            tasks: pendingFanout.labels.map((label, idx) => ({
              handle: `worker:${fanoutId}-${idx}`,
              label,
            })),
            timestamp: pendingFanout.timestamp,
          };
          const hasOtherPending = !!currentCapsule?.items.some(
            (it) => it.type === 'tool_call_row' && it.result_status === 'pending',
          );
          if (hasOtherPending) {
            // Another tool call in this SAME turn (e.g. `read`) is still
            // awaiting its own result — finalizing now would flush it
            // pending and permanently orphan that result once it arrives
            // (residual of Critical 2: this is the [fanout, read] ordering,
            // ack processed before read's result). Defer instead of
            // emitting; finalizeCapsule() drains it once the capsule is
            // actually safe to flush (see the pairing branch below and the
            // drain step in finalizeCapsule itself).
            deferredFanoutCards.push(card);
          } else {
            // No other call is pending — safe to flush whatever capsule is
            // open right now, so the card lands in `items` in the correct
            // chronological position relative to it.
            finalizeCapsule();
            items.push(card);
          }
        }
        // No match (malformed/missing ack) — drop silently, mirroring the
        // orphan-tool-result precedent below.
        i++;
        continue;
      }
      if (currentCapsule) {
        const tcId = msg.tool_call_id ?? '';
        const content = typeof msg.content === 'string' ? msg.content : '';
        // Pair by tool_call_id with the matching tool_call_row inside the
        // currently open capsule. Search from the back since out-of-order
        // arrival pairs more recent calls more cheaply.
        for (let k = currentCapsule.items.length - 1; k >= 0; k--) {
          const item = currentCapsule.items[k];
          if (item.type === 'tool_call_row' && item.tool_call_id === tcId) {
            currentCapsule.items[k] = {
              ...item,
              result_content: content,
              result_status: 'received',
            };
            break;
          }
        }
        currentCapsule.endedAtMs = tsToMs(msg.timestamp);
        // A deferred fanout_card may have been waiting on THIS row (the
        // last still-pending one) to settle before the capsule is safe to
        // flush. Once nothing is pending anymore, finalize right away
        // instead of waiting for an unrelated later boundary — this is what
        // actually emits the card in the [fanout, read] ordering (ack
        // processed before read's result).
        if (
          deferredFanoutCards.length > 0 &&
          !currentCapsule.items.some((it) => it.type === 'tool_call_row' && it.result_status === 'pending')
        ) {
          finalizeCapsule();
        }
      }
      // Orphan tool results (no open capsule, or no matching call) are
      // dropped — the LLM never associated them with a visible call.
      i++;
      continue;
    }

    i++;
  }

  // End of stream: always finalize the trailing capsule as `completed`. The
  // transform output is now purely a function of persisted history; the
  // "running" status is applied at render time in ChatView (FE-A1) based on
  // the live agentStatus + viewingHolder, so this stays a pure transform.
  finalizeCapsule('completed');

  // Mark items from historical sessions
  let lastSepIndex = -1;
  for (let k = items.length - 1; k >= 0; k--) {
    if (items[k].type === 'session_separator') {
      lastSepIndex = k;
      break;
    }
  }
  if (lastSepIndex >= 0) {
    for (let k = 0; k <= lastSepIndex; k++) {
      const item = items[k];
      if (
        item.type !== 'session_separator' &&
        item.type !== 'approval_card' &&
        item.type !== 'compaction_marker'
      ) {
        (item as { isHistorical?: boolean }).isHistorical = true;
      }
    }
  }

  return items;
}

// Mirrored by the daemon's live tool_result preview (activity_translator.py);
// the shared toolResultPreviewFixtures.json tests fail if the two drift.
const RESULT_CHAR_BOUND = 500;
const RESULT_LINE_BOUND = 12;

/**
 * Mechanical truncation for tool result content displayed in the UI.
 * Returns the content unchanged when it fits both bounds; otherwise cuts
 * at whichever bound triggers first and adds a footer noting the totals.
 *
 * `totals` is for a preview the daemon already cut to these bounds (the live
 * tool_result event): the full result's size, so the cut is still marked and
 * the footer reports the whole result, exactly as after a reload.
 */
export function truncateResult(
  content: string,
  tr: ActivityTranslate = EN_ACTIVITY,
  totals?: ResultTotals,
): { text: string; footer: string | null } {
  const lines = content.split('\n');
  const totalChars = totals?.chars ?? content.length;
  const totalLines = totals?.lines ?? lines.length;

  if (totalChars <= RESULT_CHAR_BOUND && totalLines <= RESULT_LINE_BOUND) {
    return { text: content, footer: null };
  }

  // Char count of the first N lines (joined with their newlines). If this
  // is < RESULT_CHAR_BOUND, the line bound triggers earlier in the stream.
  const firstNLinesText = lines.slice(0, RESULT_LINE_BOUND).join('\n');
  const firstNLinesLen = firstNLinesText.length;

  const charBoundFires = totalChars > RESULT_CHAR_BOUND;
  const lineBoundFires = totalLines > RESULT_LINE_BOUND;

  // When both fire, pick whichever produces the shorter (earlier) cut.
  // When only one fires, that one wins outright.
  if (charBoundFires && (!lineBoundFires || RESULT_CHAR_BOUND <= firstNLinesLen)) {
    return {
      text: content.slice(0, RESULT_CHAR_BOUND) + '…',
      footer: tr('chat.result.firstChars', { n: RESULT_CHAR_BOUND, total: totalChars }),
    };
  }
  return {
    text: firstNLinesText + '…',
    footer: tr('chat.result.firstLines', { n: RESULT_LINE_BOUND, total: totalLines }),
  };
}

/**
 * REST-recovery merge for a single recovered assistant row (the text-bubble
 * half of ChatView's /chat?limit=1 fallback).
 *
 * Identity contract (INVESTIGATION-two-speakers): the recovered item's
 * `source` is taken VERBATIM from the persisted row — exactly how
 * transformChatHistory stamps it (`source: msg.source`) — never via a
 * hardcoded default. The old inline version hardcoded source: 'assistant',
 * which rendered one management turn as two speakers ("Assistant" with the
 * Orbit glyph vs raw "assistant" with the grey AS monogram).
 *
 * Merge semantics, in order:
 * 1. Dedup: if an identical-content agent_message exists ANYWHERE in
 *    `prevItems`, return `prevItems` unchanged (the production duplicate
 *    slipped through a last-item-only check because a system message
 *    followed the WS-built bubble).
 * 2. Truncation repair: if the LAST item is an agent_message whose content
 *    is a shorter prefix-era variant (stream cut mid-delta), replace its
 *    content with the full persisted text.
 * 3. Insert: add the recovered message before any trailing user messages so
 *    a recovered reply lands before follow-up questions.
 *
 * Gate (spec 131 A): a row carrying `_kind` (memory_record, memory_tidy, and
 * whatever daemon-tagged kind comes next) is never merged — the persisted
 * transform owns rendering for tagged rows (a collapsed capsule for the
 * edits, nothing for the text). The live path is silent for them by design,
 * so the idle catch-up's tail fetch lands on the record turn's closing text,
 * which the model often writes as a restatement of the reply; without this
 * gate it became a second bubble that only a reload removed.
 */
export function mergeRecoveredAssistantMessage(
  prevItems: DisplayItem[],
  row: ChatMessage,
): DisplayItem[] {
  if (row._kind) return prevItems;
  const text = row.content;
  if (!text) return prevItems;

  // 1. Dedup against the whole list, not just the tail.
  const exists = prevItems.some(
    (it) => it.type === 'agent_message' && it.content === text,
  );
  if (exists) return prevItems;

  // 2. Extend a truncated trailing message instead of duplicating it.
  const last = prevItems[prevItems.length - 1];
  if (
    last &&
    last.type === 'agent_message' &&
    text.length > last.content.length &&
    text.startsWith(last.content)
  ) {
    const updated = [...prevItems];
    updated[prevItems.length - 1] = { ...last, content: text };
    return updated;
  }

  // 3. Genuinely missing — insert before trailing user messages, identity
  //    taken verbatim from the row (same as normal ingest).
  const newMsg: DisplayItem = {
    type: 'agent_message',
    content: text,
    source: row.source,
    timestamp: row.timestamp ?? new Date().toISOString(),
  };
  let insertIdx = prevItems.length;
  while (insertIdx > 0 && prevItems[insertIdx - 1].type === 'user_message') {
    insertIdx--;
  }
  const updated = [...prevItems];
  updated.splice(insertIdx, 0, newMsg);
  return updated;
}

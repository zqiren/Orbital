// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

export type Autonomy = 'hands_off' | 'check_in' | 'supervised';

export interface ProviderInfo {
  display_name: string;
  base_url: string | null;
  china_base_url?: string | null;
  supports_model_list: boolean;
  sdk: 'openai' | 'anthropic';
  suggested_models: string[];
  notes: string;
}

export type ProviderRegistry = Record<string, ProviderInfo>;

export interface NotificationPrefs {
  task_completed?: boolean;
  errors?: boolean;
  agent_messages?: boolean;
  trigger_started?: boolean;
}

export interface Project {
  project_id: string;
  name: string;
  workspace: string;
  model: string;
  api_key: string;
  base_url: string | null;
  autonomy: Autonomy;
  instructions: string;
  provider?: string;
  sdk?: string;
  agent_name?: string;
  is_scratch?: boolean;
  project_goals_content?: string;
  user_directives_content?: string;
  notification_prefs?: NotificationPrefs;
  llm_fallback_models?: FallbackModelEntry[];
  budget_limit_usd?: number | null;
  budget_action?: 'stop' | 'ask';
  budget_spent_usd?: number;
}

export interface ProjectCreateRequest {
  name: string;
  workspace: string;
  model: string;
  api_key: string;
  base_url?: string | null;
  autonomy?: Autonomy;
  instructions?: string;
  provider?: string;
  sdk?: string;
  agent_name?: string;
  budget_limit_usd?: number | null;
}

export interface ProjectUpdateRequest {
  name?: string;
  model?: string;
  api_key?: string;
  base_url?: string | null;
  autonomy?: Autonomy;
  instructions?: string;
  provider?: string;
  sdk?: string;
  agent_name?: string;
  project_goals_content?: string;
  user_directives_content?: string;
  notification_prefs?: NotificationPrefs;
  llm_fallback_models?: FallbackModelEntry[];
  budget_limit_usd?: number | null;
  budget_spent_usd?: number;
}

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: ToolCallFunction;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'agent' | 'system';
  content: string | null;
  source: string;
  timestamp: string;
  target?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  reasoning_content?: string;
  _status?: string;
  _compaction?: boolean;
  _meta?: Record<string, unknown>;
  _activity_descriptions?: Record<string, string>;
  /**
   * Format-1 user-facing chat session identity. Stable within a chat thread;
   * rotates when ``/new-session`` is invoked. Treated as an opaque equality
   * key in chatTransform.ts (used only to draw session-boundary separators).
   * See TASK/ACTIVE-session-and-queue-model.md and the F7 audit for the
   * F1/F2 split.
   */
  session_id?: string;
  /**
   * Format-2 JSONL filename stem (debug-only). Optional because old
   * pre-canonical-rename messages do not have it; new messages may include
   * it. The frontend does not key UI off this field — it exists for
   * debugging and forward-compat.
   */
  session_uuid?: string;
  chunk_type?: string;
  /**
   * Set on synthetic rows the /chat endpoint injects after a sub-agent
   * dispatch marker (source === "sub_agent"). Carry the sub-agent's run
   * summary read from its own transcript so the management chat renders a
   * distinct `sub_agent_run` block. See agents_v2._interleave_sub_agent_summaries.
   */
  sub_agent_handle?: string;
  /** Per-tool rows synthesized from the sub-agent transcript: one entry per
   *  `[Using tool: X]` chunk, in chronological order. Name + duration only —
   *  the SDK transport streams no args/results. */
  sub_agent_tool_rows?: Array<{ name: string; timestamp: string; duration_seconds: number }>;
  sub_agent_duration?: number;
}

export type AgentRunStatus = 'running' | 'waiting' | 'idle' | 'error' | 'new_session' | 'pending_approval';

// Per-session record of the most recent terminal event. Returned by
// GET /api/v2/projects/{pid}/sessions on each session entry. Lets the UI
// render a persistent indicator (e.g., warning glyph for type='error')
// across WS reconnects and page reloads. Cleared by the backend when the
// session transitions out of the terminal state via a fresh /inject.
// See TASK/TASK-state-model-alignment-fixes.md §2.4.
export interface LastTerminalEvent {
  type: 'error' | 'stopped' | 'new_session';
  timestamp: string;
  details: string | null;
}

export interface SessionListEntry {
  session_id: string;
  status: AgentRunStatus;
  session_uuid: string | null;
  /**
   * Human-readable display label for the session. Auto-derived from the first
   * user message (truncated, word-boundary-aware) and user-editable via the
   * inline rename. null for legacy/headless sessions with no derivable name —
   * the UI falls back to the session_id. Display-only: never an identifier.
   */
  name?: string | null;
  last_terminal_event: LastTerminalEvent | null;
  /** ISO timestamp of the last activity in this session, or null. Added in Phase 1B. */
  last_activity_at?: string | null;
  /**
   * Origin of the session. Backend does NOT populate this field yet (Phase 1B
   * visual-only capability — real wiring ships in a later batch). When absent
   * (undefined) the UI renders the session as 'manual'. When origin === 'queue',
   * a subtle hue variation is applied to the status dot to signal the session
   * was dispatched by the queue rather than typed manually.
   */
  origin?: 'manual' | 'queue';
}

export interface AgentStatusEvent {
  type: 'agent.status';
  project_id: string;
  status: AgentRunStatus;
  source?: string;
  reason?: string;
  trigger_source?: string;
}

export interface StreamDeltaEvent {
  type: 'chat.stream_delta';
  project_id: string;
  session_id?: string;
  text: string;
  /**
   * Model reasoning for this delta. During the model's <think> phase this is
   * non-empty while `text` is empty (reasoning-only phase). Once visible answer
   * text begins, `text` is non-empty. Optional for backward compatibility with
   * older relay payloads.
   */
  reasoning_content?: string;
  source: string;
  is_final: boolean;
  seq?: number;
}

export type ActivityCategory =
  | 'file_read'
  | 'file_write'
  | 'file_edit'
  | 'file_search'
  | 'content_search'
  | 'command_exec'
  | 'web_search'
  | 'web_fetch'
  | 'request_access'
  | 'agent_message'
  | 'tool_use'
  | 'tool_result'
  | 'agent_output'
  | 'network_blocked'
  | 'browser_automation'
  | 'credential_request';

export interface ActivityEvent {
  type: 'agent.activity';
  project_id: string;
  session_id?: string;
  id: string;
  category: ActivityCategory;
  description: string;
  tool_name: string;
  source: string;
  timestamp: string;
}

export interface StatusSummaryEvent {
  type: 'agent.status_summary';
  project_id: string;
  summary: string;
  timestamp: string;
}

export interface ApprovalRequestEvent {
  type: 'approval.request';
  project_id: string;
  session_id?: string;
  what: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: Record<string, unknown>;
  recent_activity: ChatMessage[];
  reasoning?: string;
}

export interface ApprovalResolvedEvent {
  type: 'approval.resolved';
  project_id: string;
  session_id?: string;
  tool_call_id: string;
  resolution: 'approved' | 'denied';
}

export interface SubAgentMessageEvent {
  type: 'chat.sub_agent_message';
  project_id: string;
  session_id?: string;
  content: string;
  source: string;
  timestamp: string;
}

/**
 * Sub-agent lifecycle broadcasts (Piece 3 Part D). `sub_agent.stopped`
 * carries the honest record of tracked background work the user stop
 * terminated.
 */
export interface SubAgentLifecycleEvent {
  type:
    | 'sub_agent.started'
    | 'sub_agent.completed'
    | 'sub_agent.error'
    | 'sub_agent.failed'
    | 'sub_agent.stopped';
  project_id: string;
  session_id?: string | null;
  handle: string;
  initiator?: string;
  summary?: string;
  error?: string;
  reason?: string;
  background_terminated?: string[];
}

/** Sub-agent status as reported by GET /agents/{id}/sub-agents/status.
 * 'background-running' = turn done but tracked background work is alive
 * (SDK claude-code only; other transports report two-state). */
export type SubAgentRunStatus = 'running' | 'background-running' | 'idle';

export interface UserMessageEvent {
  type: 'chat.user_message';
  project_id: string;
  session_id?: string;
  content: string;
  nonce: string;
  timestamp: string;
}

export interface AgentNotifyEvent {
  type: 'agent.notify';
  project_id: string;
  session_id?: string;
  title: string;
  body: string;
  urgency: 'high' | 'normal' | 'low';
  timestamp: string;
}

export interface DeviceStatusEvent {
  type: 'device.status';
  status: 'online' | 'offline';
}

export interface TriggerCreatedEvent {
  type: 'trigger.created';
  project_id: string;
  trigger: Trigger;
}

export interface TriggerDeletedEvent {
  type: 'trigger.deleted';
  project_id: string;
  trigger_id: string;
}

export interface TriggerFiredEvent {
  type: 'trigger.fired';
  project_id: string;
  trigger_id: string;
  trigger_name: string;
  timestamp: string;
}

export interface TriggerSkippedEvent {
  type: 'trigger.skipped';
  project_id: string;
  trigger_id: string;
  trigger_name: string;
  reason: string;
  timestamp: string;
}

export interface StateRefreshLifecycleEvent {
  type: 'state_refresh.lifecycle';
  project_id: string;
  session_id?: string;
  status: 'in_progress' | 'done' | 'failed' | 'skipped';
  trigger: 'turn_count' | 'agent_decided' | 'token_pressure';
  timestamp: string;
}

export interface WorkspaceClaudemdWarningEvent {
  type: 'workspace_claudemd_warning';
  project_id: string;
  claudemd_path: string;
  content_hash: string;
  matched_token: string;
}

export interface QueueItemAddedEvent {
  type: 'queue.item_added';
  project_id: string;
  item_id: string;
}

export interface QueueItemEditedEvent {
  type: 'queue.item_edited';
  project_id: string;
  item_id: string;
}

export interface QueueItemRemovedEvent {
  type: 'queue.item_removed';
  project_id: string;
  item_id: string;
}

export interface QueueItemAdvancedEvent {
  type: 'queue.item_advanced';
  project_id: string;
  item_id: string;
  outcome: 'completed' | 'blocked' | 'interrupted';
}

export interface QueueStateChangedEvent {
  type: 'queue.state_changed';
  project_id: string;
  state: QueueRunState;
}

export interface QueueReorderedEvent {
  type: 'queue.reordered';
  project_id: string;
}

/** One entry in a blocked-count snapshot: which project+session is blocked. */
export interface BlockedSessionEntry {
  project_id: string;
  session_id: string;
}

/**
 * Global WS event fired whenever any session enters or leaves `pending_approval`.
 * Not scoped to a project — broadcast to all subscribers.
 */
export interface BlockedCountChangedEvent {
  type: 'blocked-count-changed';
  blocked_count: number;
  blocked_sessions: BlockedSessionEntry[];
}

export type WebSocketEvent =
  | AgentStatusEvent
  | StreamDeltaEvent
  | ActivityEvent
  | StatusSummaryEvent
  | ApprovalRequestEvent
  | ApprovalResolvedEvent
  | SubAgentMessageEvent
  | SubAgentLifecycleEvent
  | UserMessageEvent
  | AgentNotifyEvent
  | DeviceStatusEvent
  | TriggerCreatedEvent
  | TriggerDeletedEvent
  | TriggerFiredEvent
  | TriggerSkippedEvent
  | StateRefreshLifecycleEvent
  | WorkspaceClaudemdWarningEvent
  | QueueItemAddedEvent
  | QueueItemEditedEvent
  | QueueItemRemovedEvent
  | QueueItemAdvancedEvent
  | QueueStateChangedEvent
  | QueueReorderedEvent
  | BlockedCountChangedEvent;

// Queue resource types (mirror agent_os/queue/models.py)
export type QueueItemState = 'queued' | 'running' | 'done' | 'blocked';
export type QueueAttemptOutcome = 'completed' | 'blocked' | 'interrupted' | 'cancelled';
// Mirrors the backend QueueRunState enum (agent_os/queue/models.py):
// running = actively dispatching/working, paused = user-stopped, idle = nothing
// to dispatch. (Earlier frontend used 'draining'|'stopped', which never matched
// the backend and silently broke composer-gating + pause detection.)
export type QueueRunState = 'running' | 'paused' | 'idle';

export interface QueueAttempt {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  outcome: QueueAttemptOutcome | null;
  summary: string | null;
  block_reason: string | null;
}

export interface QueueItem {
  id: string;
  content: string;
  file_refs: string[];
  priority: number;
  review_before_advance: boolean;
  state: QueueItemState;
  /**
   * Origin of the queue item. Backend currently emits only 'user' | 'upload'.
   * 'trigger' is a Phase 1B visual-only extension — the backend does NOT
   * populate this value yet; real wiring ships in a later batch.
   */
  source: 'user' | 'upload' | 'trigger';
  /**
   * Human-readable name of the trigger that spawned this item.
   * Only present when source === 'trigger'. Backend does not emit this yet
   * (Phase 1B visual-only; real wiring is a later batch).
   */
  trigger_name?: string;
  /**
   * Stable ID of the trigger that spawned this item.
   * Only present when source === 'trigger'. Backend does not emit this yet
   * (Phase 1B visual-only; real wiring is a later batch).
   */
  trigger_id?: string;
  attempts: QueueAttempt[];
  idempotency_key: string | null;
  interrupted_count: number;
  created_at: string;
}

export interface QueueSnapshot {
  version: number;
  state: QueueRunState;
  items: QueueItem[];
  chat_session_id: string | null;
}

export interface FileEntry {
  name: string;
  type: 'file' | 'directory';
  size?: number;
  modified_at?: number;
}

export interface DirectoryListing {
  path: string;
  entries: FileEntry[];
}

export interface FileContent {
  path: string;
  content: string;
  size: number;
  truncated: boolean;
  type?: 'text' | 'image' | 'binary';
  mime?: string;
  download_url?: string;
}

export interface PlatformStatus {
  status: string;
  platform: string;
  isolation_method: string;
  setup_complete: boolean;
  setup_issues: string[];
  supports_network_restriction: boolean;
  supports_folder_access: boolean;
  sandbox_username: string | null;
}

export interface FolderInfo {
  path: string;
  display_name: string;
  accessible: boolean;
  access_note: string | null;
}

export interface FallbackModelEntry {
  provider: string;
  model: string;
  base_url?: string | null;
  api_key?: string | null;
  sdk: string;
}

export interface TriggerSchedule {
  cron: string;
  human: string;
  timezone: string;
}

export interface Trigger {
  id: string;
  name: string;
  enabled: boolean;
  type: 'schedule' | 'file_watch';
  schedule?: TriggerSchedule;
  watch_path?: string;
  patterns?: string[];
  recursive?: boolean;
  debounce_seconds?: number;
  task: string;
  autonomy: string | null;
  last_triggered: string | null;
  trigger_count: number;
  created_at: string;
}

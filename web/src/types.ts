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
  /** True when the workspace has no user content (ignoring orbital/ + dotfiles).
   *  Gates the first-session cold-start scan consent card. */
  is_empty_workspace?: boolean;
  project_goals_content?: string;
  user_directives_content?: string;
  notification_prefs?: NotificationPrefs;
  llm_fallback_models?: FallbackModelEntry[];
  budget_limit_usd?: number | null;
  /** Enforcement behavior at the limit. "pause" auto-resumes; "stop" is hard. */
  budget_action?: 'pause' | 'stop';
  /** ISO 4217 currency the limit is denominated in (set-once, user-overridable). */
  budget_currency?: string;
  /** Spend window the limit applies over. */
  budget_period?: 'daily' | 'weekly' | 'monthly' | 'total';
  /** ISO timestamp the "total" window counts from (the reset anchor). */
  budget_anchor_ts?: string | null;
  /** Sub-agent slugs hidden from the management agent for this project. */
  disabled_sub_agents?: string[];
  /** Connector ids enabled for this project (Spec 011 §0.2 — authenticate
   *  globally, enable per project). Only enabled connectors' tools are
   *  reflected into this project's registry. */
  enabled_connectors?: string[];
  /** TOFU network grants (Plan 2). Bare registrable domains approved for this
   *  project's sandboxed shell/sub-agents, beyond the built-in defaults. */
  approved_domains?: string[];
  /** TOFU asks awaiting a decision — auto-denied under hands-off autonomy
   *  (Plan 2 Task 5) rather than lost; surfaced in Settings → Network access. */
  pending_domain_requests?: PendingDomainRequest[];
}

/** One TOFU network-access ask (Plan 2), as carried by
 *  Project.pending_domain_requests. */
export interface PendingDomainRequest {
  domain: string;
  reason: string;
  requested_at: string;
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
  budget_action?: 'pause' | 'stop';
  budget_currency?: string;
  budget_period?: 'daily' | 'weekly' | 'monthly' | 'total';
  budget_anchor_ts?: string | null;
  /** P3-F: total-mode spend-window reset. Replaces the legacy budget_spent_usd
   *  reset sentinel — sets budget_anchor_ts = now (server-side). */
  reset_budget_anchor?: boolean;
  /** Sub-agent slugs hidden from the management agent for this project. */
  disabled_sub_agents?: string[];
  /** Connector ids enabled for this project (rides the existing
   *  project-update flow — per-project enablement is NOT a connector route). */
  enabled_connectors?: string[];
  /** TOFU network grants (Plan 2) — see Project.approved_domains. */
  approved_domains?: string[];
  /** TOFU pending requests (Plan 2) — see Project.pending_domain_requests.
   *  Dismissals persist through this same PUT. */
  pending_domain_requests?: PendingDomainRequest[];
}

/** PUT /projects response field carrying the reset outcome (codes only). */
export interface BudgetResetOutcome {
  applied: boolean;
  code: 'not_total_mode' | null;
  budget_anchor_ts: string | null;
}

/** Cross-project read-scope mode for a scratch (Quick Tasks) session. */
export type SessionScopeMode = 'all' | 'selected' | 'off';

/**
 * Per-session cross-project READ scope record (Spec 012), carried by
 * GET/PUT /api/v2/agents/{project_id}/sessions/{session_id}/scope.
 *
 * Reads resolve against the in-scope project workspaces; writes always stay
 * in the scratch (Quick Tasks) workspace. Only the scratch project accepts
 * PUT (409 otherwise); PUT returns the stored record. New scratch sessions
 * default to mode 'all'.
 */
export interface SessionScope {
  mode: SessionScopeMode;
  selected_project_ids: string[];
}

/** Connector catalog status (Spec 011). ``pending_verification`` entries are
 *  visible in the catalog but not connectable (e.g. Gmail while Google's
 *  restricted-scope verification is in flight). */
export type ConnectorStatus = 'available' | 'pending_verification';

/** The locked auth-spec enum (Spec 011 §0.5). The launch catalog only uses
 *  'oauth2'; 'none' covers unauthenticated custom Tier-0 servers. */
export type ConnectorAuthType = 'oauth2' | 'app_password' | 'local_native' | 'none';

/**
 * A connector card from GET /api/v2/connectors: the catalog manifest fields
 * flattened together with live connection status (Task B3 `_serialize`).
 *
 * NOTE: tokens are PROVIDER-scoped — connectors sharing an `auth_provider`
 * (e.g. every Google service) share one account token, so disconnecting any
 * of them disconnects the whole provider.
 */
export interface Connector {
  id: string;
  name: string;
  icon: string;
  auth_provider: string;
  auth_type: ConnectorAuthType;
  server_url: string | null;
  oauth_scopes: string[];
  tool_overrides: Record<string, 'read' | 'write'>;
  /** Drives the onboarding "connect your accounts" card. */
  featured: boolean;
  status: ConnectorStatus;
  connected: boolean;
  /** Account identity the provider token is bound to (null when disconnected). */
  account: string | null;
  /** Non-null when the connector is enabled somewhere but unusable (e.g. its
   *  MCP session failed) — dynamic backend text, rendered untranslated. */
  enabled_error: string | null;
}

/** Response shape of GET /api/v2/connectors. */
export interface ConnectorListResponse {
  connectors: Connector[];
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
  auto_denied?: boolean;
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
    | 'sub_agent.stopped'
    | 'sub_agent.turn_interrupted';
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

/**
 * One entry in a sub-agent transcript, returned by
 * GET /api/v2/agents/{project_id}/sub-agents/{handle}/transcript (spec 009
 * §0.5, Task 4's contract). Read-only playback for the drill-in view.
 */
export interface SubAgentTranscriptEntry {
  source: string;
  content: string;
  timestamp: string;
  chunk_type: string;
}

export interface SubAgentTranscriptResult {
  handle: string;
  display_name: string;
  kind: 'worker' | 'cli';
  resumable: boolean;
  entries: SubAgentTranscriptEntry[];
  /**
   * The worker's own F2 session stem (spec 009 §0.5 Task D, issues 2+3):
   * lets the drill-in render the worker's chat history chat-shaped instead
   * of the flat transcript entries. Live via the fanout registry mid-batch,
   * disk fallback afterward; null for CLI handles and older fanouts that
   * predate the field.
   */
  session_uuid?: string | null;
}

/** Fanout task status vocabulary (spec 009 §0.5), reported per worker handle
 *  via `fanout.task_update`. */
export type FanoutTaskStatus = 'running' | 'completed' | 'error' | 'stalled' | 'interrupted';

/** One dispatched task, as carried by `fanout.started`. `session_uuid` is the
 *  worker's own F2 session stem (spec 009 §0.5 Task D) — null for CLI handles
 *  or older daemons that predate the field. */
export interface FanoutTaskDescriptor {
  handle: string;
  label: string;
  session_uuid?: string | null;
}

export interface FanoutStartedEvent {
  type: 'fanout.started';
  project_id: string;
  session_id?: string;
  fanout_id: string;
  tasks: FanoutTaskDescriptor[];
}

export interface FanoutTaskUpdateEvent {
  type: 'fanout.task_update';
  project_id: string;
  session_id?: string;
  fanout_id: string;
  handle: string;
  status: FanoutTaskStatus;
  /**
   * Wall-clock ms the task hit a terminal status, stamped server-side
   * (round 2, issue 1: per-task countdown). Absent on older daemons — the
   * frontend falls back to its own arrival-time stamp when the status is
   * terminal and this is missing.
   */
  completed_at_ms?: number;
}

/** One task's terminal snapshot as carried by `fanout.completed` (round 2:
 *  the event previously carried no per-task detail at all — every field here
 *  is additive on top of the pre-round-2 `{type, project_id, session_id,
 *  fanout_id}` shape). */
export interface FanoutCompletedTaskEntry {
  handle: string;
  label: string;
  status: FanoutTaskStatus;
  transcript_path?: string | null;
  completed_at_ms?: number;
  session_uuid?: string | null;
}

export interface FanoutCompletedEvent {
  type: 'fanout.completed';
  project_id: string;
  session_id?: string;
  fanout_id: string;
  tasks: FanoutCompletedTaskEntry[];
}

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

/**
 * Pending-input queue (spec 006, Path A). A message the user sent to session B
 * while session A holds the project's single active-loop slot was ACCEPTED and
 * queued; it dispatches when the slot frees. These three events are additive
 * (unknown types are ignored by old clients) and carry the canonical target
 * `session_id` (the queued session B), never the holder's.
 */
export interface PendingEnqueuedEvent {
  type: 'chat.pending_enqueued';
  project_id: string;
  /** The queued chat session (B) the message will dispatch into. */
  session_id: string;
  /** The session (A) currently holding the slot — shown in the wait notice. */
  holder: string;
  /** Origin's nonce, so other clients render the optimistic bubble and the
   *  origin tab dedups its own echo. */
  nonce: string;
  /** Full message content, so other clients can render the optimistic bubble. */
  content: string;
  /** FIFO position in the project's pending queue at enqueue time. */
  position: number;
}

export interface PendingDispatchedEvent {
  type: 'chat.pending_dispatched';
  project_id: string;
  /** The queued session (B) whose pending message just started dispatching. */
  session_id: string;
  /** The dispatched batch's nonce (clears that nonce's waiting affordance).
   *  Nullable to mirror the backend (a pending entry may carry no nonce). */
  nonce: string | null;
}

export interface PendingCancelledEvent {
  type: 'chat.pending_cancelled';
  project_id: string;
  /** The queued session (B) whose pending message was cancelled. */
  session_id: string;
  /** The cancelled entry's nonce (removes its optimistic bubble + map entry).
   *  Nullable to mirror the backend cancel-by-session path. */
  nonce: string | null;
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
  trigger: 'turn_count' | 'agent_decided' | 'agent_decided_coalesced' | 'token_pressure';
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
  /**
   * Why the queue is paused, when state === 'paused'. A CODE, never a display
   * string: 'user' for a manual/timed user pause, 'budget' for an enforcement
   * trip. null/absent when not paused. Carried so the UI can render a budget-
   * specific banner without re-reading config. Mirrors QueueState.pause_reason.
   */
  pause_reason?: 'user' | 'budget' | null;
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
 * One entry in the budget-paused-projects list: which project's queue is paused
 * for budget enforcement. Codes only (no spend numbers — the corner/Budget
 * section read those from GET /cost). Surfaced by GET /api/v2/blocked so the
 * sidebar Blocked surface can reason-code budget pauses distinctly from
 * approval-pending entries.
 */
export interface BudgetPausedProjectEntry {
  project_id: string;
}

/**
 * Global WS event fired whenever any session enters or leaves `pending_approval`.
 * Not scoped to a project — broadcast to all subscribers.
 */
export interface BlockedCountChangedEvent {
  type: 'blocked-count-changed';
  blocked_count: number;
  blocked_sessions: BlockedSessionEntry[];
  /**
   * Projects whose queue is currently paused for budget. Optional/back-compat:
   * absent on older daemons. Codes only — the sidebar reason-codes the marker.
   */
  budget_paused_projects?: BudgetPausedProjectEntry[];
}

/**
 * Management spend crossed the debounce threshold (≥1s) for a project's
 * current window. Display-only nudge: the Budget surface re-fetches GET /cost
 * rather than trusting the inline numbers. Carries the window + the
 * management spend/limit (ISO currency code) so a header corner can render
 * without a fetch if it wants. Sub-agent spend is NOT included (never
 * enforced).
 */
export interface BudgetSpendUpdatedEvent {
  type: 'budget.spend_updated';
  project_id: string;
  window: 'daily' | 'weekly' | 'monthly' | 'total';
  spend: number;
  limit: number | null;
  currency: string;
}

export type WebSocketEvent =
  | AgentStatusEvent
  | BudgetSpendUpdatedEvent
  | StreamDeltaEvent
  | ActivityEvent
  | StatusSummaryEvent
  | ApprovalRequestEvent
  | ApprovalResolvedEvent
  | SubAgentMessageEvent
  | SubAgentLifecycleEvent
  | UserMessageEvent
  | AgentNotifyEvent
  | PendingEnqueuedEvent
  | PendingDispatchedEvent
  | PendingCancelledEvent
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
  | BlockedCountChangedEvent
  | FanoutStartedEvent
  | FanoutTaskUpdateEvent
  | FanoutCompletedEvent;

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
  /** UTC ISO deadline for a timed pause; null/absent = paused until resumed. */
  paused_until?: string | null;
  /**
   * Why the queue is paused, when state === 'paused'. A CODE, never a display
   * string: 'user' for a manual/timed user pause, 'budget' for an enforcement
   * trip. null/absent when not paused. Mirrors the backend QueueState field;
   * the queue snapshot serializes it via model_dump.
   */
  pause_reason?: 'user' | 'budget' | null;
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
  type?: 'text' | 'image' | 'binary' | 'html';
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

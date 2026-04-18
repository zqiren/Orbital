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
  _status?: string;
  _compaction?: boolean;
  _meta?: Record<string, unknown>;
  _activity_descriptions?: Record<string, string>;
  session_id?: string;
  chunk_type?: string;
}

export type AgentRunStatus = 'running' | 'waiting' | 'idle' | 'stopped' | 'error' | 'new_session' | 'pending_approval';

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
  text: string;
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
  tool_call_id: string;
  resolution: 'approved' | 'denied';
}

export interface SubAgentMessageEvent {
  type: 'chat.sub_agent_message';
  project_id: string;
  content: string;
  source: string;
  timestamp: string;
}

export interface UserMessageEvent {
  type: 'chat.user_message';
  project_id: string;
  content: string;
  nonce: string;
  timestamp: string;
}

export interface AgentNotifyEvent {
  type: 'agent.notify';
  project_id: string;
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

export type WebSocketEvent =
  | AgentStatusEvent
  | StreamDeltaEvent
  | ActivityEvent
  | StatusSummaryEvent
  | ApprovalRequestEvent
  | ApprovalResolvedEvent
  | SubAgentMessageEvent
  | UserMessageEvent
  | AgentNotifyEvent
  | DeviceStatusEvent
  | TriggerCreatedEvent
  | TriggerDeletedEvent
  | TriggerFiredEvent
  | TriggerSkippedEvent;

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

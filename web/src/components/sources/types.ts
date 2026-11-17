// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources (任务来源) — the daemon's shapes, as written in the backend
 * contract (`/api/v2/sources`, v1 2026-11-16). The panel renders every form
 * from the catalogue; nothing here names a service's fields.
 */

export type SourceFamily = 'account' | 'bot' | 'check' | 'external';
export type ConnectMethod = 'cli_account' | 'fields' | 'qr' | 'none';

export interface CatalogField {
  key: string;
  label_key: string;
  secret: boolean;
  optional: boolean;
  default: string | null;
  placeholder: string | null;
  /** Side by side with the next pair field (server + port). */
  pair: boolean;
}

export interface CatalogParam {
  key: string;
  label_key: string;
  type: 'choice' | 'bool' | 'text';
  /** A list name for `POST /sources/choices` (repos, labels, folders). */
  choices: string | null;
  /** Fixed options (calendar minutes). */
  options: Array<string | number> | null;
  depends_on: string | null;
  required: boolean;
  default?: unknown;
}

export interface CatalogKind {
  key: string;
  label_key: string;
  default: boolean;
  params: CatalogParam[];
}

export interface CatalogConnect {
  method: ConnectMethod;
  cli?: string;
  host?: string;
  sign_in_command?: string;
  install_command?: string;
  install_url?: string;
  fields?: CatalogField[];
  /** A source-level multi-select (calendar: which calendars). Nothing stored = every one. */
  pick?: CatalogPick;
}

export interface CatalogPick {
  /** The config key it is stored under. */
  key: string;
  /** The list name for `POST /sources/choices`. */
  choices: string;
  multi: boolean;
}

export interface CatalogType {
  type: string;
  family: SourceFamily;
  available: boolean;
  unavailable_reason: 'not_built' | 'platform' | null;
  platforms: string[];
  interval_s: number;
  lossy: boolean;
  connect: CatalogConnect;
  kinds: CatalogKind[];
  setup_link: string | null;
  agent_prompt: { zh: string; en: string } | null;
}

export interface Catalog {
  platform: string;
  types: CatalogType[];
}

export type FailureAction = 'check_again' | 'reconnect' | 'sign_in' | 'open_settings' | 'remove';

export interface SourceFailure {
  code: string;
  params: Record<string, string | number | null>;
  at: string | null;
  actions: FailureAction[];
}

export interface RegisteredBy {
  by: 'agent' | 'outside';
  agent: string | null;
  project_id: string | null;
  project_name: string | null;
  session_id: string | null;
  at: string | null;
}

export interface SourceSummary {
  id: string;
  type: string;
  family: SourceFamily;
  name: string;
  display_name: string;
  label: string | null;
  on: boolean;
  is_new: boolean;
  registered_by: RegisteredBy | null;
  status: 'ok' | 'failed' | 'unavailable';
  failure: SourceFailure | null;
  last_check_at: string | null;
  waiting: number;
  held: number;
  catch_up: { count: number } | null;
  created_at: string;
}

export type ItemState = 'waiting' | 'routing' | 'routed' | 'unrouted' | 'held' | 'dismissed';
export type SessionState = 'running' | 'needs_you' | 'done';

export interface HistoryUpdate {
  title: string;
  sender: string | null;
  at: string;
}

export interface HistoryPlace {
  kind: 'dm' | 'group' | 'channel';
  name?: string | null;
}

export interface HistoryItem {
  id: string;
  kind: 'item';
  state: ItemState;
  title: string;
  sender: string | null;
  place: HistoryPlace | null;
  url: string | null;
  arrived_at: string;
  updated_at?: string | null;
  count: number;
  updates: HistoryUpdate[];
  project_id: string | null;
  project_name: string | null;
  routed_by: 'model' | 'fallback' | 'user' | null;
  session_id: string | null;
  session_state: SessionState | null;
  recovered_at: string | null;
  reason: 'routing_failed' | 'missing_api_key' | 'no_project' | 'send_failed' | null;
}

export interface HistoryGap {
  id: string;
  kind: 'gap';
  start: string;
  end: string | null;
}

export type HistoryEntry = HistoryItem | HistoryGap;

export interface SourceDetail extends SourceSummary {
  kinds: Record<string, Record<string, unknown>>;
  account: string | null;
  config: Record<string, unknown>;
  secrets: Record<string, { set: boolean; masked: string }>;
  command: { run: string; cwd: string | null; interval_s: number } | null;
  instruction: string;
  history: HistoryEntry[];
  history_more: boolean;
}

export interface SourcesList {
  sources: SourceSummary[];
  waiting: number;
}

export interface CliAccount {
  login: string;
  host: string;
  active: boolean;
}

export interface AccountsResponse {
  type: string;
  cli: string;
  state: 'signed_in' | 'signed_out' | 'not_installed';
  accounts: CliAccount[];
  sign_in_command: string;
  install_command: string;
}

export type QrState = 'pending' | 'done' | 'expired' | 'failed';

export interface QrStart {
  qr_id: string;
  state: QrState;
  qr_text: string;
  expires_at: string | null;
}

export interface QrStatus {
  qr_id: string;
  state: QrState;
  label: string | null;
}

export interface ChoiceItem {
  value: string;
  label: string;
  /** What the list groups it under (a calendar's account). */
  group?: string | null;
}

/** `400 verify_failed` / `422 invalid` / `409` details, parsed. */
export interface SourceError {
  code: string;
  field: string | null;
  reason: string | null;
  message: string | null;
}

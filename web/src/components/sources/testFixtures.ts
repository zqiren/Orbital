// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/** Test-only builders for the sources contract shapes (imported by tests only). */

import type { CatalogType, HistoryItem, SourceDetail, SourceSummary } from './types';

export function summary(o: Partial<SourceSummary> = {}): SourceSummary {
  return {
    id: 'src_1', type: 'github', family: 'account', name: '', display_name: 'GitHub', label: 'mira-chen',
    on: true, is_new: false, registered_by: null, status: 'ok', failure: null,
    last_check_at: '2026-11-16T07:20:00Z', waiting: 0, held: 0, catch_up: null,
    created_at: '2026-11-16T07:00:00Z', ...o,
  };
}

export function detail(o: Partial<SourceDetail> = {}): SourceDetail {
  return {
    ...summary(o), kinds: { assigned: {} }, account: 'mira-chen', config: {}, secrets: {}, command: null,
    instruction: '', history: [], history_more: false, ...o,
  };
}

export function item(o: Partial<HistoryItem> = {}): HistoryItem {
  return {
    id: 'arr_1', kind: 'item', state: 'routed', title: '#899 Cart total', sender: null, place: null, url: null,
    arrived_at: '2026-11-16T06:55:00Z', count: 1, updates: [], project_id: 'p1', project_name: 'client-portal',
    routed_by: 'model', session_id: 's1', session_state: 'running', recovered_at: null, reason: null, ...o,
  };
}

const base = { available: true, unavailable_reason: null, platforms: ['darwin', 'win32', 'linux'], interval_s: 300, lossy: false, setup_link: null, agent_prompt: null };

export const GITHUB: CatalogType = {
  ...base, type: 'github', family: 'account',
  connect: { method: 'cli_account', cli: 'gh', host: 'github.com', sign_in_command: 'gh auth login', install_command: 'brew install gh' },
  kinds: [
    { key: 'assigned', label_key: 'pick.assigned', default: true, params: [] },
    { key: 'mention', label_key: 'pick.mention', default: false, params: [] },
    { key: 'label', label_key: 'pick.label', default: false, params: [
      { key: 'repo', label_key: 'repo.ph', type: 'choice', choices: 'repos', options: null, depends_on: null, required: true },
      { key: 'label', label_key: 'label.ph', type: 'choice', choices: 'labels', options: null, depends_on: 'repo', required: true },
    ] },
  ],
};

export const EMAIL: CatalogType = {
  ...base, type: 'email', family: 'account',
  connect: { method: 'fields', fields: [
    { key: 'acct', label_key: 'fl.acct', secret: false, optional: false, default: null, placeholder: 'you@example.com', pair: false },
    { key: 'pass', label_key: 'fl.pass', secret: true, optional: false, default: null, placeholder: 'xxxx', pair: false },
    { key: 'server', label_key: 'fl.server', secret: false, optional: false, default: null, placeholder: 'imap.example.com', pair: true },
    { key: 'port', label_key: 'fl.port', secret: false, optional: false, default: '993', placeholder: null, pair: true },
  ] },
  kinds: [{ key: 'new_mail', label_key: 'pick.newmail', default: true, params: [
    { key: 'folder', label_key: 'fl.folder', type: 'choice', choices: 'folders', options: null, depends_on: null, required: false, default: 'INBOX' },
    { key: 'unread_only', label_key: 'pick.unread', type: 'bool', choices: null, options: null, depends_on: null, required: false, default: true },
  ] }],
  agent_prompt: { zh: '帮我把我的邮箱接入 Orbital', en: 'Connect my mailbox to Orbital' },
};

export const TELEGRAM: CatalogType = {
  ...base, type: 'telegram', family: 'bot', setup_link: 'https://t.me/BotFather',
  connect: { method: 'fields', fields: [
    { key: 'token', label_key: 'fl.btoken', secret: true, optional: false, default: null, placeholder: '123456789:AAH…', pair: false },
  ] },
  kinds: [
    { key: 'dm', label_key: 'pick.dm', default: true, params: [] },
    { key: 'at', label_key: 'pick.at', default: true, params: [] },
  ],
  agent_prompt: { zh: '帮我把一个 Telegram 机器人接入 Orbital', en: 'Connect a Telegram bot to Orbital' },
};

export const FEISHU: CatalogType = {
  ...TELEGRAM, type: 'feishu', lossy: true, setup_link: 'https://open.feishu.cn/app',
  connect: { method: 'qr', fields: [
    { key: 'appid', label_key: 'fl.appid', secret: false, optional: false, default: null, placeholder: 'cli_…', pair: false },
    { key: 'appsec', label_key: 'fl.appsec', secret: true, optional: false, default: null, placeholder: null, pair: false },
  ] },
};

export const CALENDAR: CatalogType = {
  ...base, type: 'calendar', family: 'account', platforms: ['darwin'],
  connect: { method: 'none', pick: { key: 'calendars', choices: 'calendars', multi: true } },
  kinds: [
    { key: 'new_event', label_key: 'pick.newev', default: false, params: [] },
    { key: 'before', label_key: 'pick.before', default: true, params: [
      { key: 'minutes', label_key: 'mins', type: 'choice', choices: null, options: [10, 15, 30, 60], depends_on: null, required: true, default: 30 },
    ] },
  ],
};

export const WEBSITE: CatalogType = {
  ...base, type: 'website', family: 'check',
  connect: { method: 'fields', fields: [
    { key: 'url', label_key: 'web.addr', secret: false, optional: false, default: null, placeholder: 'https://', pair: false },
    { key: 'must_contain', label_key: 'web.has', secret: false, optional: true, default: null, placeholder: null, pair: false },
  ] },
  kinds: [{ key: 'down', label_key: 'pick.down', default: true, params: [] }],
};

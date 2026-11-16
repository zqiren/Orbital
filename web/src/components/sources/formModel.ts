// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — the catalogue → form model. Every connect form is drawn from
 * the daemon's catalogue entry for its type: the connect method (a CLI
 * account, fixed fields, a QR scan, nothing), the arrival kinds and their
 * params. Pure functions; the components only render what these return.
 */

import type { Locale } from '../../i18n/locales';
import type { CreateBody } from './sourcesApi';
import type { Catalog, CatalogField, CatalogType, QrState } from './types';

export type Kinds = Record<string, Record<string, unknown>>;

export interface Draft {
  type: string;
  name: string;
  kinds: Kinds;
  /** The pinned CLI account (cli types). */
  account: string | null;
  /** Non-secret field values. */
  config: Record<string, string>;
  /** Secret field values; leave the browser only in the create/reconnect call. */
  secrets: Record<string, string>;
  qrId: string | null;
  qrState: QrState | null;
  /** QR types: the user chose the fields instead. */
  manual: boolean;
  /** E-mail: the server was filled from the address (the user has not typed one). */
  autoServer?: boolean;
}

/** The connect list: my accounts, then my bots (the reader's likely ones first), then the website check. */
const ACCOUNT_ORDER = ['github', 'gitlab', 'jira', 'email', 'calendar'];
const BOT_ORDER: Record<Locale, string[]> = {
  zh: ['feishu', 'dingtalk', 'wecom', 'telegram', 'slack', 'discord'],
  en: ['telegram', 'slack', 'discord', 'feishu', 'dingtalk', 'wecom'],
};

function ordered(types: CatalogType[], order: string[]): CatalogType[] {
  const rank = (t: CatalogType) => { const i = order.indexOf(t.type); return i < 0 ? order.length : i; };
  return [...types].sort((a, b) => rank(a) - rank(b));
}

/**
 * Only what this daemon can run: a type this platform cannot run (calendar
 * off a Mac) or this build has no adapter for is left out, not shown off.
 */
export function connectGroups(catalog: Catalog | null, locale: Locale) {
  const types = (catalog?.types ?? []).filter((t) => t.available);
  return {
    account: ordered(types.filter((t) => t.family === 'account'), ACCOUNT_ORDER),
    bot: ordered(types.filter((t) => t.family === 'bot'), BOT_ORDER[locale] ?? BOT_ORDER.en),
    check: types.filter((t) => t.family === 'check'),
  };
}

export const formFields = (t: CatalogType): CatalogField[] => t.connect.fields ?? [];

/** Fields as rows: a `pair` field shares its row with the next pair field. */
export function fieldRows(fields: CatalogField[]): Array<CatalogField | [CatalogField, CatalogField]> {
  const rows: Array<CatalogField | [CatalogField, CatalogField]> = [];
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i];
    const n = fields[i + 1];
    if (f.pair && n?.pair) { rows.push([f, n]); i++; } else rows.push(f);
  }
  return rows;
}

function kindDefaults(t: CatalogType, key: string): Record<string, unknown> {
  const k = t.kinds.find((x) => x.key === key);
  const out: Record<string, unknown> = {};
  for (const p of k?.params ?? []) if (p.default !== undefined && p.default !== null) out[p.key] = p.default;
  return out;
}

export function initDraft(t: CatalogType): Draft {
  const kinds: Kinds = {};
  for (const k of t.kinds) if (k.default) kinds[k.key] = kindDefaults(t, k.key);
  if (!Object.keys(kinds).length && t.kinds[0]) kinds[t.kinds[0].key] = kindDefaults(t, t.kinds[0].key);
  const config: Record<string, string> = {};
  for (const f of formFields(t)) if (!f.secret && f.default) config[f.key] = f.default;
  return { type: t.type, name: '', kinds, account: null, config, secrets: {}, qrId: null, qrState: null, manual: false };
}

/** A kind on or off; the last kind on stays on (a source brings in something). */
export function toggleKind(t: CatalogType, kinds: Kinds, key: string): Kinds {
  if (kinds[key]) {
    if (Object.keys(kinds).length <= 1) return kinds;
    const next = { ...kinds };
    delete next[key];
    return next;
  }
  return { ...kinds, [key]: kindDefaults(t, key) };
}

/** A param of a kind; the params that depend on it are cleared when it changes. */
export function setParam(t: CatalogType, kinds: Kinds, kind: string, param: string, value: unknown): Kinds {
  const cur = kinds[kind] ?? {};
  if (cur[param] === value) return kinds;
  const next: Record<string, unknown> = { ...cur, [param]: value };
  const k = t.kinds.find((x) => x.key === kind);
  for (const p of k?.params ?? []) if (p.depends_on === param) delete next[p.key];
  return { ...kinds, [kind]: next };
}

const filled = (v: unknown) => v !== undefined && v !== null && String(v).trim() !== '';

export function kindsReady(t: CatalogType, kinds: Kinds): boolean {
  const on = Object.keys(kinds);
  if (!on.length) return false;
  return on.every((key) => {
    const k = t.kinds.find((x) => x.key === key);
    return (k?.params ?? []).every((p) => !p.required || filled(kinds[key][p.key]));
  });
}

/** E-mail providers whose IMAP server is known (the prototype's list). */
const MAIL_HOSTS: Record<string, string> = {
  'gmail.com': 'imap.gmail.com', 'googlemail.com': 'imap.gmail.com', 'qq.com': 'imap.qq.com', 'foxmail.com': 'imap.qq.com',
  '163.com': 'imap.163.com', '126.com': 'imap.126.com', 'icloud.com': 'imap.mail.me.com', 'me.com': 'imap.mail.me.com',
  'outlook.com': 'outlook.office365.com', 'hotmail.com': 'outlook.office365.com',
};

export function setField(t: CatalogType, d: Draft, key: string, value: string): Draft {
  const f = formFields(t).find((x) => x.key === key);
  if (f?.secret) return { ...d, secrets: { ...d.secrets, [key]: value } };
  const config = { ...d.config, [key]: value };
  let autoServer = d.autoServer;
  if (t.type === 'email' && key === 'acct') {
    const host = MAIL_HOSTS[value.split('@')[1]?.trim().toLowerCase() ?? ''];
    if (host && (!filled(d.config.server) || d.autoServer)) { config.server = host; autoServer = true; }
  }
  if (t.type === 'email' && key === 'server') autoServer = false;
  return { ...d, config, autoServer };
}

export function usesQr(t: CatalogType, d: Pick<Draft, 'manual'>): boolean {
  return t.connect.method === 'qr' && !d.manual;
}

/** The host of a website address as typed (`shop.acme.com`, `https://…/x`). */
export function websiteHost(v: string | undefined | null): string {
  const s = String(v ?? '').trim().replace(/^[a-z][a-z0-9+.-]*:\/\//i, '');
  return s.split(/[/?#]/)[0].trim();
}

export function fieldsReady(t: CatalogType, d: Draft): boolean {
  return formFields(t).every((f) => f.optional || filled(f.secret ? d.secrets[f.key] : d.config[f.key]));
}

export function draftReady(t: CatalogType, d: Draft): boolean {
  if (!kindsReady(t, d.kinds)) return false;
  switch (t.connect.method) {
    case 'cli_account': return filled(d.account);
    case 'qr': return usesQr(t, d) ? d.qrState === 'done' && !!d.qrId : fieldsReady(t, d);
    case 'fields':
      if (t.family === 'check' && !websiteHost(d.config.url)) return false;
      return fieldsReady(t, d);
    default: return true;
  }
}

/** The values a create (or reconnect) sends: only what was filled. */
export function credentials(t: CatalogType, d: Draft): { config: Record<string, unknown>; secrets: Record<string, string> } {
  const config: Record<string, unknown> = {};
  const secrets: Record<string, string> = {};
  if (!usesQr(t, d)) {
    for (const f of formFields(t)) {
      const v = (f.secret ? d.secrets[f.key] : d.config[f.key]) ?? '';
      if (!filled(v)) continue;
      if (f.secret) secrets[f.key] = v.trim();
      else config[f.key] = v.trim();
    }
  }
  if (t.family === 'check' && typeof config.url === 'string' && !/^[a-z][a-z0-9+.-]*:\/\//i.test(config.url)) {
    config.url = `https://${config.url}`;
  }
  return { config, secrets };
}

export function createBody(t: CatalogType, d: Draft): CreateBody {
  const { config, secrets } = credentials(t, d);
  return {
    type: t.type,
    name: d.name.trim(),
    kinds: d.kinds,
    account: t.connect.method === 'cli_account' ? d.account : null,
    config,
    secrets,
    qr_id: usesQr(t, d) ? d.qrId : null,
    instruction: '',
  };
}

/**
 * Which control a refusal marks: a form field, the CLI account, or a kind's
 * param (repo, label, folder); null when it names none (the line then sits
 * by the button).
 */
export function markedField(t: CatalogType, field: string | null): string | null {
  if (!field) return null;
  if (formFields(t).some((f) => f.key === field)) return field;
  if (field === 'account' && t.connect.method === 'cli_account') return field;
  if (t.kinds.some((k) => k.params.some((p) => p.key === field))) return field;
  return null;
}

/** The catalogue key of a verify failure's line under the field (or the form). */
export function verifyMessageKey(type: string, reason: string | null, secretField: boolean) {
  switch (reason) {
    case 'rejected': return type === 'email' ? 'sources.verify.login' : secretField ? 'sources.verify.token' : 'sources.verify.rejected';
    case 'not_found': return 'sources.verify.notFound';
    case 'signed_out': return 'sources.verify.signedOut';
    case 'cli_missing': return 'sources.verify.cliMissing';
    case 'permission': return 'sources.verify.permission';
    case 'unreachable': return 'sources.verify.unreachable';
    case 'invalid': return 'sources.verify.invalid';
    default: return 'sources.verify.error';
  }
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/** Task sources — one function per route of the backend contract. */

import { api, ApiError } from '../../config';
import type {
  AccountsResponse, Catalog, ChoiceItem, HistoryItem, QrStart, QrStatus, SourceDetail, SourceError,
  SourcesList,
} from './types';

const BASE = '/api/v2/sources';
const enc = encodeURIComponent;
const json = (method: string, body?: unknown): RequestInit => ({
  method,
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const fetchCatalog = () => api<Catalog>(`${BASE}/catalog`);
export const fetchSources = () => api<SourcesList>(BASE);
export const fetchSource = (id: string, before?: string) =>
  api<SourceDetail>(`${BASE}/${enc(id)}${before ? `?before=${enc(before)}` : ''}`);

export interface CreateBody {
  type: string;
  name: string;
  kinds: Record<string, Record<string, unknown>>;
  account: string | null;
  config: Record<string, unknown>;
  secrets: Record<string, string>;
  qr_id: string | null;
  instruction: string;
}

export const createSource = (body: CreateBody) => api<SourceDetail>(BASE, json('POST', body));

export type SourcePatch = Partial<{
  name: string;
  on: boolean;
  kinds: Record<string, Record<string, unknown>>;
  account: string;
  instruction: string;
  config: Record<string, unknown>;
}>;

export const patchSource = (id: string, patch: SourcePatch) =>
  api<SourceDetail>(`${BASE}/${enc(id)}`, json('PATCH', patch));
export const checkSource = (id: string) => api<SourceDetail>(`${BASE}/${enc(id)}/check`, json('POST'));
export const reconnectSource = (
  id: string, body: { secrets: Record<string, string>; config: Record<string, unknown>; qr_id: string | null },
) => api<SourceDetail>(`${BASE}/${enc(id)}/reconnect`, json('POST', body));
export const catchUp = (id: string) => api<SourceDetail>(`${BASE}/${enc(id)}/catch-up`, json('POST'));
export const dismissHeld = (id: string) => api<SourceDetail>(`${BASE}/${enc(id)}/dismiss-held`, json('POST'));
export const deleteSource = (id: string) => api<{ status: string; id: string }>(`${BASE}/${enc(id)}`, json('DELETE'));
export const routeArrival = (id: string, arrivalId: string, projectId: string) =>
  api<HistoryItem>(`${BASE}/${enc(id)}/arrivals/${enc(arrivalId)}/route`, json('POST', { project_id: projectId }));
/** Show a check command's script in Finder / File Explorer. */
export const revealSource = (id: string) => api<{ path?: string }>(`${BASE}/${enc(id)}/reveal`, json('POST'));
/** macOS: System Settings › Privacy & Security › Calendars (a failure's
 *  open_settings, and Calendar access refused while connecting). */
export const openCalendarSettings = () => api<unknown>(`${BASE}/open-settings`, json('POST'));

/** The page cannot open System Settings itself reliably; the daemon does, else try the URL. */
export async function openSettingsPane(): Promise<void> {
  try { await openCalendarSettings(); } catch { window.location.href = CALENDAR_SETTINGS_URL; }
}
export const CALENDAR_SETTINGS_URL = 'x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars';

export const fetchAccounts = (type: string) => api<AccountsResponse>(`${BASE}/accounts?type=${enc(type)}`);
export const startQr = (type: string) => api<QrStart>(`${BASE}/qr`, json('POST', { type }));
export const fetchQr = (qrId: string) => api<QrStatus>(`${BASE}/qr/${enc(qrId)}`);

export interface ChoicesBody {
  type?: string;
  source_id?: string;
  list: string;
  account?: string | null;
  params: Record<string, unknown>;
  config?: Record<string, unknown>;
  secrets?: Record<string, string>;
}
export const fetchChoices = (body: ChoicesBody) =>
  api<{ items: ChoiceItem[] }>(`${BASE}/choices`, json('POST', body));

/** The daemon's error detail as an object (`api` hands it over as JSON text). */
export function sourceError(e: unknown): SourceError {
  const out: SourceError = { code: 'error', field: null, reason: null, message: null };
  if (!(e instanceof ApiError)) return out;
  let d: unknown = e.detail;
  try { d = JSON.parse(e.detail); } catch { /* a plain string */ }
  if (d && typeof d === 'object') {
    const o = d as Record<string, unknown>;
    out.code = typeof o.code === 'string' ? o.code : out.code;
    out.field = typeof o.field === 'string' ? o.field : null;
    out.reason = typeof o.reason === 'string' ? o.reason : null;
    out.message = typeof o.message === 'string' ? o.message : null;
  } else if (typeof d === 'string') {
    out.message = d;
  }
  return out;
}

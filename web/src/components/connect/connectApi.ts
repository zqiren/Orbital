// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The daemon calls the connect panel makes (CONTRACT.md, batch 1123). Each
 * one degrades to the older routes when the daemon predates the new ones, so
 * the panel still works against a daemon without /providers/catalog or
 * /providers/check.
 */
import { api, ApiError } from '../../config';
import { codedErrorOf } from '../../utils/providerError';
import type {
  CardMutationResponse,
  CatalogProvider,
  ChatgptSigninResponse,
  LLMSdk,
  ProviderCheckResult,
  ProviderRegistry,
  TokendanceSigninResponse,
} from '../../types';
import { catalogFromRegistry, CHATGPT, CUSTOM, type Region } from './providerMeta';

let catalogCache: CatalogProvider[] | null = null;
let catalogInFlight: Promise<CatalogProvider[]> | null = null;

/** Test seam: forget the cached catalog. */
export function resetCatalogCache() {
  catalogCache = null;
  catalogInFlight = null;
}

export function cachedCatalog(): CatalogProvider[] | null {
  return catalogCache;
}

async function loadCatalog(): Promise<CatalogProvider[]> {
  let list: CatalogProvider[];
  try {
    const res = await api<{ providers: CatalogProvider[] }>('/api/v2/providers/catalog');
    list = res?.providers ?? [];
  } catch (err) {
    if (!(err instanceof ApiError) || (err.status !== 404 && err.status !== 405)) throw err;
    const reg = await api<ProviderRegistry>('/api/v2/providers');
    list = catalogFromRegistry(reg ?? {});
  }
  if (!list.some((p) => p.id === CUSTOM)) {
    list = [...list, {
      id: CUSTOM, display_name: 'Custom / Self-Hosted', auth: 'none', signin: null,
      key_entry: true, regions: [], default_region: 'global', console_url: null,
      supports_model_list: true, default_model: null, china_only: false, custom_endpoint: true,
    }];
  }
  return list;
}

export function fetchCatalog(): Promise<CatalogProvider[]> {
  if (catalogCache) return Promise.resolve(catalogCache);
  catalogInFlight = catalogInFlight ?? loadCatalog().then((list) => {
    catalogCache = list;
    return list;
  }).finally(() => {
    catalogInFlight = null;
  });
  return catalogInFlight;
}

export interface CheckInput {
  provider: string;
  region: Region;
  apiKey?: string;
  cardId?: string;
  baseUrl?: string;
  sdk?: LLMSdk;
  model?: string | null;
}

/** A failure the panel can show: short code + status, the sentence behind a sign. */
export interface Failure {
  code: string | null;
  status: number | null;
  message: string;
}

function checkBody(input: CheckInput): Record<string, unknown> {
  const body: Record<string, unknown> = { provider: input.provider, region: input.region, model: input.model ?? null };
  if (input.cardId) body.card_id = input.cardId;
  if (input.apiKey) body.api_key = input.apiKey;
  if (input.provider === CUSTOM) {
    body.base_url = input.baseUrl || null;
    body.sdk = input.sdk ?? 'openai';
  }
  return body;
}

function failureFrom(err: unknown): Failure {
  const coded = codedErrorOf(err);
  if (coded) return { code: coded.code, status: err instanceof ApiError ? err.status : null, message: coded.message };
  if (err instanceof ApiError) return { code: null, status: err.status || null, message: err.detail };
  return { code: null, status: null, message: err instanceof Error ? err.message : String(err) };
}

/**
 * Test provider + region + key; on a pass, the live model list and the model
 * to preselect. Always resolves: a failure is a verdict, not a throw.
 */
export async function checkConnection(
  input: CheckInput,
  catalog: CatalogProvider | undefined,
  signal?: AbortSignal,
): Promise<ProviderCheckResult> {
  try {
    return await api<ProviderCheckResult>('/api/v2/providers/check', {
      method: 'POST',
      body: JSON.stringify(checkBody(input)),
      signal,
    });
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 405) && !codedErrorOf(err)) {
      return legacyCheck(input, catalog, signal);
    }
    if (signal?.aborted) throw err;
    const f = failureFrom(err);
    return { ok: false, status: f.status, code: f.code, message: f.message, models: [], default_model: null };
  }
}

/** The pre-1123 daemon: list models first, then a completion test on one. */
async function legacyCheck(
  input: CheckInput,
  catalog: CatalogProvider | undefined,
  signal?: AbortSignal,
): Promise<ProviderCheckResult> {
  const base: Record<string, unknown> = { provider: input.provider };
  if (input.cardId) base.card_id = input.cardId;
  if (input.apiKey) base.api_key = input.apiKey;
  if (input.provider === CUSTOM) {
    base.base_url = input.baseUrl;
    base.sdk = input.sdk ?? 'openai';
  } else if (catalog?.sdk) {
    base.sdk = catalog.sdk;
  }
  let models: string[] = [];
  if (catalog?.supports_model_list !== false) {
    try {
      const res = await api<{ models?: string[] }>('/api/v2/providers/models', {
        method: 'POST', body: JSON.stringify(base), signal,
      });
      models = res?.models ?? [];
    } catch {
      models = [];
    }
  }
  const pick = input.model
    || (catalog?.default_model && (models.length === 0 || models.includes(catalog.default_model))
      ? catalog.default_model
      : models[0])
    || catalog?.default_model
    || '';
  try {
    await api('/api/v2/providers/test', {
      method: 'POST', body: JSON.stringify({ ...base, model: pick }), signal,
    });
    const list = models.length ? models : (pick ? [pick] : []);
    return { ok: true, status: 200, code: null, message: '', models: list, default_model: pick || null };
  } catch (err) {
    if (signal?.aborted) throw err;
    const f = failureFrom(err);
    return { ok: false, status: f.status, code: f.code, message: f.message, models: [], default_model: null };
  }
}

export interface SaveInput {
  cardId?: string | null;
  provider: string;
  region: Region;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  sdk?: LLMSdk;
  /** On an edit, only these fields are sent. */
  fields?: ('provider' | 'region' | 'model' | 'api_key' | 'base_url' | 'sdk')[];
}

/** Create (POST) or update (PUT) a card. Throws a Failure on refusal. */
export async function saveCard(input: SaveInput): Promise<CardMutationResponse> {
  const all: Record<string, unknown> = {
    provider: input.provider,
    region: input.region,
    model: input.model,
  };
  if (input.apiKey) all.api_key = input.apiKey;
  if (input.provider === CUSTOM) {
    all.base_url = input.baseUrl ?? '';
    all.sdk = input.sdk ?? 'openai';
  }
  let body = all;
  if (input.cardId && input.fields) {
    body = Object.fromEntries(Object.entries(all).filter(([k]) => input.fields!.includes(k as never)));
  }
  try {
    return input.cardId
      ? await api<CardMutationResponse>(`/api/v2/settings/cards/${encodeURIComponent(input.cardId)}`, {
          method: 'PUT', body: JSON.stringify(body),
        })
      : await api<CardMutationResponse>('/api/v2/settings/cards', {
          method: 'POST', body: JSON.stringify(body),
        });
  } catch (err) {
    throw failureFrom(err);
  }
}

export type SigninKind = 'chatgpt' | 'tokendance';

export interface SigninOutcome {
  card: CardMutationResponse['card'];
  test: CardMutationResponse['test'];
  email: string | null;
}

/** The blocking one-click sign-in. Throws a Failure (code `signin_cancelled` on cancel). */
export async function signIn(kind: SigninKind, cardId: string | null, signal: AbortSignal): Promise<SigninOutcome> {
  try {
    const res = await api<ChatgptSigninResponse | TokendanceSigninResponse>(
      `/api/v2/providers/${kind}/signin`,
      { method: 'POST', body: JSON.stringify(cardId ? { card_id: cardId } : {}), signal },
    );
    if (!res?.card) throw new Error('no card');
    const email = ('account' in res && res.account?.email) || res.card.account?.email || null;
    return { card: res.card, test: res.test ?? null, email };
  } catch (err) {
    if (signal.aborted) throw { code: 'signin_cancelled', status: 499, message: '' } satisfies Failure;
    throw failureFrom(err);
  }
}

/** Best-effort: tell the daemon to stop waiting. A 404 (older daemon) is fine. */
export async function cancelSignIn(kind: SigninKind): Promise<void> {
  try {
    await api(`/api/v2/providers/${kind}/signin/cancel`, { method: 'POST' });
  } catch {
    /* nothing was waiting, or the route does not exist */
  }
}

/** Open the browser again on the pending sign-in. False when the daemon cannot. */
export async function reopenSignIn(kind: SigninKind): Promise<boolean> {
  try {
    // The daemon opens the system browser itself: a window.open from the
    // app's webview would not reach the user's browser.
    const res = await api<{ reopened?: boolean }>(
      `/api/v2/providers/${kind}/signin/reopen`, { method: 'POST' },
    );
    return !!res?.reopened;
  } catch {
    return false;
  }
}

/** Sign-out of a signed-in card = delete it (the daemon revokes first). */
export async function deleteCardById(cardId: string): Promise<void> {
  try {
    await api(`/api/v2/settings/cards/${encodeURIComponent(cardId)}`, { method: 'DELETE' });
  } catch (err) {
    throw failureFrom(err);
  }
}

export function isSigninProvider(p: CatalogProvider | undefined): boolean {
  return !!p && (p.auth === 'signin' || p.id === CHATGPT);
}

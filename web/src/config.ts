// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

declare global {
  interface Window {
    __AGENT_OS_LOCAL__?: boolean;
  }
}

/**
 * True ONLY when served by the cloud relay (not daemon).
 * - Daemon serves index.html with __AGENT_OS_LOCAL__=true (both localhost and LAN).
 * - Relay serves index.html without this flag.
 * - Vite dev server: VITE_LOCAL_MODE env var as fallback (defaults to true).
 */
export const isRelayMode: boolean = (() => {
  // Daemon-injected flag is authoritative
  if (window.__AGENT_OS_LOCAL__) return false;
  // Vite dev server fallback (no flag injected since daemon isn't serving)
  if (import.meta.env.VITE_LOCAL_MODE === 'true') return false;
  // Localhost is always local
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1' || host === '') return false;
  // No flag + non-localhost = relay
  return true;
})();

export const BASE_URL: string =
  import.meta.env.VITE_API_URL || window.location.origin;

export const WS_URL: string =
  BASE_URL.replace(/^http/, 'ws') + '/ws';

function getAuthToken(): string | null {
  try {
    return localStorage.getItem('relay_jwt');
  } catch {
    return null;
  }
}

export async function api<T = unknown>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const url = isRelayMode
    ? window.location.origin + path
    : BASE_URL + path;
  const headers: Record<string, string> = {
    ...(options?.headers as Record<string, string>),
  };

  if (isRelayMode) {
    const token = getAuthToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
  }

  if (
    options?.body &&
    typeof options.body === 'string' &&
    !headers['Content-Type']
  ) {
    headers['Content-Type'] = 'application/json';
  }

  // Cache-bust in remote mode to prevent Safari ETag/304 hangs
  let finalUrl = url;
  if (isRelayMode) {
    const sep = finalUrl.includes('?') ? '&' : '?';
    finalUrl += `${sep}_t=${Date.now()}`;
  }

  const fetchOptions: RequestInit = { ...options, headers };
  if (isRelayMode) {
    fetchOptions.cache = 'no-store';
  }

  const response = await fetch(finalUrl, fetchOptions);

  if (!response.ok) {
    const body = await response.text();
    let detail: string;
    try {
      const parsed = JSON.parse(body);
      detail = parsed.detail
        ? typeof parsed.detail === 'string'
          ? parsed.detail
          : JSON.stringify(parsed.detail)
        : body;
    } catch {
      detail = body;
    }
    throw new ApiError(response.status, detail);
  }

  const text = await response.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    console.error('[api] JSON parse failed:', e, 'Response length:', text.length, 'First 200 chars:', text.slice(0, 200));
    throw new ApiError(0, `Invalid JSON response (${text.length} bytes)`);
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = 'ApiError';
  }
}

/** Fetch JSON with X-Total-Count header (for paginated endpoints). */
export async function apiWithTotal<T = unknown>(
  path: string,
  options?: { signal?: AbortSignal },
): Promise<{ data: T; total: number }> {
  const url = isRelayMode ? window.location.origin + path : BASE_URL + path;
  const headers: Record<string, string> = {};
  if (isRelayMode) {
    const token = getAuthToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  let finalUrl = url;
  if (isRelayMode) {
    const sep = finalUrl.includes('?') ? '&' : '?';
    finalUrl += `${sep}_t=${Date.now()}`;
  }
  const fetchOpts: RequestInit = { headers, signal: options?.signal };
  if (isRelayMode) fetchOpts.cache = 'no-store';

  const response = await fetch(finalUrl, fetchOpts);
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, body);
  }
  const total = parseInt(response.headers.get('X-Total-Count') ?? '0', 10);
  const text = await response.text();
  if (!text) return { data: [] as unknown as T, total };
  try {
    return { data: JSON.parse(text) as T, total };
  } catch (e) {
    console.error('[apiWithTotal] JSON parse failed:', e, 'len:', text.length);
    throw new ApiError(0, `Invalid JSON response (${text.length} bytes)`);
  }
}

/**
 * Fetch raw bytes (spec 090 document preview) under the same URL/auth rules
 * as `api`. The relay tunnel re-serialises response bodies as text, which
 * destroys binary data, so the daemon answers a relayed request with a base64
 * JSON envelope instead; it is decoded here so callers always get bytes.
 */
export async function apiBytes(
  path: string,
  options?: { signal?: AbortSignal },
): Promise<ArrayBuffer> {
  const url = isRelayMode ? window.location.origin + path : BASE_URL + path;
  const headers: Record<string, string> = {};
  if (isRelayMode) {
    const token = getAuthToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  let finalUrl = url;
  if (isRelayMode) {
    const sep = finalUrl.includes('?') ? '&' : '?';
    finalUrl += `${sep}_t=${Date.now()}`;
  }
  const fetchOpts: RequestInit = { headers, signal: options?.signal };
  if (isRelayMode) fetchOpts.cache = 'no-store';

  const response = await fetch(finalUrl, fetchOpts);
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === 'string') detail = parsed.detail;
    } catch {
      // not JSON — keep the raw body
    }
    throw new ApiError(response.status, detail);
  }
  if ((response.headers.get('content-type') ?? '').includes('application/json')) {
    const envelope = (await response.json()) as { encoding?: string; content?: string };
    if (envelope.encoding !== 'base64' || typeof envelope.content !== 'string') {
      throw new ApiError(0, 'Unexpected preview response');
    }
    const binary = atob(envelope.content);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }
  return response.arrayBuffer();
}

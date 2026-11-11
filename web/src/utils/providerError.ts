// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Classified LLM provider/credential errors (credential-error surfacing).
 *
 * The daemon returns HTTP 400 with `detail: {code, message}` from start paths
 * (cold-start scan, /agents/start, inject auto-start) and broadcasts the same
 * `error_code` on agent.status error events. Codes are a stable contract with
 * `agent_os/daemon_v2/provider_errors.py`.
 */
import { ApiError } from '../config';
import type { StringKey } from '../i18n/strings';

export interface ProviderErrorInfo {
  code: string;
  message: string;
}

const KEY_BY_CODE: Record<string, StringKey> = {
  missing_api_key: 'agentError.missing_api_key',
  // Batch 1110 item 3: no key AND no enabled agent in this project.
  no_runner: 'agentError.no_runner',
  // Spec 082: the project's credential card is gone and there is no default.
  missing_card: 'agentError.missing_card',
  invalid_api_key: 'agentError.invalid_api_key',
  model_not_found: 'agentError.model_not_found',
  provider_unreachable: 'agentError.provider_unreachable',
  provider_error: 'agentError.provider_error',
  // Spec 141: an Orbital-side crash, not a provider failure.
  internal_error: 'agentError.internal_error',
};

/** Codes that are Orbital's own failure: the fix is not in Settings, so the
 * notice offers Send feedback instead of Open settings (spec 141). */
export function isInternalError(code: string | undefined): boolean {
  return code === 'internal_error';
}

/** i18n key for a classifier code; unknown codes fall back to the generic. */
export function providerErrorKey(code: string | undefined): StringKey {
  return (code && KEY_BY_CODE[code]) || 'agentError.provider_error';
}

/**
 * Extract a structured {code, message} from an ApiError whose detail is the
 * JSON-stringified `{code, message}` dict FastAPI serialized. Returns null
 * for anything else (plain-string details, network failures, …).
 */
export function parseProviderError(err: unknown): ProviderErrorInfo | null {
  if (!(err instanceof ApiError) || typeof err.detail !== 'string') return null;
  return codedFromText(err.detail);
}

/**
 * The same `{code, message}` read off any thrown Error's text (an ApiError's
 * message IS its detail). For surfaces whose routes always answer with a code
 * — the credential-card routes — so a refusal is never shown as raw JSON.
 */
export function codedErrorOf(err: unknown): ProviderErrorInfo | null {
  return err instanceof Error ? codedFromText(err.message) : null;
}

/**
 * The provider's own sentence from an error detail. SDK errors arrive as
 * ``Error code: 400 - {'error': {'message': '…'}}`` (a Python repr); this
 * returns the innermost ``message`` the way the daemon's
 * ``unwrap_provider_message`` does, and '' when the envelope holds no
 * readable message (the payload itself is never shown). Plain sentences pass
 * through unchanged.
 */
export function readableProviderDetail(raw: string): string {
  const m = /^\s*Error code:\s*\d+\s+-\s+([[{][\s\S]*)$/.exec(raw);
  const body = m ? m[1] : /^\s*[[{]/.test(raw) ? raw : null;
  if (body === null) return raw;
  let parsed: unknown;
  try {
    parsed = JSON.parse(pythonReprToJson(body.trim()));
  } catch {
    return '';
  }
  for (let i = 0; i < 4; i++) {
    if (typeof parsed === 'string') return parsed;
    if (parsed === null || typeof parsed !== 'object') break;
    const p = parsed as { message?: unknown; error?: unknown };
    if (typeof p.message === 'string' && p.message) return p.message;
    if (p.error === undefined) break;
    parsed = p.error;
  }
  return '';
}

/** A Python literal's repr as JSON: quoted strings re-quoted, True/False/None
 *  mapped. Anything else is left for JSON.parse to reject. */
function pythonReprToJson(src: string): string {
  let out = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "'" || ch === '"') {
      let s = '';
      i++;
      while (i < src.length && src[i] !== ch) {
        if (src[i] === '\\' && i + 1 < src.length) {
          const e = src[i + 1];
          s += e === 'n' ? '\n' : e === 't' ? '\t' : e;
          i += 2;
        } else {
          s += src[i++];
        }
      }
      i++;
      out += JSON.stringify(s);
      continue;
    }
    const word = /^(True|False|None)\b/.exec(src.slice(i));
    if (word) {
      out += word[1] === 'True' ? 'true' : word[1] === 'False' ? 'false' : 'null';
      i += word[1].length;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

function codedFromText(text: string): ProviderErrorInfo | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      typeof (parsed as { code?: unknown }).code === 'string'
    ) {
      const p = parsed as { code: string; message?: unknown };
      return {
        code: p.code,
        message: typeof p.message === 'string' ? p.message : '',
      };
    }
  } catch {
    // not structured JSON — not a coded error
  }
  return null;
}

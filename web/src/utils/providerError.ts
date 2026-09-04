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
  // Spec 082: the project's credential card is gone and there is no default.
  missing_card: 'agentError.missing_card',
  invalid_api_key: 'agentError.invalid_api_key',
  model_not_found: 'agentError.model_not_found',
  provider_unreachable: 'agentError.provider_unreachable',
  provider_error: 'agentError.provider_error',
};

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
  try {
    const parsed: unknown = JSON.parse(err.detail);
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
    // detail was not structured JSON — not a classified provider error
  }
  return null;
}

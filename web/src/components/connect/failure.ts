// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { StringKey } from '../../i18n/strings';
import type { Failure } from './connectApi';

const FAIL_KEY: Record<string, StringKey> = {
  invalid_api_key: 'connect.fail.invalid_api_key',
  insufficient_credits: 'connect.fail.insufficient_credits',
  model_not_found: 'connect.fail.model_not_found',
  rate_limited: 'connect.fail.rate_limited',
  provider_unreachable: 'connect.fail.provider_unreachable',
  provider_error: 'connect.fail.provider_error',
  missing_model: 'connect.fail.missing_model',
  needs_signin: 'connect.fail.needs_signin',
  signin_required: 'connect.fail.needs_signin',
  not_eligible: 'connect.fail.not_eligible',
  usage_limit_reached: 'connect.fail.usage_limit_reached',
  card_not_found: 'connect.fail.card_not_found',
  signin_denied: 'connect.fail.signin_denied',
  signin_timeout: 'connect.fail.signin_timeout',
  signin_in_progress: 'connect.fail.signin_in_progress',
  signin_failed: 'connect.fail.signin_failed',
};

/** The short reason: what kind of failure, and its status when there is one. */
export function shortReason(f: Failure, t: (k: StringKey) => string): string {
  const key = (f.code && FAIL_KEY[f.code]) || (f.status === 401 || f.status === 403 ? FAIL_KEY.invalid_api_key : null);
  const what = key ? t(key) : t('connect.fail.generic');
  return f.status && f.status >= 400 && f.status !== 499 ? `${what} · ${f.status}` : what;
}


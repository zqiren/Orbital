// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { readableProviderDetail } from './providerError';

// Mirrors the daemon's unwrap_provider_message (agent_os/agent/providers/types.py).
describe('readableProviderDetail — the provider sentence, never the SDK envelope (1111 pass 2, D24)', () => {
  it('pulls the message out of a nested error envelope (Python repr)', () => {
    expect(readableProviderDetail(
      "Error code: 400 - {'error': {'type': 'server_error', 'message': 'Upstream request failed: Model is unavailable.'}}",
    )).toBe('Upstream request failed: Model is unavailable.');
    expect(readableProviderDetail(
      "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}",
    )).toBe('invalid x-api-key');
  });

  it('reads double-quoted reprs, escapes and JSON bodies', () => {
    expect(readableProviderDetail(`Error code: 403 - {'error': {'message': "The key isn't allowed in this region"}}`))
      .toBe("The key isn't allowed in this region");
    expect(readableProviderDetail(`Error code: 401 - {'error': {'message': 'it\\'s bad'}}`)).toBe("it's bad");
    expect(readableProviderDetail('Error code: 401 - {"error": {"message": "Unauthorized", "code": 401}}')).toBe('Unauthorized');
  });

  it('an envelope without a readable message shows nothing rather than the payload', () => {
    expect(readableProviderDetail("Error code: 401 - {'error': {...}}")).toBe('');
    expect(readableProviderDetail("Error code: 401 - {'error': {'code': 'unauthorized'}}")).toBe('');
  });

  it('leaves plain sentences alone', () => {
    expect(readableProviderDetail('No LLM API key configured')).toBe('No LLM API key configured');
    expect(readableProviderDetail('Upstream request failed: Model is unavailable.')).toBe('Upstream request failed: Model is unavailable.');
    expect(readableProviderDetail('')).toBe('');
  });
});

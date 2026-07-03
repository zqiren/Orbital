// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { isWorkerHandle } from './subAgentHandle';

describe('isWorkerHandle', () => {
  it('recognizes a fanout worker handle', () => {
    expect(isWorkerHandle('worker:abcd1234-0')).toBe(true);
    expect(isWorkerHandle('worker:abcd1234-11')).toBe(true);
  });

  it('rejects persistent sub-agent handles', () => {
    expect(isWorkerHandle('claude-code')).toBe(false);
    expect(isWorkerHandle('codex')).toBe(false);
    expect(isWorkerHandle('gemini-cli')).toBe(false);
  });

  it('rejects null/undefined/empty without throwing', () => {
    expect(isWorkerHandle(null)).toBe(false);
    expect(isWorkerHandle(undefined)).toBe(false);
    expect(isWorkerHandle('')).toBe(false);
  });
});

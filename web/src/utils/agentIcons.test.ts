// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { getAgentIcon } from './agentIcons';

describe('getAgentIcon', () => {
  it('resolves claude-code → CC / Anthropic coral', () => {
    const i = getAgentIcon('claude-code');
    expect(i.monogram).toBe('CC');
    expect(i.color).toBe('#D97757');
  });

  it('resolves codex → CX / OpenAI green', () => {
    const i = getAgentIcon('codex');
    expect(i.monogram).toBe('CX');
    expect(i.color).toBe('#10A37F');
  });

  it('resolves gemini-cli → GM / Google blue', () => {
    const i = getAgentIcon('gemini-cli');
    expect(i.monogram).toBe('GM');
    expect(i.color).toBe('#4285F4');
  });

  it('falls back to a 2-char monogram for unknown handles', () => {
    expect(getAgentIcon('unknown-agent').monogram).toBe('UN');
    expect(getAgentIcon('').monogram).toBe('?');
  });

  it('is case-insensitive on the handle', () => {
    expect(getAgentIcon('Claude-Code').monogram).toBe('CC');
  });
});

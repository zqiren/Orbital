// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { getAgentIcon, MAIN_AGENT_HANDLE } from './agentIcons';

describe('getAgentIcon', () => {
  it('gives every known slug a distinct mark and monogram', () => {
    const known = [MAIN_AGENT_HANDLE, 'claude-code', 'codex', 'cursor', 'gemini', 'grok'];
    const icons = known.map((slug) => getAgentIcon(slug));

    for (const icon of icons) {
      expect(icon.src).toBeTruthy();
      expect(icon.monogram).toHaveLength(2);
    }
    // The whole point of the fix: no two agents look alike.
    expect(new Set(icons.map((i) => i.src)).size).toBe(known.length);
    expect(new Set(icons.map((i) => i.monogram)).size).toBe(known.length);
  });

  it('resolves claude-code → CC / Anthropic coral', () => {
    const i = getAgentIcon('claude-code');
    expect(i.monogram).toBe('CC');
    expect(i.color).toBe('#D97757');
    expect(i.src).toBe('/agents/claude-code.svg');
  });

  it('resolves codex → CX / OpenAI green', () => {
    const i = getAgentIcon('codex');
    expect(i.monogram).toBe('CX');
    expect(i.color).toBe('#10A37F');
    expect(i.src).toBe('/agents/codex.svg');
  });

  it('resolves the new cursor and grok slugs', () => {
    expect(getAgentIcon('cursor').src).toBe('/agents/cursor.svg');
    expect(getAgentIcon('grok').src).toBe('/agents/grok.svg');
  });

  it('resolves the management agent to the Orbital app icon', () => {
    expect(getAgentIcon(MAIN_AGENT_HANDLE).src).toBe('/icon-192.png');
  });

  it('matches suffixed variants on their base slug', () => {
    // "gemini-cli" is not a key; it falls back to the "gemini" prefix.
    const i = getAgentIcon('gemini-cli');
    expect(i.monogram).toBe('GM');
    expect(i.color).toBe('#4285F4');
    expect(getAgentIcon('codex-cli').monogram).toBe('CX');
  });

  it('matches exactly, not by substring', () => {
    // The old implementation used handle.includes('claude') and would have
    // wrongly branded both of these as Claude.
    expect(getAgentIcon('notclaude').monogram).toBe('NO');
    expect(getAgentIcon('notclaude').src).toBeUndefined();
    expect(getAgentIcon('my-claude-helper').monogram).toBe('MY');
  });

  it('falls back to a grey 2-char monogram with no image for unknown handles', () => {
    const i = getAgentIcon('unknown-agent');
    expect(i.monogram).toBe('UN');
    expect(i.color).toBe('#6B7280');
    expect(i.src).toBeUndefined();
    expect(getAgentIcon('').monogram).toBe('?');
  });

  it('normalizes case and surrounding whitespace', () => {
    expect(getAgentIcon('Claude-Code').monogram).toBe('CC');
    expect(getAgentIcon('  CODEX  ').src).toBe('/agents/codex.svg');
  });
});

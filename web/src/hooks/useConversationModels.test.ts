// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * 1125 — the pin menu's model picks, with and without a session.
 *
 * A fresh install lands in Quick Tasks with no session (no route session
 * id). The picks then read and write the project's own (item 158), which the
 * first send's new session starts on; inside a session they stay that
 * conversation's.
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import { __resetConversationModels, useConversationModels } from './useConversationModels';

const ENTRIES = [{
  slug: 'claude-code',
  config: {},
  param_schema: {
    model: { allowed: ['sonnet', 'opus'], labels: { sonnet: 'Sonnet', opus: 'Opus' }, default: 'opus' },
    effort: { allowed: ['low', 'high'] },
  },
}];

let projectParams: Record<string, { model?: string; effort?: string }> = {};

beforeEach(() => {
  projectParams = {};
  apiMock.mockReset();
  apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
    if (path === '/api/v2/settings/sub-agents') return ENTRIES;
    if (path === '/api/v2/settings') return { credential_cards: [], default_card_id: null };
    if (path === '/api/v2/agents/p1/sub-agent-params') {
      if (init?.method === 'PATCH') {
        const { slug, ...patch } = JSON.parse(init.body ?? '{}');
        const next = { ...(projectParams[slug] ?? {}) } as Record<string, string>;
        for (const [k, v] of Object.entries(patch)) {
          if (v) next[k] = v as string; else delete next[k];
        }
        projectParams = { ...projectParams, [slug]: next };
        return { slug, params: next };
      }
      return { params: projectParams };
    }
    if (path.startsWith('/api/v2/agents/p1/sessions/s1/sub-agent-params')) {
      if (init?.method === 'PATCH') return { params: { model: 'sonnet' } };
      return { params: {} };
    }
    throw new Error(`unexpected ${path}`);
  });
});

afterEach(() => __resetConversationModels());

describe('useConversationModels', () => {
  it('with no session, reads and writes the project pick', async () => {
    projectParams = { 'claude-code': { effort: 'high' } };
    const { result } = renderHook(() => useConversationModels('p1', undefined));
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.workers['claude-code']?.effort).toBe('high'));
    expect(result.current.workers['claude-code'].settingsModel).toBe('opus');

    let ok = false;
    await act(async () => {
      ok = await result.current.setWorkerParams('claude-code', { model: 'sonnet' });
    });
    expect(ok).toBe(true);
    expect(apiMock).toHaveBeenCalledWith('/api/v2/agents/p1/sub-agent-params', {
      method: 'PATCH', body: JSON.stringify({ slug: 'claude-code', model: 'sonnet' }),
    });
    expect(result.current.workers['claude-code']).toMatchObject({ model: 'sonnet', effort: 'high' });
  });

  it('inside a session, the conversation route as before', async () => {
    const { result } = renderHook(() => useConversationModels('p1', 's1'));
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.workers['claude-code']).toBeTruthy());
    await act(async () => {
      await result.current.setWorkerParams('claude-code', { model: 'sonnet' });
    });
    expect(apiMock).toHaveBeenCalledWith('/api/v2/agents/p1/sessions/s1/sub-agent-params', {
      method: 'PATCH', body: JSON.stringify({ slug: 'claude-code', model: 'sonnet' }),
    });
    expect(apiMock).not.toHaveBeenCalledWith('/api/v2/agents/p1/sub-agent-params', expect.anything());
    expect(result.current.workers['claude-code'].model).toBe('sonnet');
  });
});

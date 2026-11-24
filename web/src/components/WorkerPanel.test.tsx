// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 195: the docked panel shows one dispatched agent's steps.

import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock, apiWithTotal: vi.fn(), BASE_URL: '', isRelayMode: false }));
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({ on: vi.fn(), off: vi.fn(), connectionState: 'connected' }),
}));

import { WorkerPanelBody, WorkerPanelHeader } from './WorkerPanel';
import { __resetDispatchCardsStore, publishDispatchCards } from '../utils/dispatchCardsStore';
import type { DispatchCardItem } from '../utils/dispatchCards';

const TS = '2026-11-24T10:00:00Z';

function card(extra: Partial<DispatchCardItem> = {}): DispatchCardItem {
  return {
    type: 'dispatch_card',
    card_id: 'd1',
    dispatch_id: 'd1',
    handle: 'codex',
    label: '汇总小红书和大众点评上关于三家品牌的口碑关键词',
    status: 'running',
    started_at: Date.parse(TS),
    ended_at: null,
    timestamp: TS,
    steps: [
      { type: 'agent_message', content: '', source: 'codex', timestamp: TS, isHeaderOnly: true },
      {
        type: 'agent_run', capsule_id: 'k1', status: 'running', source: 'codex',
        items: [{
          type: 'tool_call_row', tool_name: 'Read', target_description: 'buzz.csv', tool_call_id: 't1',
          category: 'file_read', timestamp: TS, result_content: null, result_status: 'pending',
        }],
        tool_call_count_by_name: { Read: 1 }, has_thinking: false, started_at: 0, ended_at: null,
      },
      { type: 'sub_agent_message', content: '两份导出共 12,384 条。', source: 'codex', timestamp: TS },
    ],
    ...extra,
  };
}

const target = { kind: 'card' as const, cardId: 'd1', dispatchId: 'd1', handle: 'codex' };
const agents = [{ slug: 'codex', name: 'Codex' }];

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockResolvedValue({ session_id: 's1', agents: [] });
  __resetDispatchCardsStore();
});

describe('WorkerPanel', () => {
  it('shows the card\'s title, its live steps, and follows the card as it grows', async () => {
    publishDispatchCards('p1:s1', [card()]);
    render(<>
      <WorkerPanelHeader projectId="p1" sessionId="s1" target={target} agents={agents} />
      <WorkerPanelBody projectId="p1" sessionId="s1" target={target} agents={agents} />
    </>);
    expect(screen.getByTestId('worker-panel-header').textContent).toContain('汇总小红书');
    expect(screen.getByTestId('worker-panel-body').textContent).toContain('buzz.csv');
    expect(screen.getByTestId('worker-panel-body').textContent).toContain('12,384');

    act(() => {
      publishDispatchCards('p1:s1', [card({ steps: [...card().steps, { type: 'sub_agent_message', content: 'new step', source: 'codex', timestamp: TS }] })]);
    });
    expect(screen.getByTestId('worker-panel-body').textContent).toContain('new step');
  });

  it('Stop is in the header while it runs, asks first, then stops that one agent', async () => {
    publishDispatchCards('p1:s1', [card()]);
    render(<WorkerPanelHeader projectId="p1" sessionId="s1" target={target} agents={agents} />);
    await userEvent.click(screen.getByTestId('worker-panel-stop'));
    expect(screen.getByTestId('sub-agent-stop-confirm')).toBeTruthy();
    await userEvent.click(screen.getByTestId('sub-agent-stop-confirm-button'));
    expect(apiMock).toHaveBeenCalledWith('/api/v2/agents/p1/sub-agents/codex/stop?session_id=s1', { method: 'POST' });
  });

  it('no Stop once the agent finished', () => {
    publishDispatchCards('p1:s1', [card({ status: 'completed', ended_at: Date.parse(TS) + 5000 })]);
    render(<WorkerPanelHeader projectId="p1" sessionId="s1" target={target} agents={agents} />);
    expect(screen.queryByTestId('worker-panel-stop')).toBeNull();
  });

  it('a live card replaced by its persisted one is still found (by dispatch id)', () => {
    publishDispatchCards('p1:s1', [card({ card_id: 'd1' })]);
    render(<WorkerPanelBody projectId="p1" sessionId="s1"
      target={{ kind: 'card', cardId: 'live:codex:1', dispatchId: 'd1', handle: 'codex' }} agents={agents} />);
    expect(screen.getByTestId('worker-panel-body').getAttribute('data-card-id')).toBe('d1');
  });
});

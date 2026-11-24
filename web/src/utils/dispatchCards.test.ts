// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import type { DisplayItem } from './chatTransform';
import {
  cardLabel,
  endCard,
  ensureLiveCard,
  findCardIndex,
  isCardedInitiator,
  latestBriefFor,
  parseDispatchMarker,
  splitTrailingChildCards,
  unwrapCard,
  updateCardSteps,
  type DispatchCardItem,
} from './dispatchCards';

const TS = '2026-11-24T10:00:00Z';
const TS2 = '2026-11-24T10:02:41Z';

const card = (items: DisplayItem[], i: number) => items[i] as DispatchCardItem;

describe('isCardedInitiator', () => {
  it('cards Orbital and a worker; keeps the user\'s own sends inline', () => {
    expect(isCardedInitiator('management_agent')).toBe(true);
    expect(isCardedInitiator('worker:claude-code')).toBe(true);
    expect(isCardedInitiator('user_pinned')).toBe(false);
    expect(isCardedInitiator('queue_item')).toBe(false);
  });
  it('rows without an initiator: no suppress_wake → card, suppress_wake → inline', () => {
    expect(isCardedInitiator(undefined, false)).toBe(true);
    expect(isCardedInitiator(undefined, true)).toBe(false);
  });
});

describe('parseDispatchMarker / cardLabel', () => {
  it('reads the handle and the quoted preview, without the transcript tail', () => {
    expect(parseDispatchMarker('[Sub-agent] Message sent to codex: "修复 PDF\n第二行". Transcript: /t.jsonl'))
      .toEqual({ handle: 'codex', preview: '修复 PDF\n第二行' });
  });
  it('the title is the first non-empty line', () => {
    expect(cardLabel('\n  统计三家品牌在华东的门店数  \nsecond')).toBe('统计三家品牌在华东的门店数');
  });
});

describe('live cards', () => {
  const managerCapsule: DisplayItem = {
    type: 'agent_run',
    capsule_id: 'c1',
    status: 'running',
    items: [{
      type: 'tool_call_row', tool_name: 'agent_message', target_description: 'codex', tool_call_id: 't1',
      category: 'agent_message', timestamp: TS, result_content: null, result_status: 'pending',
      dispatch: { agent: 'codex', message: '汇总小红书口碑关键词\n要求：…' },
    }],
    tool_call_count_by_name: { agent_message: 1 },
    has_thinking: false,
    started_at: 0,
    ended_at: null,
  };

  it('a new card takes its title from the brief Orbital just sent', () => {
    expect(latestBriefFor([managerCapsule], 'codex')).toBe('汇总小红书口碑关键词\n要求：…');
    const { items, idx } = ensureLiveCard([managerCapsule], { handle: 'codex', initiator: 'management_agent', timestamp: TS });
    // The send row was the capsule's only row: the card replaces it.
    expect(idx).toBe(0);
    expect(items).toHaveLength(1);
    expect(card(items, 0)).toMatchObject({ handle: 'codex', label: '汇总小红书口碑关键词', status: 'running' });
  });

  it('the card takes its send row out of a capsule that keeps other rows', () => {
    const withRead: DisplayItem = {
      ...managerCapsule,
      items: [
        { type: 'tool_call_row', tool_name: 'read', target_description: 'a.md', tool_call_id: 'r0', category: 'file_read',
          timestamp: TS, result_content: '', result_status: 'received' },
        ...(managerCapsule as Extract<DisplayItem, { type: 'agent_run' }>).items,
      ],
      tool_call_count_by_name: { read: 1, agent_message: 1 },
    };
    const { items, idx } = ensureLiveCard([withRead], { handle: 'codex', initiator: 'management_agent', timestamp: TS });
    expect(idx).toBe(1);
    const cap = items[0] as Extract<DisplayItem, { type: 'agent_run' }>;
    expect(cap.items.map((c) => c.type === 'tool_call_row' && c.tool_name)).toEqual(['read']);
    expect(cap.tool_call_count_by_name).toEqual({ read: 1 });
  });

  it('the dispatch id arriving later is adopted, not a second card', () => {
    const first = ensureLiveCard([], { handle: 'codex', timestamp: TS });
    const second = ensureLiveCard(first.items, { handle: 'codex', dispatchId: 'd1', initiator: 'management_agent', brief: 'go', timestamp: TS });
    expect(second.items).toHaveLength(1);
    expect(card(second.items, 0)).toMatchObject({ dispatch_id: 'd1', initiator: 'management_agent', label: 'go' });
  });

  it('a new dispatch id for a handle whose card never ended closes that card', () => {
    const a = ensureLiveCard([], { handle: 'codex', dispatchId: 'd1', timestamp: TS });
    const b = ensureLiveCard(a.items, { handle: 'codex', dispatchId: 'd2', timestamp: TS2 });
    expect(b.items).toHaveLength(2);
    expect(card(b.items, 0).status).toBe('completed');
    expect(card(b.items, 1).status).toBe('running');
    expect(findCardIndex(b.items, { handle: 'codex' })).toBe(1);
  });

  it('a worker dispatch is folded', () => {
    const { items } = ensureLiveCard([], { handle: 'codex', initiator: 'worker:claude-code', timestamp: TS });
    expect(card(items, 0).folded).toBe(true);
  });

  it('end: status, end time, and the reply right under an Orbital card', () => {
    let { items } = ensureLiveCard([], { handle: 'codex', dispatchId: 'd1', initiator: 'management_agent', timestamp: TS });
    items = [...items, { type: 'user_message', content: 'later', timestamp: TS }];
    items = updateCardSteps(items, 0, (s) => [...s, { type: 'sub_agent_message', content: '好了', source: 'codex', timestamp: TS }]);
    const ended = endCard(items, 0, 'completed', TS2, (s) => s);
    expect(card(ended, 0)).toMatchObject({ status: 'completed', ended_at: Date.parse(TS2) });
    expect(ended[1]).toMatchObject({ type: 'agent_message', source: 'codex', content: '好了' });
    expect(ended[2].type).toBe('user_message');
    // Idempotent: ending again does not add a second bubble.
    expect(endCard(ended, 0, 'completed', TS2, (s) => s)).toHaveLength(3);
  });

  it('end of a folded card adds no bubble', () => {
    let { items } = ensureLiveCard([], { handle: 'codex', initiator: 'worker:claude-code', timestamp: TS });
    items = updateCardSteps(items, 0, (s) => [...s, { type: 'sub_agent_message', content: 'green', source: 'codex', timestamp: TS }]);
    expect(endCard(items, 0, 'completed', TS2, (s) => s)).toHaveLength(1);
  });

  it('unwrap puts the steps back inline where the card stood', () => {
    let { items } = ensureLiveCard([{ type: 'user_message', content: 'u', timestamp: TS }], { handle: 'codex', timestamp: TS });
    items = updateCardSteps(items, 1, (s) => [...s, { type: 'sub_agent_message', content: 'x', source: 'codex', timestamp: TS }]);
    expect(unwrapCard(items, 1).map((i) => i.type)).toEqual(['user_message', 'sub_agent_message']);
  });

  it('a worker\'s own trailing child cards are held so its events go before them', () => {
    const { items } = ensureLiveCard([{ type: 'user_message', content: 'u', timestamp: TS }], {
      handle: 'codex', initiator: 'worker:claude-code', timestamp: TS,
    });
    const split = splitTrailingChildCards(items, 'claude-code');
    expect(split.head).toHaveLength(1);
    expect(split.tail).toHaveLength(1);
    expect(splitTrailingChildCards(items, 'pi').tail).toHaveLength(0);
  });

  it('a child that ended is not held: the owner\'s next turn (its result) goes after it', () => {
    const { items } = ensureLiveCard([], { handle: 'codex', initiator: 'worker:claude-code', timestamp: TS });
    const ended = endCard(items, 0, 'completed', TS2, (s) => s);
    expect(splitTrailingChildCards(ended, 'claude-code').tail).toHaveLength(0);
  });
});

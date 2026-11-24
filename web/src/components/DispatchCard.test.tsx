// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import DispatchCard from './DispatchCard';
import { cardDot } from '../utils/dispatchCards';
import type { DispatchCardItem } from '../utils/dispatchCards';

const base: DispatchCardItem = {
  type: 'dispatch_card', card_id: 'd1', handle: 'claude-code', label: '整理三家品牌的饮品定价对比表',
  status: 'completed', started_at: 0, ended_at: 161_000, steps: [], timestamp: '',
};

describe('DispatchCard', () => {
  it('the dot carries the state; no status word', () => {
    expect(cardDot('running', false)).toBe('running');
    expect(cardDot('running', true)).toBe('needs-you');
    expect(cardDot('error', false)).toBe('error');
    expect(cardDot('stopped', false)).toBe('stopped');
    expect(cardDot('completed', true)).toBe('none');
  });

  it('a finished card shows its frozen time, its brief, the agent only as the tooltip', () => {
    render(<DispatchCard card={base} name="Claude Code" />);
    expect(screen.getByTestId('dispatch-card-time').textContent).toBe('2m 41s');
    const row = screen.getByTestId('dispatch-card-row');
    expect(row.textContent).toContain('整理三家品牌的饮品定价对比表');
    expect(row.textContent).not.toContain('Claude Code');
    expect(row.getAttribute('title')).toBe('Claude Code');
  });

  it('without a brief the agent name is the title', () => {
    render(<DispatchCard card={{ ...base, label: '' }} name="Codex" />);
    expect(screen.getByTestId('dispatch-card-row').textContent).toContain('Codex');
  });
});

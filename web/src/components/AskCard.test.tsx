// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import AskCard from './AskCard';
import type { SessionAsk } from '../hooks/useSessionAsks';
import { formatAskAnswer, parseAskAnswer } from '../utils/askAnswer';

const ASK: SessionAsk = { id: 'aaaaaa', text: 'Keep the 9,800 creator?', due: '2026-11-03', options: [], detail: '' };
const CHOICE: SessionAsk = { id: 'bbbbbb', text: 'Which venue?', due: null, options: ['Rooftop', 'Hall', 'Garden'], detail: '' };

function setup(ask: SessionAsk = ASK) {
  const props = { onOption: vi.fn(), onDone: vi.fn(), onNotNeeded: vi.fn(), onLater: vi.fn() };
  render(<AskCard ask={ask} {...props} />);
  return props;
}

describe('AskCard (item 160, Claude Code style)', () => {
  it('shows the question and its due date', () => {
    setup();
    expect(screen.getByTestId('ask-card-text').textContent).toContain('Keep the 9,800 creator?');
    expect(screen.getByTestId('ask-card').textContent).toContain('by 2026-11-03');
  });

  it('offers the options as buttons, and a click answers with that option', () => {
    const p = setup(CHOICE);
    const pills = screen.getAllByTestId('ask-card-option');
    expect(pills.map((b) => b.textContent)).toEqual(['Rooftop', 'Hall', 'Garden']);
    fireEvent.click(screen.getByRole('button', { name: 'Hall' }));
    expect(p.onOption).toHaveBeenCalledWith('Hall');
  });

  it('an ask without options shows none', () => {
    setup();
    expect(screen.queryAllByTestId('ask-card-option')).toHaveLength(0);
  });

  it('has no Reply button and no pager: the composer under it is the answer field', () => {
    setup(CHOICE);
    expect(screen.queryByRole('button', { name: 'Reply' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Next' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Previous' })).toBeNull();
  });

  it('closes the ask on screen', () => {
    const p = setup(CHOICE);
    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(p.onDone).toHaveBeenCalledWith('bbbbbb');
    fireEvent.click(screen.getByRole('button', { name: 'Not needed' }));
    expect(p.onNotNeeded).toHaveBeenCalledWith('bbbbbb');
  });

  it('Later is the way out', () => {
    const p = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Later' }));
    expect(p.onLater).toHaveBeenCalled();
  });
});

describe('AskCard detail (markdown under the question)', () => {
  const DETAILED: SessionAsk = {
    id: 'cccccc', text: 'Which account first?', due: '2026-11-19', options: ['Chase', 'Amex'],
    detail: 'The **Chase** statement covers `Jul–Sep`.\n\nStarting with Chase unblocks the $1,280 question.',
  };

  it('renders the detail as markdown under the question, in its own scroll box', () => {
    render(<AskCard ask={DETAILED} onOption={vi.fn()} onDone={vi.fn()} onNotNeeded={vi.fn()} onLater={vi.fn()} />);
    expect(screen.getByTestId('ask-card-text').textContent).toBe('Which account first?');
    const detail = screen.getByTestId('ask-card-detail');
    expect(detail.querySelector('strong')?.textContent).toBe('Chase');
    expect(detail.querySelector('code')?.textContent).toBe('Jul–Sep');
    expect(detail.querySelectorAll('p')).toHaveLength(2);
    expect(detail.className).toContain('overflow-y-auto');
    expect(screen.getByTestId('ask-card-text').className).not.toContain('line-clamp');
  });

  it('an ask without detail has no detail box', () => {
    render(<AskCard ask={ASK} onOption={vi.fn()} onDone={vi.fn()} onNotNeeded={vi.fn()} onLater={vi.fn()} />);
    expect(screen.queryByTestId('ask-card-detail')).toBeNull();
  });
});

describe('askAnswer', () => {
  it('round-trips, even when the question has quotes in it', () => {
    const wire = formatAskAnswer('abc123', 'Use "A" or "B"?', 'A\nfor now');
    expect(parseAskAnswer(wire)).toEqual({ question: 'Use "A" or "B"?', rest: 'A\nfor now' });
  });

  it('leaves an ordinary message alone', () => {
    expect(parseAskAnswer('hello')).toEqual({ question: null, rest: 'hello' });
  });
});

describe('AskCard — the question renders as inline markdown', () => {
  it('bold and code are elements; no raw marks and no nested paragraph', () => {
    setup({ id: 'cccccc', text: '看**这里** 和 `x`', due: null, options: [], detail: '' });
    const q = screen.getByTestId('ask-card-text');
    expect(q.querySelector('strong')).toHaveTextContent('这里');
    expect(q.querySelector('code')).toHaveTextContent('x');
    expect(q.querySelector('p')).toBeNull();
    expect(q.textContent).toBe('看这里 和 x');
  });
});

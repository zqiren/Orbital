// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import AskTray from './AskTray';
import { formatAskAnswer, parseAskAnswer } from '../utils/askAnswer';

const ASKS = [
  { id: 'aaaaaa', text: 'Keep the 9,800 creator?', due: '2026-11-03' },
  { id: 'bbbbbb', text: 'Send the letter', due: null },
];

function setup(index = 0) {
  const props = { onIndex: vi.fn(), onDone: vi.fn(), onNotNeeded: vi.fn(), onLater: vi.fn() };
  render(<AskTray asks={ASKS} index={index} {...props} />);
  return props;
}

describe('AskTray', () => {
  it('shows one ask and pages to the next', () => {
    const p = setup();
    expect(screen.getByTestId('ask-tray-text').textContent).toBe('Keep the 9,800 creator?');
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(p.onIndex).toHaveBeenCalledWith(1);
  });

  it('closes the ask on screen, not another one', () => {
    const p = setup(1);
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

describe('askAnswer', () => {
  it('round-trips, even when the question has quotes in it', () => {
    const wire = formatAskAnswer('abc123', 'Use "A" or "B"?', 'A\nfor now');
    expect(parseAskAnswer(wire)).toEqual({ question: 'Use "A" or "B"?', rest: 'A\nfor now' });
  });

  it('leaves an ordinary message alone', () => {
    expect(parseAskAnswer('hello')).toEqual({ question: null, rest: 'hello' });
  });
});

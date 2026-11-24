// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/** Spec 197 — the usage-limit row: the value, the vendor's sentence behind
 *  hover/tap, and one logo button that switches the conversation. */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import UsageLimitRow from './UsageLimitRow';

const NOW = new Date('2026-11-24T14:00:00');
const at = (h: number, m = 0) => new Date(2026, 10, 24, h, m).toISOString();

describe('UsageLimitRow', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  const base = {
    handle: 'codex', name: 'Codex', resetsAt: at(17, 23),
    vendorText: "You've hit your usage limit … or try again at 5:23 PM.",
    switchTo: { slug: 'claude-code', name: 'Claude Code' }, onSwitch: () => {},
  };

  it('states the value only; the vendor sentence is behind hover', () => {
    render(<UsageLimitRow {...base} />);
    expect(screen.getByTestId('usage-limit-value').textContent).toBe('Usage limit · back at 17:23');
    expect(screen.queryByTestId('usage-limit-vendor')).toBeNull();
    fireEvent.pointerEnter(screen.getByTestId('usage-limit-value'), { pointerType: 'mouse' });
    expect(screen.getByTestId('usage-limit-vendor').textContent).toContain('try again at 5:23 PM');
  });

  it('a mouse click after the hover keeps the sentence open', () => {
    render(<UsageLimitRow {...base} />);
    const value = screen.getByTestId('usage-limit-value');
    fireEvent.pointerEnter(value, { pointerType: 'mouse' });
    fireEvent.pointerDown(value, { pointerType: 'mouse' });
    fireEvent.click(value);
    expect(screen.getByTestId('usage-limit-vendor')).toBeTruthy();
  });

  it('a tap toggles the vendor sentence (phones have no hover)', () => {
    render(<UsageLimitRow {...base} />);
    fireEvent.pointerDown(screen.getByTestId('usage-limit-value'), { pointerType: 'touch' });
    fireEvent.click(screen.getByTestId('usage-limit-value'));
    expect(screen.getByTestId('usage-limit-vendor')).toBeTruthy();
    fireEvent.pointerDown(screen.getByTestId('usage-limit-value'), { pointerType: 'touch' });
    fireEvent.click(screen.getByTestId('usage-limit-value'));
    expect(screen.queryByTestId('usage-limit-vendor')).toBeNull();
  });

  it('an unknown reset says the limit without a time', () => {
    render(<UsageLimitRow {...base} resetsAt={null} />);
    expect(screen.getByTestId('usage-limit-value').textContent).toBe('Usage limit');
  });

  it('the one button is the next agent, logo only, and switches on click', () => {
    const onSwitch = vi.fn();
    render(<UsageLimitRow {...base} onSwitch={onSwitch} />);
    const btn = screen.getByTestId('usage-limit-switch');
    expect(btn.getAttribute('aria-label')).toBe('Switch to Claude Code');
    expect(btn.textContent).toBe('');
    fireEvent.click(btn);
    expect(onSwitch).toHaveBeenCalledWith('claude-code');
  });

  it('no button when there is nowhere to switch', () => {
    render(<UsageLimitRow {...base} switchTo={null} />);
    expect(screen.queryByTestId('usage-limit-switch')).toBeNull();
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import TokenCostSign from './TokenCostSign';
import { LocaleProvider } from '../i18n/LocaleContext';
import { STRINGS } from '../i18n/strings';

afterEach(() => {
  vi.useRealTimers();
  window.localStorage.removeItem('orbital.locale');
});

function hover(el: Element) {
  fireEvent.pointerEnter(el, { pointerType: 'mouse' });
}

describe('TokenCostSign (spec 160 §5.3)', () => {
  it('renders nothing with no unscanned import', () => {
    const { container } = render(<TokenCostSign n={0} />);
    expect(container.innerHTML).toBe('');
  });

  it('shows the count and no sentence until asked', () => {
    render(<TokenCostSign n={3} />);
    const sign = screen.getByTestId('token-cost-sign');
    expect(sign.textContent).toBe('3');
    expect(sign.getAttribute('aria-label')).toBe('What saving this key does');
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('opens on hover after a short dwell and closes on leave', () => {
    vi.useFakeTimers();
    render(<TokenCostSign n={3} />);
    const sign = screen.getByTestId('token-cost-sign');
    hover(sign);
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
    act(() => { vi.advanceTimersByTime(130); });
    const pop = screen.getByTestId('token-cost-pop');
    expect(pop.textContent).toBe(
      'Saving lets Orbital read your 3 imported projects once with this key to draft their goals and status. One-time; uses tokens.');
    fireEvent.pointerLeave(sign, { pointerType: 'mouse' });
    // Round 2: a longer grace so the pointer can cross onto the switch.
    act(() => { vi.advanceTimersByTime(150); });
    expect(screen.getByTestId('token-cost-pop')).toBeTruthy();
    act(() => { vi.advanceTimersByTime(100); });
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('a touch hover does nothing; a tap toggles', () => {
    render(<TokenCostSign n={1} />);
    const sign = screen.getByTestId('token-cost-sign');
    fireEvent.pointerEnter(sign, { pointerType: 'touch' });
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
    fireEvent.pointerDown(sign, { pointerType: 'touch' });
    fireEvent.click(sign);
    expect(screen.getByTestId('token-cost-pop').textContent).toBe(
      'Saving lets Orbital read your 1 imported project once with this key to draft its goals and status. One-time; uses tokens.');
    expect(sign.getAttribute('aria-expanded')).toBe('true');
    fireEvent.pointerDown(sign, { pointerType: 'touch' });
    fireEvent.click(sign);
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('a tap elsewhere and Escape close it', () => {
    render(<div><TokenCostSign n={2} /><p data-testid="away">x</p></div>);
    const sign = screen.getByTestId('token-cost-sign');
    fireEvent.click(sign);
    expect(screen.getByTestId('token-cost-pop')).toBeTruthy();
    fireEvent.click(screen.getByTestId('away'));
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
    fireEvent.click(sign);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('speaks the prototype\'s Chinese verbatim, and the bubble is width-bounded', () => {
    window.localStorage.setItem('orbital.locale', 'zh');
    render(<LocaleProvider><TokenCostSign n={6} /></LocaleProvider>);
    fireEvent.click(screen.getByTestId('token-cost-sign'));
    const pop = screen.getByTestId('token-cost-pop');
    expect(pop.textContent).toBe(
      '保存后，Orbital 会用这个 Key 把导入的 6 个项目各读一遍，理出目标和现状。仅此一次，会消耗 token。');
    // zh fits: the bubble wraps inside 280px (or the phone width), never one line.
    expect(pop.style.maxWidth).toBe('min(280px, calc(100vw - 32px))');
    expect(pop.className).toContain('w-max');
    expect(STRINGS['llm.tokenHint.one'].zh).toBe(STRINGS['llm.tokenHint.other'].zh);
  });
});


describe('TokenCostSign kinds (spec 160 round 2)', () => {
  it('skip names the agent that spends, and needs one', () => {
    window.localStorage.setItem('orbital.locale', 'zh');
    const { unmount } = render(<LocaleProvider><TokenCostSign n={3} kind="skip" agent="Claude Code" /></LocaleProvider>);
    const sign = screen.getByTestId('token-cost-sign');
    expect(sign.getAttribute('aria-label')).toBe('跳过会做什么');
    fireEvent.click(sign);
    expect(screen.getByTestId('token-cost-pop').textContent).toBe(
      '跳过后，Orbital 会用 Claude Code 把导入的 3 个项目各读一遍，理出目标和现状。仅此一次，会用掉 Claude Code 的额度。');
    unmount();
    const { container } = render(<TokenCostSign n={3} kind="skip" agent={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('import with a key and with the agent', () => {
    const { unmount } = render(<TokenCostSign n={2} kind="importKey" />);
    fireEvent.click(screen.getByTestId('token-cost-sign'));
    expect(screen.getByTestId('token-cost-pop').textContent).toBe(
      'Importing lets Orbital read these 2 projects once with your key to draft their goals and status. One-time; uses tokens.');
    unmount();
    render(<TokenCostSign n={1} kind="importAgent" agent="Codex" />);
    fireEvent.click(screen.getByTestId('token-cost-sign'));
    expect(screen.getByTestId('token-cost-pop').textContent).toBe(
      'Importing lets Codex read 1 project once to draft its goals and status. One-time; uses your Codex quota.');
  });
});


describe('TokenCostSign opt-out switch (spec 160 round 2)', () => {
  function Controlled({ initial = true }: { initial?: boolean }) {
    const [on, setOn] = useState(initial);
    return <TokenCostSign n={4} scan={on} onScanChange={setOn} />;
  }

  it('defaults on; off dims and slashes the glyph and hides the count; on restores it', () => {
    render(<Controlled />);
    const sign = screen.getByTestId('token-cost-sign');
    expect(sign.textContent).toBe('4');
    expect(screen.queryByTestId('token-cost-slash')).toBeNull();
    fireEvent.click(sign);
    const sw = screen.getByTestId('token-cost-switch');
    expect(sw.getAttribute('role')).toBe('switch');
    expect(sw.getAttribute('aria-checked')).toBe('true');
    expect(sw.textContent).toContain("Draft these projects' goals");
    fireEvent.click(sw);
    expect(screen.getByTestId('token-cost-switch').getAttribute('aria-checked')).toBe('false');
    expect(sign.textContent).toBe('');                       // no count, no words
    expect(screen.getByTestId('token-cost-slash')).toBeTruthy();
    // The bubble stayed open: a click inside it is not a click away.
    expect(screen.getByTestId('token-cost-pop')).toBeTruthy();
    fireEvent.click(screen.getByTestId('token-cost-switch'));
    expect(sign.textContent).toBe('4');
    expect(screen.queryByTestId('token-cost-slash')).toBeNull();
  });

  it('hover reaches the switch: glyph and bubble are one area', () => {
    vi.useFakeTimers();
    render(<Controlled />);
    const sign = screen.getByTestId('token-cost-sign');
    fireEvent.pointerEnter(sign, { pointerType: 'mouse' });
    act(() => { vi.advanceTimersByTime(130); });
    const pop = screen.getByTestId('token-cost-pop');
    fireEvent.pointerLeave(sign, { pointerType: 'mouse' });
    act(() => { vi.advanceTimersByTime(100); });
    fireEvent.pointerEnter(pop, { pointerType: 'mouse' });
    act(() => { vi.advanceTimersByTime(500); });
    expect(screen.getByTestId('token-cost-switch')).toBeTruthy();
    fireEvent.pointerLeave(pop, { pointerType: 'mouse' });
    act(() => { vi.advanceTimersByTime(300); });
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('keyboard: focus moves from the sign onto the switch without closing; Esc closes', () => {
    render(<Controlled />);
    const sign = screen.getByTestId('token-cost-sign');
    fireEvent.click(sign);
    const sw = screen.getByTestId('token-cost-switch');
    fireEvent.blur(sign, { relatedTarget: sw });
    sw.focus();
    expect(screen.getByTestId('token-cost-pop')).toBeTruthy();
    fireEvent.keyDown(sw, { key: ' ' });
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('token-cost-pop')).toBeNull();
  });

  it('no handler, no switch (the bubble stays a sentence)', () => {
    render(<TokenCostSign n={2} />);
    fireEvent.click(screen.getByTestId('token-cost-sign'));
    expect(screen.queryByTestId('token-cost-switch')).toBeNull();
  });
});

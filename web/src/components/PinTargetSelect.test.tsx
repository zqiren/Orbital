// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 074 — the composer "Talking to" pin control and the target-resolution
 * precedence rule.
 *
 * Covers the spec's Vitest list for this surface:
 *  - dropdown renders Orbital + every installed sub-agent, and renders
 *    NOTHING when no sub-agents are installed;
 *  - selection payloads: an agent → its slug, Orbital → null (the unpin);
 *  - target resolution (spec 091): the sticky pin, else management. A
 *    leading `@slug` is plain text — it never picks a target, and `@orbital`
 *    is no longer a one-message aside while pinned.
 */

import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import PinTargetSelect, { resolveSendTarget } from './PinTargetSelect';
import {
  __resetAgentAvailability,
  applyQuotaUpdate,
  setAgentAvailability,
} from '../hooks/useAgentAvailability';
import { LocaleProvider } from '../i18n/LocaleContext';
import type { AgentAvailability, QuotaSnapshot } from '../types';

beforeEach(() => {
  apiMock.mockReset();
  // Menu open re-fetches availability; by default the daemon is unreachable,
  // which must leave the loaded state alone.
  apiMock.mockRejectedValue(new Error('offline'));
});

afterEach(() => {
  cleanup();
  act(() => __resetAgentAvailability());
  vi.useRealTimers();
});

const AGENTS = [
  { slug: 'claude-code', name: 'Claude Code' },
  { slug: 'codex', name: 'Codex' },
];

describe('resolveSendTarget', () => {
  it('unpinned → management', () => {
    expect(resolveSendTarget('hello there', null)).toEqual({
      target: undefined, content: 'hello there', pinned: false,
    });
  });

  it('pinned → the pinned worker, pinned=true', () => {
    expect(resolveSendTarget('hello there', 'codex')).toEqual({
      target: 'codex', content: 'hello there', pinned: true,
    });
  });

  it('a leading @slug to Orbital is sent verbatim to Orbital (no target)', () => {
    expect(resolveSendTarget('@codex do the thing', null)).toEqual({
      target: undefined, content: '@codex do the thing', pinned: false,
    });
  });

  it('a leading @slug never overrides the pin — the pinned worker gets it verbatim', () => {
    expect(resolveSendTarget('@claude-code do the thing', 'codex')).toEqual({
      target: 'codex', content: '@claude-code do the thing', pinned: true,
    });
  });

  it('@orbital while pinned is plain text to the pinned worker (no aside)', () => {
    expect(resolveSendTarget('@orbital status update please', 'codex')).toEqual({
      target: 'codex', content: '@orbital status update please', pinned: true,
    });
  });
});

describe('PinTargetSelect', () => {
  /** The fused trigger button (logo mark + chevron). */
  const trigger = () =>
    screen.getByRole('button', { name: 'Choose who this chat talks to' });

  it('renders nothing when no sub-agents are installed', () => {
    const { container } = render(
      <PinTargetSelect agents={[]} value={null} onChange={() => {}} />,
    );
    expect(container.innerHTML).toBe('');
  });

  it('shows the Orbital mark at rest and the pinned agent mark while pinned', () => {
    const { rerender } = render(
      <PinTargetSelect agents={AGENTS} value={null} onChange={() => {}} />,
    );
    // No agentHandle → the avatar resolves to Orbital's own mark.
    const restAvatar = trigger().querySelector('[data-testid="message-avatar"]');
    expect(restAvatar?.getAttribute('data-agent-handle') ?? null).toBeNull();
    expect(trigger().title).toBe('Orbital — manager');

    rerender(<PinTargetSelect agents={AGENTS} value="codex" onChange={() => {}} />);
    const pinnedAvatar = trigger().querySelector('[data-testid="message-avatar"]');
    expect(pinnedAvatar?.getAttribute('data-agent-handle')).toBe('codex');
    expect(trigger().title).toBe('Codex — direct chat, Orbital stays out');
  });

  it('opens a menu listing Orbital plus every installed agent', () => {
    render(
      <PinTargetSelect agents={AGENTS} value={null} onChange={() => {}} />,
    );
    expect(screen.queryByRole('listbox')).toBeNull();
    fireEvent.click(trigger());
    const options = screen.getAllByRole('option');
    // Spec 099 mockup: "Manager" and a check column on every row. With no
    // availability loaded there is no usage text at all (unknown ≠ fine).
    expect(options.map((o) => o.textContent)).toEqual([
      'OrbitalManager✓', 'Claude Code', 'Codex',
    ]);
  });

  it('selecting an agent fires onChange with its slug and closes the menu', () => {
    const onChange = vi.fn();
    render(
      <PinTargetSelect agents={AGENTS} value={null} onChange={onChange} />,
    );
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole('option', { name: /Codex/ }));
    expect(onChange).toHaveBeenCalledWith('codex');
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('selecting Orbital fires onChange(null) — the unpin', () => {
    const onChange = vi.fn();
    render(
      <PinTargetSelect agents={AGENTS} value="codex" onChange={onChange} />,
    );
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole('option', { name: /Orbital/ }));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('a stale pin (agent no longer installed) still renders so it can be cleared', () => {
    render(
      <PinTargetSelect agents={AGENTS} value="gone-agent" onChange={() => {}} />,
    );
    const avatar = trigger().querySelector('[data-testid="message-avatar"]');
    expect(avatar?.getAttribute('data-agent-handle')).toBe('gone-agent');
    fireEvent.click(trigger());
    // The bare slug is appended as a selectable row.
    expect(screen.getByRole('option', { name: /gone-agent/ })).toBeTruthy();
  });

  it('Escape closes the menu without selecting', () => {
    const onChange = vi.fn();
    render(
      <PinTargetSelect agents={AGENTS} value={null} onChange={onChange} />,
    );
    fireEvent.click(trigger());
    expect(screen.getByRole('listbox')).toBeTruthy();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
  });
});


// ---------------------------------------------------------------------------
// Keyless onboarding (contract §4) — no key, so Orbital cannot run
// ---------------------------------------------------------------------------

describe('PinTargetSelect — managerReady=false', () => {
  const trigger = () =>
    screen.getByRole('button', { name: 'Choose who this chat talks to' });

  function captureOpenSettings() {
    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    window.addEventListener('open-global-settings', listener);
    return { events, stop: () => window.removeEventListener('open-global-settings', listener) };
  }

  it('keeps Orbital first, with an "API key required" pill instead of the role', () => {
    render(
      <PinTargetSelect agents={AGENTS} value="codex" onChange={() => {}} managerReady={false} />,
    );
    fireEvent.click(trigger());
    const options = screen.getAllByRole('option');
    expect(options[0].textContent).toContain('Orbital');
    expect(screen.getByTestId('pin-orbital-key-required').textContent).toBe('API key required');
    expect(options[0].textContent).not.toContain('Manager');
  });

  it('choosing Orbital opens Global Settings at the LLM section and does not unpin', () => {
    const onChange = vi.fn();
    const cap = captureOpenSettings();
    try {
      render(
        <PinTargetSelect agents={AGENTS} value="codex" onChange={onChange} managerReady={false} />,
      );
      fireEvent.click(trigger());
      fireEvent.click(screen.getByRole('option', { name: /Orbital/ }));
      expect(onChange).not.toHaveBeenCalled();
      expect(cap.events).toHaveLength(1);
      expect(cap.events[0].detail).toEqual({ section: 'llm' });
      expect(screen.queryByRole('listbox')).toBeNull();
    } finally {
      cap.stop();
    }
  });

  it('workers stay selectable', () => {
    const onChange = vi.fn();
    render(
      <PinTargetSelect agents={AGENTS} value="codex" onChange={onChange} managerReady={false} />,
    );
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole('option', { name: /Claude Code/ }));
    expect(onChange).toHaveBeenCalledWith('claude-code');
  });

  it('with a key (the default) Orbital is the ordinary unpin and shows no pill', () => {
    const onChange = vi.fn();
    const cap = captureOpenSettings();
    try {
      render(<PinTargetSelect agents={AGENTS} value="codex" onChange={onChange} />);
      fireEvent.click(trigger());
      expect(screen.queryByTestId('pin-orbital-key-required')).toBeNull();
      fireEvent.click(screen.getByRole('option', { name: /Orbital/ }));
      expect(onChange).toHaveBeenCalledWith(null);
      expect(cap.events).toHaveLength(0);
    } finally {
      cap.stop();
    }
  });

  it('zh pill copy', () => {
    window.localStorage.setItem('orbital.locale', 'zh');
    try {
      render(
        <LocaleProvider>
          <PinTargetSelect agents={AGENTS} value="codex" onChange={() => {}} managerReady={false} />
        </LocaleProvider>,
      );
      fireEvent.click(screen.getAllByRole('button')[0]);
      expect(screen.getByTestId('pin-orbital-key-required').textContent).toBe('需要 API 密钥');
    } finally {
      window.localStorage.removeItem('orbital.locale');
    }
  });
});

// ---------------------------------------------------------------------------
// Spec 099 — availability dot, usage line, hover card
// ---------------------------------------------------------------------------

describe('PinTargetSelect — usage + availability (spec 099)', () => {
  const NOW = new Date('2026-09-27T19:00:00');
  const inH = (h: number) => new Date(NOW.getTime() + h * 3600_000).toISOString();
  const ago = (m: number) => new Date(NOW.getTime() - m * 60_000).toISOString();

  const quota = (agent: string, five: number, week: number,
    extra: Partial<QuotaSnapshot> = {}): QuotaSnapshot => ({
    agent, observed_at: ago(4), plan: null, limited: false,
    windows: [
      { kind: 'five_hour', used_pct: 100 - five, resets_at: inH(3) },
      { kind: 'weekly', used_pct: 100 - week, resets_at: inH(24 * 2 + 22) },
    ],
    ...extra,
  });

  const FOUR: AgentAvailability[] = [
    { slug: 'claude-code', name: 'Claude Code', installed: true, ready: true,
      quota: quota('claude-code', 21, 92) },
    { slug: 'codex', name: 'Codex', installed: true, ready: true,
      quota: quota('codex', 82, 97, { plan: 'plus', observed_at: ago(0) }) },
    { slug: 'cursor', name: 'Cursor', installed: true, ready: true },
    { slug: 'pi', name: 'Pi', installed: true, ready: false,
      missing_credentials: ['pi_auth'] },
  ];
  const ROWS = FOUR.map(({ slug, name }) => ({ slug, name }));

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(NOW);
  });

  const trigger = () =>
    screen.getByRole('button', { name: 'Choose who this chat talks to' });
  const open = () => fireEvent.click(trigger());
  const line = (slug: string) => screen.getByTestId(`pin-quota-${slug}`);
  const dot = (slug: string) => screen.queryByTestId(`pin-dot-${slug}`);

  it('rows show what is left of the shown window, and no dot when healthy', () => {
    act(() => setAgentAvailability(FOUR));
    render(<PinTargetSelect agents={ROWS} value="claude-code" onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('5h · 21% left');
    expect(line('codex').textContent).toBe('5h · 82% left');
    expect(dot('claude-code')).toBeNull();
    expect(dot('codex')).toBeNull();
    // Cursor exposes no usage and its existing probe carries no plan tier.
    expect(screen.queryByTestId('pin-quota-cursor')).toBeNull();
    expect(dot('cursor')).toBeNull();
  });

  it('weekly replaces 5h when it is the tighter window', () => {
    act(() => setAgentAvailability([{ ...FOUR[0], quota: quota('claude-code', 64, 9) }]));
    render(<PinTargetSelect agents={ROWS.slice(0, 1)} value={null} onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('Week · 9% left');
    expect(line('claude-code').dataset.tone).toBe('low');
    expect(dot('claude-code')?.dataset.color).toBe('amber');
  });

  it('limit reached is red with the reset time', () => {
    act(() => setAgentAvailability([{ ...FOUR[0],
      quota: quota('claude-code', 0, 71, { limited: true }) }]));
    render(<PinTargetSelect agents={ROWS.slice(0, 1)} value={null} onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('Limit reached · 22:00');
    expect(line('claude-code').dataset.tone).toBe('out');
    expect(dot('claude-code')?.dataset.color).toBe('red');
  });

  it('any agent that hit its limit gets the red dot and the reset (spec 197)', () => {
    act(() => setAgentAvailability([
      { slug: 'cursor', name: 'Cursor', installed: true, ready: true,
        quota: { agent: 'cursor', observed_at: ago(1), plan: null, limited: true,
          windows: [], resets_at: inH(3) } },
      { slug: 'kimi-code', name: 'Kimi Code', installed: true, ready: true,
        quota: { agent: 'kimi-code', observed_at: ago(1), plan: null, limited: true,
          windows: [], resets_at: inH(-1) } },
    ]));
    render(<PinTargetSelect agents={[{ slug: 'cursor', name: 'Cursor' }, { slug: 'kimi-code', name: 'Kimi Code' }]}
      value="cursor" onChange={() => {}} />);
    open();
    expect(line('cursor').textContent).toBe('Limit reached · 22:00');
    expect(dot('cursor')?.dataset.color).toBe('red');
    // A passed reset is void: no mark at all.
    expect(screen.queryByTestId('pin-quota-kimi-code')).toBeNull();
    expect(dot('kimi-code')).toBeNull();
  });

  it('claude-code with no snapshot says so; a not-ready agent gets the grey dot', () => {
    act(() => setAgentAvailability([{ ...FOUR[0], quota: undefined }, FOUR[3]]));
    render(<PinTargetSelect agents={[ROWS[0], ROWS[3]]} value={null} onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('No reading yet');
    expect(line('pi').textContent).toBe('Needs login');
    expect(dot('pi')?.dataset.color).toBe('grey');
  });

  it('an installed-but-too-old agent reads "Needs update", not "Needs login" (spec 116)', () => {
    act(() => setAgentAvailability([{
      slug: 'claude-code', name: 'Claude Code', installed: true, ready: false,
      version: '2.1.100 (Claude Code)', update_needed: true, min_version: '2.1.114',
      // An old CLI reads signed-out too — the version branch must win.
      missing_credentials: ['claude_auth'],
    }]));
    render(<PinTargetSelect agents={ROWS.slice(0, 1)} value={null} onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('Needs update');
    expect(dot('claude-code')?.dataset.color).toBe('grey');
    fireEvent.mouseEnter(screen.getByRole('option', { name: /Claude Code/ }));
    const card = screen.getByTestId('pin-quota-card');
    expect(card.textContent).toContain(
      'Claude Code v2.1.100 is too old; Orbital needs 2.1.114 or newer.');
    expect(card.textContent).not.toContain('not signed in');
  });

  it('the running worker pulses green', () => {
    act(() => setAgentAvailability(FOUR));
    render(<PinTargetSelect agents={ROWS} value="codex" onChange={() => {}}
      runningSlugs={['codex']} />);
    open();
    expect(dot('codex')?.dataset.color).toBe('green');
    expect(dot('codex')?.dataset.pulse).toBe('true');
  });

  it('phone rows carry both windows, tighter first', () => {
    act(() => setAgentAvailability([{ ...FOUR[0], quota: quota('claude-code', 64, 9) }]));
    render(<PinTargetSelect agents={ROWS.slice(0, 1)} value={null} onChange={() => {}} />);
    open();
    const stack = screen.getByTestId('pin-quota-stack-claude-code');
    expect(Array.from(stack.children).map((c) => c.textContent))
      .toEqual(['Week · 9% left', '5h · 64% left']);
  });

  it('collapsed control: no dot when healthy, a corner dot when it needs attention', () => {
    act(() => setAgentAvailability(FOUR));
    const { rerender } = render(
      <PinTargetSelect agents={ROWS} value="claude-code" onChange={() => {}} />);
    expect(screen.queryByTestId('pin-control-dot')).toBeNull();
    act(() => applyQuotaUpdate('claude-code', quota('claude-code', 12, 88)));
    rerender(<PinTargetSelect agents={ROWS} value="claude-code" onChange={() => {}} />);
    expect(screen.getByTestId('pin-control-dot').dataset.color).toBe('amber');
    // Orbital (unpinned) never carries a dot.
    rerender(<PinTargetSelect agents={ROWS} value={null} onChange={() => {}} />);
    expect(screen.queryByTestId('pin-control-dot')).toBeNull();
  });

  it('the open menu shows the pinned worker\'s card; hovering a row switches it', () => {
    act(() => setAgentAvailability(FOUR));
    render(<PinTargetSelect agents={ROWS} value="claude-code" onChange={() => {}} />);
    open();
    const card = () => screen.getByTestId('pin-quota-card');
    expect(card().textContent).toContain('Claude Code usage');
    expect(card().textContent).toContain('5-hour window');
    expect(card().textContent).toContain('21% left · resets 22:00 (in 3h)');
    expect(card().textContent).toContain('Weekly');
    expect(card().textContent).toContain('Updated 4 min ago, from its last reply');

    fireEvent.mouseEnter(screen.getByRole('option', { name: /Codex/ }));
    expect(card().textContent).toContain('Codex usage');
    expect(card().textContent).toContain('Plus plan');
    expect(card().textContent).toContain('Updated just now');

    fireEvent.mouseEnter(screen.getByRole('option', { name: /Pi/ }));
    expect(card().textContent).toContain('Pi is installed but not signed in');

    fireEvent.mouseEnter(screen.getByRole('option', { name: /Orbital/ }));
    expect(screen.queryByTestId('pin-quota-card')).toBeNull();
  });

  it('hovering the collapsed control shows the pinned worker\'s card', () => {
    act(() => setAgentAvailability(FOUR));
    render(<PinTargetSelect agents={ROWS} value="codex" onChange={() => {}} />);
    expect(screen.queryByTestId('pin-quota-card')).toBeNull();
    fireEvent.mouseEnter(trigger());
    expect(screen.getByTestId('pin-quota-card').textContent).toContain('Codex usage');
    fireEvent.mouseLeave(trigger());
    expect(screen.queryByTestId('pin-quota-card')).toBeNull();
  });

  it('a window that reset since the reading shows no number', () => {
    act(() => setAgentAvailability([{ ...FOUR[0], quota: {
      ...quota('claude-code', 21, 92),
      windows: [
        { kind: 'five_hour', used_pct: 79, resets_at: inH(-1) },
        { kind: 'weekly', used_pct: 8, resets_at: inH(-0.5) },
      ] } }]));
    render(<PinTargetSelect agents={ROWS.slice(0, 1)} value="claude-code" onChange={() => {}} />);
    open();
    expect(line('claude-code').textContent).toBe('No current reading');
    expect(screen.getByTestId('pin-quota-card').textContent)
      .toContain('Reset since the last reading');
  });

  it('opening the menu refreshes availability, and a newer quota lands live', async () => {
    act(() => setAgentAvailability(FOUR));
    apiMock.mockResolvedValue(FOUR.map((a) => a.slug === 'codex'
      ? { ...a, quota: quota('codex', 40, 90, { plan: 'plus', observed_at: ago(0) }) } : a));
    render(<PinTargetSelect agents={ROWS} value="codex" onChange={() => {}} />);
    await act(async () => { open(); });
    expect(apiMock).toHaveBeenCalledWith('/api/v2/agents/available');
    expect(line('codex').textContent).toBe('5h · 40% left');
  });

  it('zh copy follows the mockup', () => {
    localStorage.setItem('orbital.locale', 'zh');
    try {
      act(() => setAgentAvailability([{ ...FOUR[0], quota: quota('claude-code', 64, 9) }]));
      render(
        <LocaleProvider>
          <PinTargetSelect agents={ROWS.slice(0, 1)} value={null} onChange={() => {}} />
        </LocaleProvider>,
      );
      fireEvent.click(screen.getByRole('button'));
      expect(line('claude-code').textContent).toBe('本周 · 剩余 9%');
    } finally {
      localStorage.removeItem('orbital.locale');
    }
  });
});

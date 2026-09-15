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

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';

import PinTargetSelect, { resolveSendTarget } from './PinTargetSelect';

afterEach(() => cleanup());

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
    expect(options.map((o) => o.textContent)).toEqual([
      'Orbitalmanager', 'Claude Code', 'Codex',
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

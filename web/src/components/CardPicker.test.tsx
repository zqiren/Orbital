// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * CardPicker unit tests — spec 082 §5 (frontend).
 *
 * Covers: every card is offered with its model, key suffix and health;
 * "Global default" resolves to `card_id: null` (a reference, never a copy of
 * the default's id); a card carrying `last_error` is flagged but still
 * selectable (spec 072 D2 — errors are shown, never acted on); a pinned id
 * with no matching card does not masquerade as "global default".
 */

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { CredentialCard } from '../types';
import CardPicker, { GLOBAL_DEFAULT_VALUE } from './CardPicker';

afterEach(() => cleanup());

function makeCard(overrides: Partial<CredentialCard> = {}): CredentialCard {
  return {
    id: 'card_aaa',
    name: 'DeepSeek · deepseek-chat',
    provider: 'deepseek',
    region: 'global',
    base_url: null,
    sdk: null,
    model: 'deepseek-chat',
    created_at: '2026-09-01T10:00:00+00:00',
    verified_at: null,
    last_used_at: null,
    last_error: null,
    key_set: true,
    key_masked: 'sk-d...1234',
    key_source: 'keychain',
    is_default: false,
    read_only: false,
    ...overrides,
  };
}

const DEFAULT_CARD = makeCard({
  id: 'card_default',
  name: 'OpenCode Go · deepseek-v4-flash',
  model: 'deepseek-v4-flash',
  key_masked: 'sk-o...wxyz',
  is_default: true,
  verified_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
});

const BAD_CARD = makeCard({
  id: 'card_bad',
  name: 'MiniMax · MiniMax-M3',
  model: 'MiniMax-M3',
  last_error: {
    status: 401,
    code: 'invalid_api_key',
    message: 'Invalid API key',
    at: '2026-09-03T22:45:00+00:00',
  },
});

const CARDS = [DEFAULT_CARD, BAD_CARD];

function optionLabels(): string[] {
  return Array.from(document.querySelectorAll('option')).map((o) => o.textContent || '');
}

describe('CardPicker', () => {
  it('lists every card with name, model and key suffix', () => {
    render(
      <CardPicker
        cards={CARDS}
        defaultCardId="card_default"
        value={null}
        onChange={vi.fn()}
        data-testid="picker"
      />,
    );
    const labels = optionLabels();
    expect(labels).toContain('OpenCode Go · deepseek-v4-flash · deepseek-v4-flash · sk-o...wxyz');
    expect(labels.some((l) => l.includes('MiniMax · MiniMax-M3'))).toBe(true);
  });

  it('names the default card in the first option and selects it for card_id null', () => {
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value={null} onChange={vi.fn()} data-testid="picker" />,
    );
    expect(optionLabels()[0]).toBe('Global default — OpenCode Go · deepseek-v4-flash');
    expect((screen.getByTestId('picker') as HTMLSelectElement).value).toBe(GLOBAL_DEFAULT_VALUE);
  });

  it('choosing "Global default" emits null, not the default card id', () => {
    const onChange = vi.fn();
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value="card_bad" onChange={onChange} data-testid="picker" />,
    );
    fireEvent.change(screen.getByTestId('picker'), { target: { value: GLOBAL_DEFAULT_VALUE } });
    // null is a live reference to whatever the default is at run time; copying
    // the id would freeze this project onto today's default card.
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('choosing a card emits its id', () => {
    const onChange = vi.fn();
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value={null} onChange={onChange} data-testid="picker" />,
    );
    fireEvent.change(screen.getByTestId('picker'), { target: { value: 'card_bad' } });
    expect(onChange).toHaveBeenCalledWith('card_bad');
  });

  it('flags a card with a last_error but keeps it selectable', () => {
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value="card_bad" onChange={vi.fn()} data-testid="picker" />,
    );
    const flagged = optionLabels().find((l) => l.includes('MiniMax'))!;
    expect(flagged.startsWith('⚠')).toBe(true);
    const opt = Array.from(document.querySelectorAll('option')).find(
      (o) => (o as HTMLOptionElement).value === 'card_bad',
    ) as HTMLOptionElement;
    expect(opt.disabled).toBe(false);
    expect((screen.getByTestId('picker') as HTMLSelectElement).value).toBe('card_bad');
  });

  it('shows the selected card health, error text included', () => {
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value="card_bad" onChange={vi.fn()} data-testid="picker" />,
    );
    expect(screen.getByTestId('picker-health').textContent).toContain('401 Invalid API key');
  });

  it('shows the DEFAULT card health when the value is null', () => {
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value={null} onChange={vi.fn()} data-testid="picker" />,
    );
    // What actually runs is the default card, so its health is the honest line.
    expect(screen.getByTestId('picker-health').textContent).toContain('Verified 2h ago');
  });

  it('a dangling card id reads as missing, never as "global default"', () => {
    render(
      <CardPicker cards={CARDS} defaultCardId="card_default" value="card_gone" onChange={vi.fn()} data-testid="picker" />,
    );
    expect(optionLabels()).toContain('That provider no longer exists');
    expect((screen.getByTestId('picker') as HTMLSelectElement).value).toBe('');
  });

  it('says a card with no model needs one', () => {
    render(
      <CardPicker
        cards={[makeCard({ id: 'card_inc', name: 'OpenRouter (launch-vid)', model: '' })]}
        defaultCardId={null}
        value="card_inc"
        onChange={vi.fn()}
        data-testid="picker"
      />,
    );
    expect(optionLabels().some((l) => l.includes('needs a model'))).toBe(true);
  });

  it('omits the "Global default" option when a card must be named', () => {
    render(
      <CardPicker
        cards={CARDS}
        defaultCardId="card_default"
        value={null}
        onChange={vi.fn()}
        allowGlobalDefault={false}
        data-testid="picker"
      />,
    );
    expect(optionLabels().some((l) => l.startsWith('Global default'))).toBe(false);
    expect(optionLabels()).toContain('Choose a provider…');
  });

  it('points at Global Settings when there are no cards at all', () => {
    render(
      <CardPicker cards={[]} defaultCardId={null} value={null} onChange={vi.fn()} data-testid="picker" />,
    );
    expect(screen.getByText(/add one below/)).toBeTruthy();
  });
});

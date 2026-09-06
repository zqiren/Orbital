// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * FallbackModelsEditor unit tests — spec 082 §5 (frontend).
 *
 * A fallback rung is now a credential-card reference, not a hand-typed
 * provider + model + key. The old datalist/provider-select tests (bug #50)
 * are gone with the fields they covered: the model a rung uses is the card's
 * model, chosen once when the card was made.
 *
 * Covers: rungs render as pickers in chain order, adding appends an empty
 * rung, picking a card emits `{card_id}` and nothing else, removing drops the
 * right rung, and legacy 5-field entries loaded from an older settings file
 * are normalised to card references on the next change.
 */

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { CredentialCard } from '../types';
import FallbackModelsEditor from './FallbackModelsEditor';

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
    verified_at: '2026-09-04T09:00:00+00:00',
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

const CARDS: CredentialCard[] = [
  makeCard({ id: 'card_default', name: 'OpenCode Go · deepseek-v4-flash', is_default: true }),
  makeCard({ id: 'card_or', name: 'My OpenRouter', model: 'deepseek/deepseek-chat-v3.1' }),
];

function expand() {
  fireEvent.click(screen.getByText('Fallback Models'));
}

describe('FallbackModelsEditor — card rungs (spec 082)', () => {
  it('renders one picker per rung, in chain order', () => {
    render(
      <FallbackModelsEditor
        models={[{ card_id: 'card_or' }, { card_id: 'card_default' }]}
        onChange={vi.fn()}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    const first = screen.getByTestId('fallback-picker-0') as HTMLSelectElement;
    const second = screen.getByTestId('fallback-picker-1') as HTMLSelectElement;
    expect(first.value).toBe('card_or');
    expect(second.value).toBe('card_default');
  });

  it('does NOT offer "Global default" — a rung has to name a card', () => {
    render(
      <FallbackModelsEditor
        models={[{ card_id: 'card_or' }]}
        onChange={vi.fn()}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    expect(screen.queryByText(/Global default/)).toBeNull();
  });

  it('adding a rung appends an empty card reference', () => {
    const onChange = vi.fn();
    render(
      <FallbackModelsEditor
        models={[{ card_id: 'card_or' }]}
        onChange={onChange}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    fireEvent.click(screen.getByTestId('fallback-add'));
    // A null rung is legal and inert — the daemon's chain builder skips it —
    // so an unfinished row can never break a run.
    expect(onChange).toHaveBeenCalledWith([{ card_id: 'card_or' }, { card_id: null }]);
  });

  it('picking a card emits {card_id} and nothing else', () => {
    const onChange = vi.fn();
    render(
      <FallbackModelsEditor
        models={[{ card_id: null }]}
        onChange={onChange}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    fireEvent.change(screen.getByTestId('fallback-picker-0'), {
      target: { value: 'card_or' },
    });
    expect(onChange).toHaveBeenCalledWith([{ card_id: 'card_or' }]);
  });

  it('removing a rung drops that one and renumbers the rest', () => {
    const onChange = vi.fn();
    render(
      <FallbackModelsEditor
        models={[{ card_id: 'card_or' }, { card_id: 'card_default' }]}
        onChange={onChange}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    fireEvent.click(screen.getByTestId('fallback-remove-0'));
    expect(onChange).toHaveBeenCalledWith([{ card_id: 'card_default' }]);
  });

  it('normalises legacy 5-field entries to card references on the next change', () => {
    const onChange = vi.fn();
    render(
      <FallbackModelsEditor
        // The pre-082 shape an older settings.json still holds.
        models={[
          { provider: 'openrouter', model: 'openai/gpt-5', sdk: 'openai', api_key: 'x' },
        ]}
        onChange={onChange}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    fireEvent.change(screen.getByTestId('fallback-picker-0'), {
      target: { value: 'card_or' },
    });
    // No provider/model/sdk/api_key survives the write — the rung is a
    // reference now, and a stray key in settings.json is exactly what the
    // cards replaced.
    expect(onChange).toHaveBeenCalledWith([{ card_id: 'card_or' }]);
  });

  it('starts collapsed with no rungs and expands to the add control', () => {
    render(
      <FallbackModelsEditor
        models={[]}
        onChange={vi.fn()}
        cards={CARDS}
        defaultCardId="card_default"
      />,
    );
    expect(screen.queryByTestId('fallback-add')).toBeNull();
    expand();
    expect(screen.getByTestId('fallback-add')).toBeTruthy();
  });
});

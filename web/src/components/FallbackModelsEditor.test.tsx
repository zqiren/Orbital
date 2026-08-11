// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * FallbackModelsEditor unit tests — bug #50 (no recommended models).
 *
 * The model field is a free-text <input> backed by a <datalist> seeded from the
 * provider registry's static `suggested_models`. Covers: options render for the
 * default provider, refresh when the provider select changes, and free text
 * outside the list still saves.
 */

import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { ProviderInfo, ProviderRegistry } from '../types';
import FallbackModelsEditor from './FallbackModelsEditor';

afterEach(() => cleanup());

function makeProvider(overrides: Partial<ProviderInfo> = {}): ProviderInfo {
  return {
    display_name: 'Test Provider',
    base_url: 'https://api.test.example/v1',
    supports_model_list: false,
    sdk: 'openai',
    suggested_models: [],
    notes: '',
    ...overrides,
  };
}

const REGISTRY: ProviderRegistry = {
  openrouter: makeProvider({
    display_name: 'OpenRouter',
    suggested_models: ['openai/gpt-5', 'anthropic/claude-sonnet-5'],
  }),
  anthropic: makeProvider({
    display_name: 'Anthropic',
    sdk: 'anthropic',
    suggested_models: ['claude-fable-5', 'claude-opus-4-8', 'claude-haiku-4-5'],
  }),
  bare: makeProvider({ display_name: 'Bare Provider' }),
};

/** Open the section and the add-entry form, then hand back the model input. */
function openAddForm(): HTMLInputElement {
  fireEvent.click(screen.getByText('Fallback Models'));
  fireEvent.click(screen.getByText('Add fallback model'));
  return screen.getByPlaceholderText(/e\.g\. gpt-4o/) as HTMLInputElement;
}

/** The provider <select>. Not getByRole('combobox') — `list=` on the model
 * input gives it that role too. */
function providerSelect(): HTMLSelectElement {
  return document.querySelector('select') as HTMLSelectElement;
}

/** The <option> values of the datalist the model input points at. */
function datalistValuesFor(input: HTMLInputElement): string[] {
  const id = input.getAttribute('list');
  expect(id).toBeTruthy();
  const list = document.getElementById(id!) as HTMLDataListElement | null;
  expect(list).toBeTruthy();
  return Array.from(list!.querySelectorAll('option')).map((o) => o.value);
}

describe('FallbackModelsEditor — model suggestions (bug #50)', () => {
  it('seeds the datalist from the default provider suggested_models', () => {
    render(<FallbackModelsEditor models={[]} onChange={vi.fn()} providers={REGISTRY} />);
    const input = openAddForm();
    expect(datalistValuesFor(input)).toEqual(REGISTRY.openrouter.suggested_models);
  });

  it('refreshes the options when the provider select changes', () => {
    render(<FallbackModelsEditor models={[]} onChange={vi.fn()} providers={REGISTRY} />);
    const input = openAddForm();

    fireEvent.change(providerSelect(), { target: { value: 'anthropic' } });
    expect(datalistValuesFor(input)).toEqual(REGISTRY.anthropic.suggested_models);

    fireEvent.change(providerSelect(), { target: { value: 'openrouter' } });
    expect(datalistValuesFor(input)).toEqual(REGISTRY.openrouter.suggested_models);
  });

  it('renders an empty datalist for a provider with no suggestions', () => {
    render(<FallbackModelsEditor models={[]} onChange={vi.fn()} providers={REGISTRY} />);
    const input = openAddForm();

    fireEvent.change(providerSelect(), { target: { value: 'bare' } });
    expect(datalistValuesFor(input)).toEqual([]);
  });

  it('keeps free-text entry working for a model outside the list', () => {
    const onChange = vi.fn();
    render(<FallbackModelsEditor models={[]} onChange={onChange} providers={REGISTRY} />);
    const input = openAddForm();

    fireEvent.change(input, { target: { value: 'some-unlisted-model-2030' } });
    fireEvent.click(screen.getByText('Add'));

    expect(onChange).toHaveBeenCalledWith([
      { provider: 'openrouter', model: 'some-unlisted-model-2030', sdk: 'openai' },
    ]);
  });
});

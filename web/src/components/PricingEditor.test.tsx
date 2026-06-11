// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import type { PricingTable } from '../budget/pricing';

afterEach(() => cleanup());

// Mock the api helper + ApiError. apiFn is swapped per test. The ApiError class
// is declared INSIDE the (hoisted) factory to avoid the temporal-dead-zone the
// hoisting otherwise triggers; tests import it back from the mocked module.
let apiFn = vi.fn();
vi.mock('../config', () => {
  class ApiError extends Error {
    constructor(public status: number, public detail: string) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return {
    api: (...args: unknown[]) => apiFn(...args),
    ApiError,
    isRelayMode: false,
    BASE_URL: 'http://localhost:8000',
  };
});

import PricingEditor from './PricingEditor';
import { ApiError as MockApiError } from '../config';

function makeTable(): PricingTable {
  return {
    override_path: '/data/pricing-overrides.json',
    providers: {
      openai: {
        currency: 'USD',
        currency_origin: 'default',
        models: {
          'gpt-4o': {
            input_per_1m: { value: 2.5, origin: 'default' },
            cached_input_per_1m: { value: 1.25, origin: 'default' },
            cache_write_per_1m: { value: 3.0, origin: 'default' },
            output_per_1m: { value: 99.0, origin: 'override' },
          },
          _default: {
            input_per_1m: { value: 1.0, origin: 'default' },
            cached_input_per_1m: { value: 0.5, origin: 'default' },
            cache_write_per_1m: { value: 1.0, origin: 'default' },
            output_per_1m: { value: 2.0, origin: 'default' },
          },
        },
      },
      anthropic: {
        currency: 'CNY',
        currency_origin: 'override',
        models: {
          _default: {
            input_per_1m: { value: 7.0, origin: 'override' },
            cached_input_per_1m: { value: 0.7, origin: 'default' },
            cache_write_per_1m: { value: 8.75, origin: 'default' },
            output_per_1m: { value: 21.0, origin: 'default' },
          },
        },
      },
    },
  };
}

beforeEach(() => {
  apiFn = vi.fn();
});

describe('PricingEditor', () => {
  it('renders provider sections with models and a clearly-labeled _default row', async () => {
    apiFn.mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());
    expect(screen.getByText('anthropic')).toBeInTheDocument();
    expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    // _default rows render the "Default (all other models)" label (one per provider).
    expect(screen.getAllByText(/Default \(all other models\)/).length).toBe(2);
  });

  it('marks override cells visually distinct (accent border) from defaults (muted)', async () => {
    apiFn.mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());

    // The output cell on openai/gpt-4o is a persisted override → accent border.
    // (data-origin is the unambiguous origin marker; the className carries the
    // bare `border-accent` utility, distinct from the always-present
    // `focus:border-accent`.)
    const overrideInput = screen.getByLabelText('Output rate for openai gpt-4o');
    expect(overrideInput.getAttribute('data-origin')).toBe('override');
    expect(overrideInput.className.split(/\s+/)).toContain('border-accent');

    // The input cell on the same model is a default → muted border, no bare
    // `border-accent` class.
    const defaultInput = screen.getByLabelText('Input rate for openai gpt-4o');
    expect(defaultInput.getAttribute('data-origin')).toBe('default');
    expect(defaultInput.className.split(/\s+/)).toContain('border-border');
    expect(defaultInput.className.split(/\s+/)).not.toContain('border-accent');
  });

  it('shows a revert control only on override cells', async () => {
    apiFn.mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());
    // override on openai/gpt-4o output → has a revert button
    expect(
      screen.getByLabelText('Revert Output for openai to default'),
    ).toBeInTheDocument();
    // a default cell has no revert button
    expect(
      screen.queryByLabelText('Revert Input for openai to default'),
    ).toBeNull();
  });

  it('rejects negative input client-side and blocks save', async () => {
    // A native <input type="number"> rejects non-numeric strings before they
    // reach React (value becomes ''), so the "not a number" branch is covered
    // at the pure-function level (validateRateInput in budget/pricing.test.ts).
    // Negative values DO reach the field, so we exercise that branch via the DOM.
    apiFn.mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());

    const cell = screen.getByLabelText('Input rate for openai gpt-4o');
    fireEvent.change(cell, { target: { value: '-5' } });
    expect(screen.getByText('Must be zero or greater.')).toBeInTheDocument();
    expect(cell.getAttribute('aria-invalid')).toBe('true');
    // Save is disabled while a field error is present.
    expect((screen.getByText('Save pricing') as HTMLButtonElement).disabled).toBe(true);
  });

  it('staging an edit builds a FULL override doc preserving other providers on save', async () => {
    apiFn
      .mockResolvedValueOnce(makeTable()) // GET
      .mockResolvedValueOnce(makeTable()); // PUT echoes table
    const onSaved = vi.fn();
    render(<PricingEditor onSaved={onSaved} />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());

    // Edit openai/gpt-4o input → 3.33.
    fireEvent.change(screen.getByLabelText('Input rate for openai gpt-4o'), {
      target: { value: '3.33' },
    });
    fireEvent.click(screen.getByText('Save pricing'));

    await waitFor(() => expect(apiFn).toHaveBeenCalledTimes(2));
    const [path, opts] = apiFn.mock.calls[1];
    expect(path).toBe('/api/v2/pricing/overrides');
    expect(opts.method).toBe('PUT');
    const body = JSON.parse(opts.body);
    // FULL-REPLACE: the edited field AND the pre-existing openai output override
    // AND anthropic's untouched currency + input override are all present.
    expect(body).toEqual({
      openai: { 'gpt-4o': { input_per_1m: 3.33, output_per_1m: 99.0 } },
      anthropic: { _currency: 'CNY', _default: { input_per_1m: 7.0 } },
    });
    // Recompute signal fired.
    expect(onSaved).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByText('Saved')).toBeInTheDocument());
  });

  it('reverting a field omits exactly that field from the PUT', async () => {
    apiFn
      .mockResolvedValueOnce(makeTable())
      .mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());

    // Revert openai/gpt-4o output (its only override).
    fireEvent.click(screen.getByLabelText('Revert Output for openai to default'));
    fireEvent.click(screen.getByText('Save pricing'));

    await waitFor(() => expect(apiFn).toHaveBeenCalledTimes(2));
    const body = JSON.parse(apiFn.mock.calls[1][1].body);
    // openai drops out entirely; anthropic's overrides survive.
    expect(body.openai).toBeUndefined();
    expect(body.anthropic).toEqual({ _currency: 'CNY', _default: { input_per_1m: 7.0 } });
  });

  it('maps a codes-only 400 to a localized error message', async () => {
    apiFn
      .mockResolvedValueOnce(makeTable())
      .mockRejectedValueOnce(new MockApiError(400, JSON.stringify({ code: 'invalid_rate' })));
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());

    // Make a valid edit so save is enabled.
    fireEvent.change(screen.getByLabelText('Input rate for openai gpt-4o'), {
      target: { value: '4' },
    });
    fireEvent.click(screen.getByText('Save pricing'));

    await waitFor(() =>
      expect(screen.getByText('A rate must be a non-negative number.')).toBeInTheDocument(),
    );
  });

  it('Save is disabled until there is a staged change', async () => {
    apiFn.mockResolvedValueOnce(makeTable());
    render(<PricingEditor />);
    await waitFor(() => expect(screen.getByText('openai')).toBeInTheDocument());
    expect((screen.getByText('Save pricing') as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByLabelText('Input rate for openai gpt-4o'), {
      target: { value: '2.9' },
    });
    expect((screen.getByText('Save pricing') as HTMLButtonElement).disabled).toBe(false);
  });

  it('shows a load error with retry when GET fails', async () => {
    apiFn.mockRejectedValueOnce(new MockApiError(500, 'boom'));
    render(<PricingEditor />);
    await waitFor(() =>
      expect(screen.getByText('Retry')).toBeInTheDocument(),
    );
  });
});

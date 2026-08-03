// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import type { CostResponse } from '../budget/types';
import type { Project } from '../types';

afterEach(() => cleanup());

vi.mock('../config', () => ({
  api: vi.fn(),
  BASE_URL: '',
  isRelayMode: false,
  ApiError: class ApiError extends Error {
    detail = '';
  },
}));

// Control the useCost return per test — the whole point of these cases is what
// the section renders BEFORE /cost resolves, and when it fails.
const mockUseCost = vi.fn();
vi.mock('../hooks/useCost', () => ({
  useCost: () => mockUseCost(),
}));

import BudgetSection from './BudgetSection';

function costOf(amount: number, currency = 'USD'): CostResponse {
  return {
    window: 'daily',
    by_currency: { [currency]: amount },
    converted_total: { currency, amount, estimated: false },
    breakdown: [],
    subagents: [],
  };
}

function project(over: Partial<Project> = {}): Project {
  return {
    project_id: 'p1',
    name: 'Proj',
    workspace: '/tmp/p1',
    model: 'm',
    api_key: '',
    base_url: null,
    autonomy: 'hands_off',
    instructions: '',
    budget_period: 'daily',
    ...over,
  };
}

function renderSection(limit = '') {
  return render(
    <BudgetSection
      project={project()}
      limit={limit}
      onLimitChange={vi.fn()}
      currency="USD"
      onCurrencyChange={vi.fn()}
      period="daily"
      onPeriodChange={vi.fn()}
      action="pause"
      onActionChange={vi.fn()}
      onEditPricing={vi.fn()}
    />,
  );
}

beforeEach(() => {
  mockUseCost.mockReset();
});

describe('BudgetSection spend meter loading/error states (bug #36)', () => {
  it('says it is loading instead of claiming $0.00 while /cost is in flight', () => {
    mockUseCost.mockReturnValue({ cost: null, loading: true, error: null, refresh: vi.fn() });
    renderSection();

    expect(screen.getByText('Loading spend…')).toBeInTheDocument();
    // The false-zero meter line must not be on screen at all.
    expect(screen.queryByText(/\$0\.00/)).toBeNull();
    // …nor the "you have spent nothing this period" empty state.
    expect(screen.queryByText('No usage recorded yet this period.')).toBeNull();
  });

  it('does not flash a zero on the mount frame, before useCost flips loading', () => {
    // useCost starts at loading=false and only sets it inside its effect, so
    // this shape is what the very first paint sees.
    mockUseCost.mockReturnValue({ cost: null, loading: false, error: null, refresh: vi.fn() });
    renderSection();

    expect(screen.getByText('Loading spend…')).toBeInTheDocument();
    expect(screen.queryByText(/\$0\.00/)).toBeNull();
  });

  it('hides the progress bar while loading even when a limit is set', () => {
    mockUseCost.mockReturnValue({ cost: null, loading: true, error: null, refresh: vi.fn() });
    renderSection('10');

    expect(screen.getByText('Loading spend…')).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('surfaces a load failure instead of showing a confident zero forever', () => {
    mockUseCost.mockReturnValue({
      cost: null,
      loading: false,
      error: 'network down',
      refresh: vi.fn(),
    });
    renderSection('10');

    expect(screen.getByText('Could not load spend.')).toBeInTheDocument();
    expect(screen.queryByText(/\$0\.00/)).toBeNull();
    expect(screen.queryByRole('progressbar')).toBeNull();
    expect(screen.queryByText('No usage recorded yet this period.')).toBeNull();
  });

  it('renders the real meter once /cost resolves', () => {
    mockUseCost.mockReturnValue({
      cost: costOf(2.5),
      loading: false,
      error: null,
      refresh: vi.fn(),
    });
    renderSection('10');

    expect(screen.getByText('Spent today ≈ $2.50 of $10.00')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.queryByText('Loading spend…')).toBeNull();
  });

  it('still shows a genuine zero once /cost resolves empty', () => {
    mockUseCost.mockReturnValue({
      cost: costOf(0),
      loading: false,
      error: null,
      refresh: vi.fn(),
    });
    renderSection();

    expect(screen.getByText('Spent today ≈ $0.00 · no limit')).toBeInTheDocument();
    expect(screen.getByText('No usage recorded yet this period.')).toBeInTheDocument();
  });

  it('keeps showing the last known spend during a background refresh', () => {
    // A WS-triggered re-fetch sets loading=true with cost still populated —
    // that must not blank out the meter.
    mockUseCost.mockReturnValue({
      cost: costOf(2.5),
      loading: true,
      error: null,
      refresh: vi.fn(),
    });
    renderSection('10');

    expect(screen.getByText('Spent today ≈ $2.50 of $10.00')).toBeInTheDocument();
    expect(screen.queryByText('Loading spend…')).toBeNull();
  });
});

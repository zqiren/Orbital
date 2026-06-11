// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import type { CostResponse } from '../budget/types';
import type { Project } from '../types';

afterEach(() => cleanup());

// Control the useCost return per test.
const mockUseCost = vi.fn();
vi.mock('../hooks/useCost', () => ({
  useCost: () => mockUseCost(),
}));

import BudgetCorner from './BudgetCorner';

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

beforeEach(() => {
  mockUseCost.mockReset();
});

describe('BudgetCorner', () => {
  it('renders nothing when /cost has not resolved and no budget pause', () => {
    mockUseCost.mockReturnValue({ cost: null, loading: true, error: null, refresh: vi.fn() });
    const { container } = render(
      <BudgetCorner project={project()} pauseReason={null} onOpenBudgetSettings={vi.fn()} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('normal: shows converted spend + window word with no limit', () => {
    mockUseCost.mockReturnValue({ cost: costOf(0.84), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner project={project()} pauseReason={null} onOpenBudgetSettings={vi.fn()} />,
    );
    const corner = screen.getByTestId('budget-corner');
    expect(corner).toHaveAttribute('data-state', 'normal');
    expect(corner.textContent).toContain('today');
    expect(corner.textContent).toContain('$0.84');
  });

  it('suppresses the noise corner when there is no spend and no limit', () => {
    mockUseCost.mockReturnValue({ cost: costOf(0), loading: false, error: null, refresh: vi.fn() });
    const { container } = render(
      <BudgetCorner project={project()} pauseReason={null} onOpenBudgetSettings={vi.fn()} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('warn: ≥80% of the limit renders the warn state with spend/limit', () => {
    mockUseCost.mockReturnValue({ cost: costOf(8.5), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner
        project={project({ budget_limit_usd: 10 })}
        pauseReason={null}
        onOpenBudgetSettings={vi.fn()}
      />,
    );
    const corner = screen.getByTestId('budget-corner');
    expect(corner).toHaveAttribute('data-state', 'warn');
    expect(corner.textContent).toContain('$8.50');
    expect(corner.textContent).toContain('$10.00');
  });

  it('exhausted: ≥100% renders the red pill', () => {
    mockUseCost.mockReturnValue({ cost: costOf(11), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner
        project={project({ budget_limit_usd: 10 })}
        pauseReason={null}
        onOpenBudgetSettings={vi.fn()}
      />,
    );
    const corner = screen.getByTestId('budget-corner');
    expect(corner).toHaveAttribute('data-state', 'exhausted');
    expect(corner.textContent).toContain('Budget paused');
  });

  it('exhausted precedence: budget pause forces the pill even when spend is under the limit', () => {
    mockUseCost.mockReturnValue({ cost: costOf(2), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner
        project={project({ budget_limit_usd: 10 })}
        pauseReason="budget"
        onOpenBudgetSettings={vi.fn()}
      />,
    );
    expect(screen.getByTestId('budget-corner')).toHaveAttribute('data-state', 'exhausted');
  });

  it('exhausted renders even when /cost is still null (budget pause known first)', () => {
    mockUseCost.mockReturnValue({ cost: null, loading: true, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner project={project()} pauseReason="budget" onOpenBudgetSettings={vi.fn()} />,
    );
    expect(screen.getByTestId('budget-corner')).toHaveAttribute('data-state', 'exhausted');
  });

  it('user pause does NOT force exhausted', () => {
    mockUseCost.mockReturnValue({ cost: costOf(2), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner
        project={project({ budget_limit_usd: 10 })}
        pauseReason="user"
        onOpenBudgetSettings={vi.fn()}
      />,
    );
    expect(screen.getByTestId('budget-corner')).toHaveAttribute('data-state', 'normal');
  });

  it('click navigates to budget settings', () => {
    mockUseCost.mockReturnValue({ cost: costOf(0.5), loading: false, error: null, refresh: vi.fn() });
    const onOpen = vi.fn();
    render(
      <BudgetCorner project={project()} pauseReason={null} onOpenBudgetSettings={onOpen} />,
    );
    fireEvent.click(screen.getByTestId('budget-corner'));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('uses the converted-total currency for the symbol (CNY)', () => {
    mockUseCost.mockReturnValue({ cost: costOf(8, 'CNY'), loading: false, error: null, refresh: vi.fn() });
    render(
      <BudgetCorner project={project({ budget_currency: 'CNY' })} pauseReason={null} onOpenBudgetSettings={vi.fn()} />,
    );
    expect(screen.getByTestId('budget-corner').textContent).toContain('¥8.00');
  });
});

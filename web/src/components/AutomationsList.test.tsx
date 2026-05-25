// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import type { Trigger } from '../types';

afterEach(() => cleanup());

// Mock useTriggers so we control what triggers are returned without any
// network or WebSocket work. The mock factory is reset per-test via vi.fn().
const mockFetchTriggers = vi.fn(async (): Promise<Trigger[]> => []);
let mockTriggers: Trigger[] = [];
let mockLoading = false;

vi.mock('../hooks/useTriggers', () => ({
  useTriggers: (_projectId: string) => ({
    triggers: mockTriggers,
    loading: mockLoading,
    fetchTriggers: mockFetchTriggers,
    toggleTrigger: vi.fn(),
    deleteTrigger: vi.fn(),
  }),
}));

import AutomationsList from './AutomationsList';

function makeScheduleTrigger(overrides: Partial<Trigger> = {}): Trigger {
  return {
    id: 'trig-sched-1',
    name: 'Daily Build',
    enabled: true,
    type: 'schedule',
    schedule: { cron: '0 9 * * 1-5', human: 'Weekdays at 9 AM', timezone: 'UTC' },
    watch_path: undefined,
    patterns: [],
    recursive: false,
    debounce_seconds: 0,
    task: 'Run the build',
    autonomy: null,
    last_triggered: '2026-05-23T09:00:00Z',
    trigger_count: 12,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

function makeFileWatchTrigger(overrides: Partial<Trigger> = {}): Trigger {
  return {
    id: 'trig-fw-1',
    name: 'On Src Change',
    enabled: true,
    type: 'file_watch',
    schedule: undefined,
    watch_path: '/home/user/src',
    patterns: ['*.py'],
    recursive: true,
    debounce_seconds: 5,
    task: 'Run lint',
    autonomy: null,
    last_triggered: '2026-05-22T14:30:00Z',
    trigger_count: 7,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('AutomationsList', () => {
  it('shows empty state when no triggers are present', async () => {
    mockTriggers = [];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    expect(screen.getByTestId('automations-empty')).toBeInTheDocument();
    expect(screen.getByTestId('automations-empty').textContent).toContain(
      'No automations configured.',
    );
  });

  it('renders a schedule trigger with name, cron condition, and last-fired date', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([trigger]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    expect(screen.getByTestId('automations-list')).toBeInTheDocument();
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();

    // Name is rendered
    expect(screen.getByText('Daily Build')).toBeInTheDocument();

    // Cron condition is rendered
    expect(
      screen.getByTestId(`automation-condition-${trigger.id}`).textContent,
    ).toBe('0 9 * * 1-5');

    // last-fired is rendered (non-empty)
    const lastFired = screen.getByTestId(`automation-last-fired-${trigger.id}`);
    expect(lastFired.textContent).not.toBe('—');
    // The date string should contain '2026' from the ISO date we passed in
    expect(lastFired.textContent).toContain('2026');
  });

  it('renders a file_watch trigger with path condition and last-fired date', async () => {
    const trigger = makeFileWatchTrigger();
    mockTriggers = [trigger];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([trigger]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    expect(screen.getByTestId('automations-list')).toBeInTheDocument();
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();

    // Name is rendered
    expect(screen.getByText('On Src Change')).toBeInTheDocument();

    // Watch path condition is rendered (labeled "path" not "cron")
    expect(
      screen.getByTestId(`automation-condition-${trigger.id}`).textContent,
    ).toBe('/home/user/src');

    // The condition label text for file_watch uses "path"
    const card = screen.getByTestId(`automation-card-${trigger.id}`);
    expect(card.textContent).toContain('path');

    // last-fired is rendered
    const lastFired = screen.getByTestId(`automation-last-fired-${trigger.id}`);
    expect(lastFired.textContent).not.toBe('—');
    expect(lastFired.textContent).toContain('2026');
  });

  it('shows "—" for last-fired when last_triggered is null (never fired)', async () => {
    const trigger = makeScheduleTrigger({ last_triggered: null, trigger_count: 0 });
    mockTriggers = [trigger];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([trigger]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    const lastFired = screen.getByTestId(`automation-last-fired-${trigger.id}`);
    expect(lastFired.textContent).toBe('—');
  });

  it('renders multiple triggers', async () => {
    const sched = makeScheduleTrigger();
    const fw = makeFileWatchTrigger();
    mockTriggers = [sched, fw];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([sched, fw]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    expect(screen.getByTestId(`automation-card-${sched.id}`)).toBeInTheDocument();
    expect(screen.getByTestId(`automation-card-${fw.id}`)).toBeInTheDocument();

    // READ-ONLY contract: the Automations surface must NOT render any
    // interactive controls (no toggle/delete buttons, no checkboxes/switches).
    // Phase 1B is read-only — this guards against a future dev wiring trigger
    // mutation UI into this surface; adding one will fail the suite here.
    expect(screen.queryByRole('button')).toBeNull();
    expect(screen.queryByRole('checkbox')).toBeNull();
    expect(screen.queryByRole('switch')).toBeNull();
  });

  it('shows enabled/disabled status indicator on each trigger card', async () => {
    const enabledTrigger = makeScheduleTrigger({ enabled: true });
    const disabledTrigger = makeFileWatchTrigger({ id: 'trig-fw-off', enabled: false });
    mockTriggers = [enabledTrigger, disabledTrigger];
    mockLoading = false;
    mockFetchTriggers.mockResolvedValue([enabledTrigger, disabledTrigger]);

    await act(async () => {
      render(<AutomationsList projectId="proj-1" />);
    });

    const enabledStatus = screen.getByTestId(`automation-status-${enabledTrigger.id}`);
    expect(enabledStatus.textContent).toBe('on');

    const disabledStatus = screen.getByTestId(`automation-status-${disabledTrigger.id}`);
    expect(disabledStatus.textContent).toBe('off');
  });
});

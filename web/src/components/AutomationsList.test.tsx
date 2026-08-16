// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, act, fireEvent } from '@testing-library/react';
import type { Trigger } from '../types';

afterEach(() => cleanup());

// Mock useTriggers so we control what triggers are returned without any
// network or WebSocket work. The mock factory is reset per-test via vi.fn().
// (The WS half — a trigger.updated event landing in a mounted list, and the
// vanish-on-disable regression — is exercised against the REAL hook in
// src/hooks/useTriggers.test.tsx.)
const mockFetchTriggers = vi.fn(async (): Promise<Trigger[]> => []);
const mockToggleTrigger = vi.fn(async () => undefined);
const mockDeleteTrigger = vi.fn(async () => undefined);
const mockUpdateTrigger = vi.fn(async () => undefined);
const mockCreateTrigger = vi.fn(async () => undefined);
let mockTriggers: Trigger[] = [];
let mockLoading = false;

vi.mock('../hooks/useTriggers', () => ({
  useTriggers: () => ({
    triggers: mockTriggers,
    loading: mockLoading,
    fetchTriggers: mockFetchTriggers,
    toggleTrigger: mockToggleTrigger,
    deleteTrigger: mockDeleteTrigger,
    updateTrigger: mockUpdateTrigger,
    createTrigger: mockCreateTrigger,
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
    watch_path: 'src',
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

async function renderPane() {
  await act(async () => {
    render(<AutomationsList projectId="proj-1" />);
  });
}

beforeEach(() => {
  mockTriggers = [];
  mockLoading = false;
  mockFetchTriggers.mockClear();
  mockToggleTrigger.mockClear();
  mockDeleteTrigger.mockClear();
  mockUpdateTrigger.mockClear();
  mockCreateTrigger.mockClear();
});

describe('AutomationsList — rows', () => {
  it('shows empty state when no triggers are present', async () => {
    await renderPane();

    expect(screen.getByTestId('automations-empty')).toBeInTheDocument();
    expect(screen.getByTestId('automations-empty').textContent).toContain(
      'No automations configured.',
    );
  });

  it('renders a schedule trigger with name, schedule caption, and last-fired date', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    expect(screen.getByTestId('automations-list')).toBeInTheDocument();
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
    expect(screen.getByText('Daily Build')).toBeInTheDocument();

    // The caption is re-derived from the cron in the reader's language rather
    // than echoing whatever language wrote schedule.human.
    expect(
      screen.getByTestId(`automation-condition-${trigger.id}`).textContent,
    ).toBe('Weekdays at 09:00');

    const lastFired = screen.getByTestId(`automation-last-fired-${trigger.id}`);
    expect(lastFired.textContent).not.toBe('—');
    expect(lastFired.textContent).toContain('2026');
    expect(screen.getByTestId(`automation-runs-${trigger.id}`).textContent).toBe('12');
  });

  it('falls back to the stored caption when the cron matches no preset', async () => {
    const trigger = makeScheduleTrigger({
      schedule: { cron: '*/7 3 5 * 2', human: 'A very odd cadence', timezone: 'UTC' },
    });
    mockTriggers = [trigger];
    await renderPane();

    expect(
      screen.getByTestId(`automation-condition-${trigger.id}`).textContent,
    ).toBe('A very odd cadence');
  });

  it('renders a file_watch trigger with path condition and last-fired date', async () => {
    const trigger = makeFileWatchTrigger();
    mockTriggers = [trigger];
    await renderPane();

    expect(screen.getByText('On Src Change')).toBeInTheDocument();
    expect(
      screen.getByTestId(`automation-condition-${trigger.id}`).textContent,
    ).toBe('src');
    // Labelled by what it is, not by the field name.
    expect(screen.getByTestId(`automation-card-${trigger.id}`).textContent).toContain(
      'Watching',
    );
  });

  it('shows "—" for last-fired when last_triggered is null (never fired)', async () => {
    const trigger = makeScheduleTrigger({ last_triggered: null, trigger_count: 0 });
    mockTriggers = [trigger];
    await renderPane();

    expect(screen.getByTestId(`automation-last-fired-${trigger.id}`).textContent).toBe('—');
    expect(screen.queryByTestId(`automation-runs-${trigger.id}`)).toBeNull();
  });

  it('shows enabled/disabled status on each row, on the dot AND the switch', async () => {
    const on = makeScheduleTrigger({ enabled: true });
    const off = makeFileWatchTrigger({ id: 'trig-fw-off', enabled: false });
    mockTriggers = [on, off];
    await renderPane();

    expect(screen.getByTestId(`automation-status-${on.id}`).textContent).toBe('on');
    expect(screen.getByTestId(`automation-status-${off.id}`).textContent).toBe('off');
    expect(screen.getByTestId(`automation-toggle-${on.id}`)).toHaveAttribute(
      'aria-checked',
      'true',
    );
    expect(screen.getByTestId(`automation-toggle-${off.id}`)).toHaveAttribute(
      'aria-checked',
      'false',
    );
  });
});

describe('AutomationsList — toggle and delete', () => {
  it('toggling a row switches it off through the hook', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-toggle-${trigger.id}`));
    });
    expect(mockToggleTrigger).toHaveBeenCalledWith(trigger.id, false);
  });

  it('toggling a disabled row switches it back on', async () => {
    const trigger = makeScheduleTrigger({ enabled: false });
    mockTriggers = [trigger];
    await renderPane();

    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-toggle-${trigger.id}`));
    });
    expect(mockToggleTrigger).toHaveBeenCalledWith(trigger.id, true);
  });

  it('delete is a two-step inline confirm, and cancelling deletes nothing', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    expect(screen.queryByTestId(`automation-delete-confirm-${trigger.id}`)).toBeNull();
    fireEvent.click(screen.getByTestId(`automation-delete-${trigger.id}`));
    expect(screen.getByTestId(`automation-delete-confirm-${trigger.id}`)).toBeInTheDocument();
    expect(mockDeleteTrigger).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText('Cancel'));
    expect(screen.queryByTestId(`automation-delete-confirm-${trigger.id}`)).toBeNull();
    expect(mockDeleteTrigger).not.toHaveBeenCalled();
  });

  it('confirming the delete calls the hook', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-delete-${trigger.id}`));
    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-delete-confirm-btn-${trigger.id}`));
    });
    expect(mockDeleteTrigger).toHaveBeenCalledWith(trigger.id);
  });
});

describe('AutomationsList — edit', () => {
  it('opens the form seeded from the trigger and saves a patch without type/enabled', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-edit-${trigger.id}`));
    const form = screen.getByTestId('automation-form');
    expect(form).toBeInTheDocument();
    // The row it replaced is gone while editing.
    expect(screen.queryByTestId(`automation-card-${trigger.id}`)).toBeNull();

    // Seeded: name, prompt, and the preset reverse-parsed from the cron.
    expect(screen.getByTestId('automation-form-name')).toHaveValue('Daily Build');
    expect(screen.getByTestId('automation-form-prompt')).toHaveValue('Run the build');
    expect(screen.getByTestId('schedule-preset')).toHaveValue('weekdays');
    expect(screen.getByTestId('schedule-time')).toHaveValue('09:00');
    // Type is fixed after create — no selector, just a label.
    expect(screen.queryByTestId('automation-form-type')).toBeNull();
    expect(screen.getByTestId('automation-form-type-fixed')).toBeInTheDocument();

    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Nightly Build' },
    });
    fireEvent.change(screen.getByTestId('schedule-preset'), { target: { value: 'daily' } });
    fireEvent.change(screen.getByTestId('schedule-time'), { target: { value: '23:30' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    expect(mockUpdateTrigger).toHaveBeenCalledTimes(1);
    const [id, patch] = mockUpdateTrigger.mock.calls[0] as unknown as [
      string,
      Record<string, unknown>,
    ];
    expect(id).toBe(trigger.id);
    expect(patch).toEqual({
      name: 'Nightly Build',
      task: 'Run the build',
      schedule: {
        cron: '30 23 * * *',
        human: 'Every day at 23:30',
        timezone: 'UTC',
      },
    });
    expect(patch).not.toHaveProperty('type');
    expect(patch).not.toHaveProperty('enabled');
  });

  it('edits the file_watch fields the agent tool cannot reach', async () => {
    const trigger = makeFileWatchTrigger();
    mockTriggers = [trigger];
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-edit-${trigger.id}`));
    expect(screen.getByTestId('automation-form-watch-path')).toHaveValue('src');
    expect(screen.getByTestId('automation-form-patterns')).toHaveValue('*.py');

    fireEvent.change(screen.getByTestId('automation-form-watch-path'), {
      target: { value: 'uploads' },
    });
    fireEvent.change(screen.getByTestId('automation-form-patterns'), {
      target: { value: '*.png, *.gif' },
    });
    fireEvent.click(screen.getByTestId('automation-form-recursive'));
    fireEvent.change(screen.getByTestId('automation-form-debounce'), {
      target: { value: '20' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    const [, patch] = mockUpdateTrigger.mock.calls[0] as unknown as [
      string,
      Record<string, unknown>,
    ];
    expect(patch).toMatchObject({
      watch_path: 'uploads',
      patterns: ['*.png', '*.gif'],
      recursive: false, // fixture starts recursive:true; the click cleared it
      debounce_seconds: 20,
    });
  });

  it('cancelling the edit restores the row and saves nothing', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-edit-${trigger.id}`));
    fireEvent.click(screen.getByTestId('automation-form-cancel'));
    expect(screen.queryByTestId('automation-form')).toBeNull();
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
    expect(mockUpdateTrigger).not.toHaveBeenCalled();
  });

  it('surfaces the server error inline instead of closing the form', async () => {
    const trigger = makeScheduleTrigger();
    mockTriggers = [trigger];
    mockUpdateTrigger.mockRejectedValueOnce(new Error('Invalid cron expression: 99 99 * * *'));
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-edit-${trigger.id}`));
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    expect(screen.getByTestId('automation-form-error').textContent).toContain(
      'Invalid cron expression',
    );
    expect(screen.getByTestId('automation-form')).toBeInTheDocument();
  });
});

describe('AutomationsList — create', () => {
  it('creates a schedule automation with both cron and a derived caption', async () => {
    await renderPane();

    fireEvent.click(screen.getByTestId('automations-new'));
    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Weekly report' },
    });
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Write the weekly report' },
    });
    fireEvent.change(screen.getByTestId('schedule-preset'), { target: { value: 'weekly' } });
    fireEvent.change(screen.getByTestId('schedule-weekday'), { target: { value: '5' } });
    fireEvent.change(screen.getByTestId('schedule-time'), { target: { value: '17:00' } });
    fireEvent.change(screen.getByTestId('schedule-timezone'), { target: { value: 'UTC' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    expect(mockCreateTrigger).toHaveBeenCalledWith({
      name: 'Weekly report',
      type: 'schedule',
      task: 'Write the weekly report',
      schedule: {
        cron: '0 17 * * 5',
        human: 'Every Friday at 17:00',
        timezone: 'UTC',
      },
    });
  });

  it('custom cron stores the expression as the caption rather than a broken sentence', async () => {
    await renderPane();

    fireEvent.click(screen.getByTestId('automations-new'));
    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Odd one' },
    });
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Do the odd thing' },
    });
    fireEvent.change(screen.getByTestId('schedule-preset'), { target: { value: 'custom' } });
    fireEvent.change(screen.getByTestId('schedule-cron'), {
      target: { value: '*/15 2 * * 3' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    const draft = (mockCreateTrigger.mock.calls as unknown as [
      { schedule: { cron: string; human: string } },
    ][])[0][0];
    expect(draft.schedule.cron).toBe('*/15 2 * * 3');
    expect(draft.schedule.human).toBe('*/15 2 * * 3');
  });

  it('creates a file_watch automation', async () => {
    await renderPane();

    fireEvent.click(screen.getByTestId('automations-new'));
    fireEvent.change(screen.getByTestId('automation-form-type'), {
      target: { value: 'file_watch' },
    });
    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Photo watcher' },
    });
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Sort the photos' },
    });
    fireEvent.change(screen.getByTestId('automation-form-watch-path'), {
      target: { value: 'incoming' },
    });
    fireEvent.change(screen.getByTestId('automation-form-patterns'), {
      target: { value: '*.jpg, *.png' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    expect(mockCreateTrigger).toHaveBeenCalledWith({
      name: 'Photo watcher',
      type: 'file_watch',
      task: 'Sort the photos',
      watch_path: 'incoming',
      patterns: ['*.jpg', '*.png'],
      recursive: false,
      debounce_seconds: 5,
    });
  });

  it('blocks submit on a missing name and never calls the API', async () => {
    await renderPane();

    fireEvent.click(screen.getByTestId('automations-new'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    expect(screen.getByTestId('automation-form-error').textContent).toContain(
      'Name is required',
    );
    expect(mockCreateTrigger).not.toHaveBeenCalled();
  });
});

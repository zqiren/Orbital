// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * The WS half of item #57, exercised against the REAL useTriggers hook with a
 * fake transport (AutomationsList.test.tsx stubs the hook to test the UI).
 *
 * Headline case: the vanish-on-disable regression. Switching an automation off
 * used to arrive as `trigger.deleted` — the REST toggle route called
 * unregister_trigger, which broadcast a delete — so the row disappeared from
 * every mounted list until the next refetch. Wiring a toggle into a live list
 * without fixing that would have shipped a control that makes its own row
 * vanish on first use.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act, fireEvent } from '@testing-library/react';
import type { Trigger, WebSocketEvent } from '../types';

const h = vi.hoisted(() => {
  const handlers = new Map<string, Set<(e: unknown) => void>>();
  return {
    handlers,
    on: (type: string, fn: (e: unknown) => void) => {
      if (!handlers.has(type)) handlers.set(type, new Set());
      handlers.get(type)!.add(fn);
    },
    off: (type: string, fn: (e: unknown) => void) => {
      handlers.get(type)?.delete(fn);
    },
    api: vi.fn(),
  };
});

vi.mock('../config', () => ({ api: h.api }));
vi.mock('./useWebSocket', () => ({
  useWebSocket: () => ({
    on: h.on,
    off: h.off,
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

import AutomationsList from '../components/AutomationsList';

function emit(event: WebSocketEvent) {
  act(() => {
    h.handlers.get(event.type)?.forEach((fn) => fn(event));
  });
}

function makeTrigger(overrides: Partial<Trigger> = {}): Trigger {
  return {
    id: 'trg_aaa',
    name: 'Morning brief',
    enabled: true,
    type: 'schedule',
    schedule: { cron: '0 7 * * *', human: 'Every day at 07:00', timezone: 'UTC' },
    task: 'Summarize the inbox',
    autonomy: null,
    last_triggered: null,
    trigger_count: 0,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

const PROJECT_ID = 'proj-1';

beforeEach(() => {
  h.handlers.clear();
  h.api.mockReset();
});

afterEach(() => cleanup());

/** GET returns `list`; every other verb returns whatever `mutate` yields. */
function stubApi(list: Trigger[], mutate?: (path: string, init?: RequestInit) => unknown) {
  h.api.mockImplementation(async (path: string, init?: RequestInit) => {
    if (!init || !init.method || init.method === 'GET') return list;
    return mutate ? mutate(path, init) : undefined;
  });
}

async function renderPane() {
  await act(async () => {
    render(<AutomationsList projectId={PROJECT_ID} />);
  });
}

describe('useTriggers — vanish-on-disable regression', () => {
  it('keeps the row in the list, updated in place, when it is switched off', async () => {
    const trigger = makeTrigger();
    const disabled = { ...trigger, enabled: false };
    stubApi([trigger], () => disabled);
    await renderPane();

    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
    expect(screen.getByTestId(`automation-status-${trigger.id}`).textContent).toBe('on');

    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-toggle-${trigger.id}`));
    });
    // …and the broadcast the backend now sends for a toggle arrives.
    emit({ type: 'trigger.updated', project_id: PROJECT_ID, trigger: disabled });

    // THE assertion: the row is still there, showing its new state.
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
    expect(screen.getByTestId(`automation-status-${trigger.id}`).textContent).toBe('off');
    expect(screen.getByTestId(`automation-toggle-${trigger.id}`)).toHaveAttribute(
      'aria-checked',
      'false',
    );
    // No refetch was needed to recover it.
    const gets = h.api.mock.calls.filter(([, init]) => !init || !init.method);
    expect(gets.length).toBe(1);
    // What the wire is allowed to carry for a disable is asserted where it is
    // decided: tests/unit/test_triggers.py::TestTriggerBroadcastSemantics.
  });

  it('a disable performed on ANOTHER surface updates this row instead of removing it', async () => {
    // The load-bearing case: no local PATCH response to fall back on, only the
    // broadcast. This is the App-level strip toggling, or the agent's
    // update_trigger tool firing. Under the old event model the wire carried
    // trigger.deleted here and the row disappeared.
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    emit({
      type: 'trigger.updated',
      project_id: PROJECT_ID,
      trigger: { ...trigger, enabled: false },
    });

    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
    expect(screen.getByTestId(`automation-status-${trigger.id}`).textContent).toBe('off');
  });

  it('sends the toggle as a PATCH of {enabled}', async () => {
    const trigger = makeTrigger();
    stubApi([trigger], () => ({ ...trigger, enabled: false }));
    await renderPane();

    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-toggle-${trigger.id}`));
    });

    const patch = h.api.mock.calls.find(([, init]) => init?.method === 'PATCH');
    expect(patch).toBeTruthy();
    expect(patch![0]).toBe(`/api/v2/projects/proj-1/triggers/${trigger.id}`);
    expect(JSON.parse(patch![1].body as string)).toEqual({ enabled: false });
  });
});

describe('useTriggers — live event handling', () => {
  it('updates a row in place without reordering the list', async () => {
    const a = makeTrigger({ id: 'trg_a', name: 'A' });
    const b = makeTrigger({ id: 'trg_b', name: 'B' });
    stubApi([a, b]);
    await renderPane();

    emit({
      type: 'trigger.updated',
      project_id: PROJECT_ID,
      trigger: { ...a, name: 'A renamed' },
    });

    const cards = screen.getAllByTestId(/^automation-card-/);
    expect(cards.map((c) => c.getAttribute('data-testid'))).toEqual([
      'automation-card-trg_a',
      'automation-card-trg_b',
    ]);
    expect(screen.getByText('A renamed')).toBeInTheDocument();
  });

  it('ignores trigger.updated for a different project', async () => {
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    emit({
      type: 'trigger.updated',
      project_id: 'some-other-project',
      trigger: { ...trigger, name: 'Should not appear' },
    });

    expect(screen.queryByText('Should not appear')).toBeNull();
    expect(screen.getByText('Morning brief')).toBeInTheDocument();
  });

  it('ignores trigger.updated for an id it does not have (no phantom row)', async () => {
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    emit({
      type: 'trigger.updated',
      project_id: PROJECT_ID,
      trigger: makeTrigger({ id: 'trg_ghost', name: 'Ghost' }),
    });

    expect(screen.queryByText('Ghost')).toBeNull();
    expect(screen.getAllByTestId(/^automation-card-/).length).toBe(1);
  });

  it('still adds a row on trigger.created and drops one on trigger.deleted', async () => {
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    emit({
      type: 'trigger.created',
      project_id: PROJECT_ID,
      trigger: makeTrigger({ id: 'trg_new', name: 'Fresh' }),
    });
    expect(screen.getByText('Fresh')).toBeInTheDocument();

    emit({ type: 'trigger.deleted', project_id: PROJECT_ID, trigger_id: 'trg_new' });
    expect(screen.queryByText('Fresh')).toBeNull();
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
  });

  it('bumps last-fired and the run count on trigger.fired', async () => {
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    emit({
      type: 'trigger.fired',
      project_id: PROJECT_ID,
      trigger_id: trigger.id,
      trigger_name: trigger.name,
      timestamp: '2026-08-16T07:00:00Z',
    });

    expect(screen.getByTestId(`automation-runs-${trigger.id}`).textContent).toBe('1');
    expect(screen.getByTestId(`automation-last-fired-${trigger.id}`).textContent).not.toBe('—');
  });
});

describe('useTriggers — create and delete over the wire', () => {
  it('POSTs a create and appends the returned record', async () => {
    const created = makeTrigger({ id: 'trg_created', name: 'Weekly report' });
    stubApi([], () => created);
    await renderPane();

    fireEvent.click(screen.getByTestId('automations-new'));
    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Weekly report' },
    });
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Write it' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('automation-form-save'));
    });

    const post = h.api.mock.calls.find(([, init]) => init?.method === 'POST');
    expect(post![0]).toBe('/api/v2/projects/proj-1/triggers');
    expect(JSON.parse(post![1].body as string)).toMatchObject({
      name: 'Weekly report',
      type: 'schedule',
      task: 'Write it',
      schedule: { cron: '0 9 * * *', human: 'Every day at 09:00' },
    });
    expect(screen.getByText('Weekly report')).toBeInTheDocument();
    expect(screen.queryByTestId('automation-form')).toBeNull();
  });

  it('DELETEs and removes the row', async () => {
    const trigger = makeTrigger();
    stubApi([trigger]);
    await renderPane();

    fireEvent.click(screen.getByTestId(`automation-delete-${trigger.id}`));
    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-delete-confirm-btn-${trigger.id}`));
    });

    const del = h.api.mock.calls.find(([, init]) => init?.method === 'DELETE');
    expect(del![0]).toBe(`/api/v2/projects/proj-1/triggers/${trigger.id}`);
    expect(screen.queryByTestId(`automation-card-${trigger.id}`)).toBeNull();
  });

  it('surfaces a failed toggle without dropping the row', async () => {
    const trigger = makeTrigger();
    h.api.mockImplementation(async (_path: string, init?: RequestInit) => {
      if (!init || !init.method || init.method === 'GET') return [trigger];
      throw new Error('Trigger not found');
    });
    await renderPane();

    await act(async () => {
      fireEvent.click(screen.getByTestId(`automation-toggle-${trigger.id}`));
    });

    expect(screen.getByTestId('automations-error').textContent).toContain(
      'Trigger not found',
    );
    expect(screen.getByTestId(`automation-card-${trigger.id}`)).toBeInTheDocument();
  });
});

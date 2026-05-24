// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import QueueItemCard from './QueueItemCard';
import type { QueueItem } from '../types';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeItem(overrides: Partial<QueueItem> = {}): QueueItem {
  return {
    id: 'item-1',
    content: 'Test task content',
    file_refs: [],
    priority: 0,
    review_before_advance: false,
    state: 'queued',
    source: 'user',
    attempts: [],
    idempotency_key: null,
    interrupted_count: 0,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Trigger-origin visuals
// ---------------------------------------------------------------------------

describe('QueueItemCard — trigger-origin visuals', () => {
  it('renders the trigger origin badge and "from: <name>" subtitle for a trigger-spawned item', () => {
    // MOCK: backend does not emit source='trigger' yet (Phase 1B visual-only).
    // We inject a synthetic item here to verify the rendering capability.
    const item = makeItem({
      id: 'trigger-item-1',
      source: 'trigger',
      trigger_name: 'Nightly report',
    });
    render(<QueueItemCard item={item} />);

    const badge = screen.getByTestId('queue-item-trigger-origin');
    expect(badge).toBeInTheDocument();
    expect(badge.textContent).toContain('from: Nightly report');
  });

  it('renders "from: <trigger_name>" when trigger_name is present (regardless of source field)', () => {
    // Covers the case where only trigger_name is set (forward-compat path).
    const item = makeItem({
      id: 'trigger-item-2',
      source: 'user', // source still 'user', but trigger_name is set
      trigger_name: 'File watcher',
    });
    render(<QueueItemCard item={item} />);

    const badge = screen.getByTestId('queue-item-trigger-origin');
    expect(badge).toBeInTheDocument();
    expect(badge.textContent).toContain('from: File watcher');
  });

  it('falls back to "from: automation" when source is trigger but trigger_name is absent', () => {
    const item = makeItem({ source: 'trigger' });
    render(<QueueItemCard item={item} />);

    const badge = screen.getByTestId('queue-item-trigger-origin');
    expect(badge).toBeInTheDocument();
    expect(badge.textContent).toContain('from: automation');
  });

  it('does NOT render the trigger origin badge for a normal user item', () => {
    const item = makeItem({ source: 'user' });
    render(<QueueItemCard item={item} />);

    expect(screen.queryByTestId('queue-item-trigger-origin')).toBeNull();
  });

  it('does NOT render the trigger origin badge for an upload item', () => {
    const item = makeItem({ source: 'upload' });
    render(<QueueItemCard item={item} />);

    expect(screen.queryByTestId('queue-item-trigger-origin')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// State coverage — trigger-spawned item renders correctly across all states
// ---------------------------------------------------------------------------

describe('QueueItemCard — state coverage with trigger origin', () => {
  const triggerBase: Partial<QueueItem> = {
    source: 'trigger',
    trigger_name: 'Daily sync',
  };

  it('queued state renders correctly', () => {
    const item = makeItem({ ...triggerBase, state: 'queued', id: 'tq-1' });
    render(<QueueItemCard item={item} />);
    expect(screen.getByTestId('queue-item-tq-1')).toHaveAttribute('data-state', 'queued');
    expect(screen.getByTestId('queue-item-trigger-origin')).toBeInTheDocument();
  });

  it('running state renders correctly', () => {
    const item = makeItem({ ...triggerBase, state: 'running', id: 'tq-2' });
    render(<QueueItemCard item={item} />);
    expect(screen.getByTestId('queue-item-tq-2')).toHaveAttribute('data-state', 'running');
    expect(screen.getByTestId('queue-item-trigger-origin')).toBeInTheDocument();
  });

  it('done state renders correctly (shows summary when present)', () => {
    const item = makeItem({
      ...triggerBase,
      state: 'done',
      id: 'tq-3',
      attempts: [
        {
          session_id: 'sess-1',
          started_at: '2026-01-01T00:00:00Z',
          ended_at: '2026-01-01T00:01:00Z',
          outcome: 'completed',
          summary: 'All done',
          block_reason: null,
        },
      ],
    });
    render(<QueueItemCard item={item} />);
    expect(screen.getByTestId('queue-item-tq-3')).toHaveAttribute('data-state', 'done');
    expect(screen.getByTestId('queue-item-trigger-origin')).toBeInTheDocument();
    expect(screen.getByText('All done')).toBeInTheDocument();
  });

  it('blocked state renders correctly (shows block_reason when present)', () => {
    const item = makeItem({
      ...triggerBase,
      state: 'blocked',
      id: 'tq-4',
      attempts: [
        {
          session_id: 'sess-2',
          started_at: '2026-01-01T00:00:00Z',
          ended_at: '2026-01-01T00:01:00Z',
          outcome: 'blocked',
          summary: null,
          block_reason: 'Needs approval',
        },
      ],
    });
    render(<QueueItemCard item={item} />);
    expect(screen.getByTestId('queue-item-tq-4')).toHaveAttribute('data-state', 'blocked');
    expect(screen.getByTestId('queue-item-trigger-origin')).toBeInTheDocument();
    expect(screen.getByText('Needs approval')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Existing user/upload item rendering is unchanged
// ---------------------------------------------------------------------------

describe('QueueItemCard — existing item rendering unchanged', () => {
  it('renders item content for a normal user item', () => {
    const item = makeItem({ content: 'Do the thing', source: 'user', id: 'u-1' });
    render(<QueueItemCard item={item} />);
    expect(screen.getByText('Do the thing')).toBeInTheDocument();
    expect(screen.queryByTestId('queue-item-trigger-origin')).toBeNull();
  });

  it('renders interrupted count for a user item', () => {
    const item = makeItem({ source: 'user', interrupted_count: 2, id: 'u-2' });
    render(<QueueItemCard item={item} />);
    expect(screen.getByText(/Interrupted 2×/)).toBeInTheDocument();
    expect(screen.queryByTestId('queue-item-trigger-origin')).toBeNull();
  });

  it('remove button is present when onRemove is supplied and state is not running', () => {
    const onRemove = vi.fn();
    const item = makeItem({ id: 'u-3', state: 'queued', source: 'user' });
    render(<QueueItemCard item={item} onRemove={onRemove} />);
    expect(screen.getByRole('button', { name: /remove item/i })).toBeInTheDocument();
  });

  it('remove button fires onRemove with item id', async () => {
    const onRemove = vi.fn();
    const item = makeItem({ id: 'u-4', state: 'queued', source: 'user' });
    render(<QueueItemCard item={item} onRemove={onRemove} />);
    await userEvent.click(screen.getByRole('button', { name: /remove item/i }));
    expect(onRemove).toHaveBeenCalledWith('u-4');
  });

  it('remove button is NOT shown when item is running', () => {
    const item = makeItem({ id: 'u-5', state: 'running', source: 'user' });
    render(<QueueItemCard item={item} onRemove={vi.fn()} />);
    expect(screen.queryByRole('button', { name: /remove item/i })).toBeNull();
  });
});

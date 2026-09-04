// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

afterEach(() => cleanup());
import QueueItemCard from '../QueueItemCard';
import type { QueueItem } from '../../types';

function makeItem(overrides: Partial<QueueItem> = {}): QueueItem {
  return {
    id: 'item_test',
    content: 'do the thing',
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

describe('QueueItemCard', () => {
  it('renders queued variant with X button', () => {
    const onRemove = vi.fn();
    render(<QueueItemCard item={makeItem()} onRemove={onRemove} />);
    expect(screen.getByText('do the thing')).toBeTruthy();
    const removeBtn = screen.getByLabelText('Remove item');
    fireEvent.click(removeBtn);
    expect(onRemove).toHaveBeenCalledWith('item_test');
  });

  it('renders running variant with a DISABLED remove button (locked behavior)', () => {
    const onRemove = vi.fn();
    render(
      <QueueItemCard
        item={makeItem({ state: 'running' })}
        onRemove={onRemove}
      />,
    );
    const removeBtn = screen.getByLabelText('Remove item');
    expect(removeBtn).toBeTruthy();
    expect(removeBtn).toBeDisabled();
    fireEvent.click(removeBtn);
    expect(onRemove).not.toHaveBeenCalled();
  });

  it('renders done variant with summary', () => {
    render(
      <QueueItemCard
        item={makeItem({
          state: 'done',
          attempts: [
            {
              session_id: 's',
              started_at: 's',
              ended_at: 'e',
              outcome: 'completed',
              summary: 'wrote the feature',
              block_reason: null,
            },
          ],
        })}
      />,
    );
    expect(screen.getByText('wrote the feature')).toBeTruthy();
  });

  it('renders blocked variant with block_reason', () => {
    render(
      <QueueItemCard
        item={makeItem({
          state: 'blocked',
          attempts: [
            {
              session_id: 's',
              started_at: 's',
              ended_at: 'e',
              outcome: 'blocked',
              summary: null,
              block_reason: 'needs API key',
            },
          ],
        })}
      />,
    );
    expect(screen.getByText('needs API key')).toBeTruthy();
  });

  it('shows interrupted_count when > 0', () => {
    render(
      <QueueItemCard
        item={makeItem({ interrupted_count: 2, state: 'blocked' })}
      />,
    );
    expect(screen.getByText(/Interrupted 2/)).toBeTruthy();
  });
});

describe('QueueItemCard — attached files', () => {
  it('renders the basename of each file_ref on a queued item', () => {
    render(
      <QueueItemCard
        item={makeItem({
          file_refs: ['uploads/2026-08-11T101010-shot.png', 'uploads/notes.txt'],
        })}
      />,
    );
    const strip = screen.getByTestId('queue-item-file-refs');
    expect(strip.textContent).toContain('2026-08-11T101010-shot.png');
    expect(strip.textContent).toContain('notes.txt');
    // Directory components are dropped from the label but kept in the title.
    expect(strip.textContent).not.toContain('uploads/');
    expect(screen.getByTitle('uploads/notes.txt')).toBeTruthy();
  });

  it('renders nothing when file_refs is empty', () => {
    render(<QueueItemCard item={makeItem()} />);
    expect(screen.queryByTestId('queue-item-file-refs')).toBeNull();
  });

  it('renders attached files on a running item too', () => {
    render(
      <QueueItemCard
        item={makeItem({ state: 'running', file_refs: ['uploads/a.png'] })}
      />,
    );
    expect(screen.getByTestId('queue-item-file-refs').textContent).toContain('a.png');
  });
});

// Spec 079 — the chip naming the worker the user assigned to this item.
describe('QueueItemCard — agent chip', () => {
  it('is absent on an unassigned item (Orbital runs it, which is the norm)', () => {
    render(<QueueItemCard item={makeItem()} />);
    expect(screen.queryByTestId('queue-item-agent-item_test')).toBeNull();
  });

  it.each(['queued', 'running', 'blocked', 'done'] as const)(
    'renders on a %s item so the runner is visible in every state',
    (state) => {
      render(<QueueItemCard item={makeItem({ state, agent: 'codex' })} />);
      const chip = screen.getByTestId('queue-item-agent-item_test');
      expect(chip.getAttribute('title')).toContain('codex');
    },
  );

  it('renders for a slug whose agent is no longer installed', () => {
    // A stale choice must stay visible rather than silently reading as
    // "Orbital runs it" — the avatar falls back to a monogram badge.
    render(<QueueItemCard item={makeItem({ agent: 'uninstalled-worker' })} />);
    expect(
      screen.getByTestId('queue-item-agent-item_test').getAttribute('title'),
    ).toContain('uninstalled-worker');
  });
});

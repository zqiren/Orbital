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

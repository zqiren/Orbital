// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/** One Workbench card on its own: the reply field's IME guard. */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import TaskCard from './TaskCard';
import type { FeedItem } from './feedModel';
import type { WorkbenchFeedRow } from '../../types';

afterEach(cleanup);

function row(over: Partial<WorkbenchFeedRow> = {}): WorkbenchFeedRow {
  return {
    project_id: 'p1', session_id: 's1', session_uuid: 's1', name: 'Hero copy', agent: null,
    status: 'idle', worker_running: false, last_user_at: '2026-11-05T09:00:00Z',
    last_reply_at: '2026-11-05T09:30:00Z', checked_reply_at: null, ask: null,
    result_line: null, stop: null, ...over,
  } as WorkbenchFeedRow;
}

function renderCard(r: WorkbenchFeedRow = row()) {
  const item: FeedItem = { key: 'k', row: r, pending: null, dot: 'new', waiting: true, time: r.last_reply_at };
  const onReply = vi.fn((_item: FeedItem, _words: string) => Promise.resolve(true));
  render(
    <TaskCard
      item={item}
      projectName={() => 'Marketing'}
      agentName={(s) => s}
      activity={null}
      phone={false}
      onOpen={vi.fn()}
      onConfirm={vi.fn()}
      onReply={onReply}
    />,
  );
  return { onReply };
}

describe('TaskCard — reply field IME guard', () => {
  it('an Enter with keyCode 229 (IME commit) does not reply', () => {
    const { onReply } = renderCard();
    const field = screen.getByTestId('wb-card-reply');
    fireEvent.change(field, { target: { value: '好的' } });
    fireEvent.keyDown(field, { key: 'Enter', keyCode: 229 });
    expect(onReply).not.toHaveBeenCalled();
    expect(field).toHaveValue('好的');
  });

  it('a plain Enter replies', () => {
    const { onReply } = renderCard();
    const field = screen.getByTestId('wb-card-reply');
    fireEvent.change(field, { target: { value: '好的' } });
    fireEvent.keyDown(field, { key: 'Enter', keyCode: 13 });
    expect(onReply).toHaveBeenCalledTimes(1);
    expect(onReply.mock.calls[0][1]).toBe('好的');
  });
});

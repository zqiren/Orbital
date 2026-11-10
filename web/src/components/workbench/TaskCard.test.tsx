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

describe('TaskCard — an ask\'s detail renders as markdown', () => {
  it('bold and a list come through as elements, under the question', () => {
    renderCard(row({
      ask: { id: 'abc123', text: '先看哪里？', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: ['A', 'B'],
        detail: '先看 **这里**\n\n- a\n- b' },
    }));
    const detail = screen.getByTestId('wb-card-askdetail');
    expect(detail.querySelector('strong')).toHaveTextContent('这里');
    expect(detail.querySelectorAll('li')).toHaveLength(2);
    expect(screen.getByTestId('wb-card-meta')).toHaveTextContent('Marketing · Hero copy');
  });

  it('a non-ask title is plain text, with no clamp span', () => {
    renderCard();
    expect(screen.getByTestId('wb-card-title').textContent).toBe('Hero copy');
    expect(screen.queryByTestId('wb-card-title-clamp')).toBeNull();
  });

  it('no detail, no detail block', () => {
    renderCard(row({ ask: { id: 'abc124', text: 'Keep it?', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: [], detail: '' } }));
    expect(screen.queryByTestId('wb-card-askdetail')).toBeNull();
  });
});

describe('TaskCard — an ask\'s question renders as inline markdown', () => {
  it('bold and code come through inside the title button, in quotes', () => {
    renderCard(row({ ask: { id: 'abc125', text: '看**这里** 和 `x`', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: [], detail: '' } }));
    const title = screen.getByTestId('wb-card-title');
    // The clamp lives on the inner span (WebKit ignores it on a button).
    expect(title.firstElementChild).toBe(screen.getByTestId('wb-card-title-clamp'));
    expect(title.querySelector('strong')).toHaveTextContent('这里');
    expect(title.querySelector('code')).toHaveTextContent('x');
    expect(title.querySelector('p')).toBeNull();
    expect(title.textContent).toBe('“看这里 和 x”');
    expect(title.getAttribute('title') ?? '').not.toContain('**');
  });

  it('a link in the question is text inside the title button', () => {
    renderCard(row({ ask: { id: 'abc126', text: '读 [这篇](https://example.com)', created: '2026-11-05T09:00:00Z', due: null, overdue: false, options: [], detail: '' } }));
    const title = screen.getByTestId('wb-card-title');
    expect(title.querySelector('a')).toBeNull();
    expect(title.textContent).toBe('“读 这篇”');
  });
});


describe('TaskCard — the goals card\'s change option asks for the correction (batch 1110 item 2)', () => {
  const ask = {
    id: 'g1', text: '这样理解对吗？', created: '2026-11-05T09:00:00Z', due: null, overdue: false,
    options: ['对', '需要改'], detail: '', change_option: '需要改',
  };

  it('tapping 需要改 prefills the reply and sends nothing', () => {
    const onReply = renderCardWithReply(row({ ask }));
    const buttons = screen.getAllByTestId('wb-card-option');
    const change = buttons.find((b) => b.textContent === '需要改') as HTMLButtonElement;
    expect(change.getAttribute('data-change-option')).toBe('true');
    fireEvent.click(change);
    expect(onReply).not.toHaveBeenCalled();
    expect((screen.getByTestId('wb-card-reply') as HTMLInputElement).value).toBe('需要改：');
  });

  it('the yes option still sends at once', () => {
    const onReply = renderCardWithReply(row({ ask }));
    const yes = screen.getAllByTestId('wb-card-option').find((b) => b.textContent === '对') as HTMLButtonElement;
    expect(yes.getAttribute('data-change-option')).toBeNull();
    fireEvent.click(yes);
    expect(onReply).toHaveBeenCalledTimes(1);
    expect(onReply.mock.calls[0][1]).toBe('对');
  });

  it('an English change option gets a spaced colon', () => {
    renderCardWithReply(row({ ask: { ...ask, options: ['Yes', 'Needs changes'], change_option: 'Needs changes' } }));
    const change = screen.getAllByTestId('wb-card-option').find((b) => b.textContent === 'Needs changes') as HTMLButtonElement;
    fireEvent.click(change);
    expect((screen.getByTestId('wb-card-reply') as HTMLInputElement).value).toBe('Needs changes: ');
  });
});

function renderCardWithReply(r: WorkbenchFeedRow) {
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
  return onReply;
}

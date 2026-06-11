// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import QueueHeader from './QueueHeader';
import type { QueueSnapshot, QueueItem } from '../types';

afterEach(() => cleanup());

function snap(state: QueueSnapshot['state'], opts?: {
  queued?: number;
  paused_until?: string | null;
}): QueueSnapshot {
  const items: QueueItem[] = Array.from({ length: opts?.queued ?? 0 }, (_, i) => ({
    id: `item_${i}`,
    content: `task ${i}`,
    file_refs: [],
    priority: 0,
    review_before_advance: false,
    state: 'queued',
    source: 'user',
    attempts: [],
    idempotency_key: null,
    interrupted_count: 0,
    created_at: '2026-06-11T00:00:00+00:00',
  }));
  return {
    version: 1,
    state,
    items,
    chat_session_id: null,
    paused_until: opts?.paused_until ?? null,
  };
}

describe('QueueHeader', () => {
  it('idle with queued items shows Start wired to onResume', () => {
    const onResume = vi.fn();
    render(
      <QueueHeader snapshot={snap('idle', { queued: 2 })} onStop={vi.fn()} onResume={onResume} />,
    );
    const start = screen.getByTestId('queue-start-btn');
    fireEvent.click(start);
    expect(onResume).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('queue-stop-btn')).toBeNull();
  });

  it('idle with no queued items shows no action button', () => {
    render(
      <QueueHeader snapshot={snap('idle')} onStop={vi.fn()} onResume={vi.fn()} />,
    );
    expect(screen.queryByTestId('queue-start-btn')).toBeNull();
    expect(screen.queryByTestId('queue-stop-btn')).toBeNull();
    expect(screen.queryByTestId('queue-resume-btn')).toBeNull();
  });

  it('running shows Stop which opens the pause menu', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    expect(screen.getByTestId('queue-pause-menu')).toBeTruthy();
    fireEvent.click(screen.getByTestId('queue-pause-until-resume'));
    expect(onStop).toHaveBeenCalledWith(undefined);
  });

  it('pause menu 1-hour option passes 3600 seconds', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    fireEvent.click(screen.getByTestId('queue-pause-1h'));
    expect(onStop).toHaveBeenCalledWith(3600);
  });

  it('pause menu until-tomorrow passes a positive duration', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    fireEvent.click(screen.getByTestId('queue-pause-tomorrow'));
    expect(onStop).toHaveBeenCalledTimes(1);
    const arg = onStop.mock.calls[0][0] as number;
    expect(arg).toBeGreaterThan(0);
    expect(arg).toBeLessThanOrEqual(33 * 3600); // ≤ 33h covers any clock time
  });

  it('paused shows Resume; timed pause shows auto-resume hint', () => {
    const future = new Date(Date.now() + 30 * 60 * 1000).toISOString();
    render(
      <QueueHeader
        snapshot={snap('paused', { queued: 1, paused_until: future })}
        onStop={vi.fn()}
        onResume={vi.fn()}
      />,
    );
    expect(screen.getByTestId('queue-resume-btn')).toBeTruthy();
    expect(screen.getByTestId('queue-autoresume-hint')).toBeTruthy();
  });

  it('Escape closes the pause menu without calling onStop', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    expect(screen.getByTestId('queue-pause-menu')).toBeTruthy();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('queue-pause-menu')).toBeNull();
    expect(onStop).not.toHaveBeenCalled();
  });

  it('untimed pause shows no auto-resume hint', () => {
    render(
      <QueueHeader snapshot={snap('paused', { queued: 1 })} onStop={vi.fn()} onResume={vi.fn()} />,
    );
    expect(screen.queryByTestId('queue-autoresume-hint')).toBeNull();
  });
});

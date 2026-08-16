// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import type { QueueSnapshot } from '../types';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mock useQueue — minimal snapshot so existing queue sections render
// ---------------------------------------------------------------------------
const mockSnapshot: QueueSnapshot = {
  version: 1,
  state: 'idle',
  items: [],
  chat_session_id: null,
};

vi.mock('../hooks/useQueue', () => ({
  useQueue: () => ({
    snapshot: mockSnapshot,
    loading: false,
    error: null,
    addItem: vi.fn(),
    removeItem: vi.fn(),
    stopQueue: vi.fn(),
    resumeQueue: vi.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Mock useWebSocket (used transitively by useQueue)
// ---------------------------------------------------------------------------
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: vi.fn(),
    off: vi.fn(),
    connectionState: 'connected',
    subscribe: vi.fn(),
  }),
}));

import QueueTab from './QueueTab';

describe('QueueTab — Automations moved out of the queue scroll', () => {
  it('no longer renders automations inside the queue', async () => {
    // Item #57: automations are a sibling pane behind the Queue│Automations
    // switch (ProjectDetail), not the Nth section of the queue's own scroll —
    // a create/edit form does not belong under a live streaming queue.
    await act(async () => {
      render(<QueueTab projectId="proj-1" />);
    });

    expect(screen.queryByTestId('queue-section-automations')).toBeNull();
    expect(screen.queryByTestId('automations-pane')).toBeNull();
  });

  it('does NOT remove existing queue section structure', async () => {
    await act(async () => {
      render(<QueueTab projectId="proj-1" />);
    });

    // Existing "Now Running" section — its data-testid is queue-section-now-running
    expect(
      screen.getByTestId('queue-section-now-running'),
    ).toBeInTheDocument();
  });
});

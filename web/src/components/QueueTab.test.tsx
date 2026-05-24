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
  state: 'draining',
  items: [],
  chat_session_id: null,
};

vi.mock('../hooks/useQueue', () => ({
  useQueue: (_projectId: string) => ({
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
// Mock useTriggers — return empty so AutomationsList renders its empty state
// ---------------------------------------------------------------------------
vi.mock('../hooks/useTriggers', () => ({
  useTriggers: (_projectId: string) => ({
    triggers: [],
    loading: false,
    fetchTriggers: vi.fn(async () => []),
    toggleTrigger: vi.fn(),
    deleteTrigger: vi.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Mock useWebSocket (used transitively by useTriggers / useQueue)
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

describe('QueueTab — Automations section', () => {
  it('renders the Automations section alongside the queue sections', async () => {
    await act(async () => {
      render(<QueueTab projectId="proj-1" />);
    });

    // The Automations section must be present
    expect(screen.getByTestId('queue-section-automations')).toBeInTheDocument();

    // AutomationsList empty state is visible (no triggers mocked)
    expect(screen.getByTestId('automations-empty')).toBeInTheDocument();
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

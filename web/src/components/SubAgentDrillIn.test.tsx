// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Task 5 (spec 009 §0.5): the drill-in view replaces the chat message area
// (never a modal) and fetches the sub-agent's transcript read-only. The
// composer is gated on `resumable` from the transcript response: worker
// handles come back non-resumable (disabled, read-only placeholder); cli
// handles come back resumable (enabled composer wired to the @mention inject
// funnel).

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const getSubAgentTranscriptMock = vi.hoisted(() => vi.fn());
const injectMessageMock = vi.hoisted(() => vi.fn());
vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    getSubAgentTranscript: getSubAgentTranscriptMock,
    injectMessage: injectMessageMock,
  }),
}));

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import SubAgentDrillIn from './SubAgentDrillIn';

function workerTranscript() {
  return {
    handle: 'worker:a1b2c3d4-0',
    display_name: 'research task A',
    kind: 'worker' as const,
    resumable: false,
    entries: [
      { source: 'worker:a1b2c3d4-0', content: 'Starting research…', timestamp: '2026-07-01T00:00:00Z', chunk_type: 'response' },
      { source: 'worker:a1b2c3d4-0', content: 'Done. Found 3 sources.', timestamp: '2026-07-01T00:01:00Z', chunk_type: 'response' },
    ],
  };
}

function cliTranscript() {
  return {
    handle: 'claude-code',
    display_name: 'Claude Code',
    kind: 'cli' as const,
    resumable: true,
    entries: [
      { source: 'claude-code', content: 'Ready.', timestamp: '2026-07-01T00:00:00Z', chunk_type: 'response' },
    ],
  };
}

beforeEach(() => {
  getSubAgentTranscriptMock.mockReset();
  injectMessageMock.mockReset();
  apiMock.mockReset();
  apiMock.mockResolvedValue({ agents: [] });
});

describe('SubAgentDrillIn', () => {
  it('renders a disabled, read-only composer footer for a non-resumable (worker) handle', async () => {
    getSubAgentTranscriptMock.mockResolvedValue(workerTranscript());
    render(
      <SubAgentDrillIn
        projectId="p1"
        sessionId="s1"
        handle="worker:a1b2c3d4-0"
        displayName="research task A"
        onBack={() => {}}
      />,
    );

    expect(await screen.findByText('Starting research…')).toBeInTheDocument();
    expect(screen.getByText('Done. Found 3 sources.')).toBeInTheDocument();

    const composer = screen.getByPlaceholderText('Worker task — read-only');
    expect(composer).toBeDisabled();
  });

  it('renders an enabled composer for a resumable (cli) handle, wired to injectMessage with target=handle', async () => {
    getSubAgentTranscriptMock.mockResolvedValue(cliTranscript());
    render(
      <SubAgentDrillIn
        projectId="p1"
        sessionId="s1"
        handle="claude-code"
        displayName="Claude Code"
        onBack={() => {}}
      />,
    );

    expect(await screen.findByText('Ready.')).toBeInTheDocument();

    const composer = await screen.findByTestId('drillin-composer-input');
    expect(composer).toBeEnabled();

    const user = userEvent.setup();
    await user.type(composer, 'follow up question');
    await user.click(screen.getByTestId('drillin-composer-send'));

    await waitFor(() => {
      expect(injectMessageMock).toHaveBeenCalledWith(
        'p1',
        'follow up question',
        'claude-code',
        undefined,
        undefined,
        's1',
      );
    });
  });

  it('the back button fires the onBack callback', async () => {
    getSubAgentTranscriptMock.mockResolvedValue(workerTranscript());
    const onBack = vi.fn();
    render(
      <SubAgentDrillIn
        projectId="p1"
        sessionId="s1"
        handle="worker:a1b2c3d4-0"
        displayName="research task A"
        onBack={onBack}
      />,
    );
    await screen.findByText('Starting research…');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Back' }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it('shows the display name from the fetched transcript in the sticky header', async () => {
    getSubAgentTranscriptMock.mockResolvedValue(workerTranscript());
    render(
      <SubAgentDrillIn
        projectId="p1"
        sessionId="s1"
        handle="worker:a1b2c3d4-0"
        displayName="research task A"
        onBack={() => {}}
      />,
    );
    expect(await screen.findByText('research task A')).toBeInTheDocument();
  });

  // Review round 2, Important 5: the status poll used to run forever (a
  // setInterval created once on mount, never cleared on its own) — every 3s,
  // for as long as the drill-in view stayed open, even long after the task
  // had gone idle. It must stop itself once a tick observes idle.
  it('stops polling /sub-agents/status once the handle goes idle', async () => {
    vi.useFakeTimers();
    try {
      getSubAgentTranscriptMock.mockResolvedValue(workerTranscript());
      apiMock
        .mockResolvedValueOnce({ agents: [{ handle: 'worker:a1b2c3d4-0', status: 'running' }] })
        .mockResolvedValueOnce({ agents: [{ handle: 'worker:a1b2c3d4-0', status: 'idle' }] })
        .mockResolvedValue({ agents: [{ handle: 'worker:a1b2c3d4-0', status: 'running' }] });

      render(
        <SubAgentDrillIn
          projectId="p1"
          sessionId="s1"
          handle="worker:a1b2c3d4-0"
          displayName="research task A"
          onBack={() => {}}
        />,
      );
      await vi.advanceTimersByTimeAsync(0); // flush the mount-time transcript fetch

      await vi.advanceTimersByTimeAsync(3000); // tick #1: running
      await vi.advanceTimersByTimeAsync(3000); // tick #2: idle -> poll stops itself
      expect(apiMock).toHaveBeenCalledTimes(2);

      await vi.advanceTimersByTimeAsync(3000 * 5); // plenty more time passes
      expect(apiMock).toHaveBeenCalledTimes(2); // unchanged — no more ticks
    } finally {
      vi.useRealTimers();
    }
  });
});

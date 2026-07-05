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

import { act, render, screen, waitFor } from '@testing-library/react';
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

// Live streaming (round 3): the drill-in subscribes to chat.stream_delta /
// fanout.task_update via useWebSocket. Handler registry + emitWs mirror
// ChatView.test.tsx's mocking pattern in miniature.
const wsHandlers = vi.hoisted(() => new Map<string, Set<(e: unknown) => void>>());
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: (type: string, fn: (e: unknown) => void) => {
      if (!wsHandlers.has(type)) wsHandlers.set(type, new Set());
      wsHandlers.get(type)!.add(fn);
    },
    off: (type: string, fn: (e: unknown) => void) => {
      wsHandlers.get(type)?.delete(fn);
    },
    connectionState: 'connected',
  }),
}));

function emitWs(type: string, event: unknown) {
  wsHandlers.get(type)?.forEach((fn) => fn(event));
}

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

// Round 2 (spec 009 §0.5, Task D, issues 2+3): a worker transcript whose
// session_uuid is discoverable renders chat-shaped (via transformChatHistory
// + ChatMessage) instead of the flat entries above.
function workerTranscriptWithSession() {
  return {
    handle: 'worker:a1b2c3d4-0',
    display_name: 'research task A',
    kind: 'worker' as const,
    resumable: false,
    session_uuid: 'sess-worker-uuid-1',
    entries: [
      { source: 'worker:a1b2c3d4-0', content: 'Starting research…', timestamp: '2026-07-01T00:00:00Z', chunk_type: 'response' },
    ],
  };
}

function workerChatMessages() {
  return [
    { role: 'user', content: 'Research topic X', source: 'user', timestamp: '2026-07-01T00:00:00Z' },
    { role: 'assistant', content: 'Done. Found 3 sources.', source: 'management', timestamp: '2026-07-01T00:01:00Z' },
  ];
}

function workerChatMessagesWithToolCall() {
  return [
    { role: 'user', content: 'Research topic X', source: 'user', timestamp: '2026-07-01T00:00:00Z' },
    {
      role: 'assistant',
      content: '',
      source: 'management',
      timestamp: '2026-07-01T00:00:30Z',
      tool_calls: [{ id: 'tc1', type: 'function' as const, function: { name: 'grep', arguments: '{}' } }],
    },
    { role: 'tool', content: 'match found', source: 'tool', timestamp: '2026-07-01T00:00:45Z', tool_call_id: 'tc1' },
    { role: 'assistant', content: 'Done. Found 3 sources.', source: 'management', timestamp: '2026-07-01T00:01:00Z' },
  ];
}

beforeEach(() => {
  getSubAgentTranscriptMock.mockReset();
  injectMessageMock.mockReset();
  apiMock.mockReset();
  apiMock.mockResolvedValue({ agents: [] });
  wsHandlers.clear();
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

  // Round 2 (spec 009 §0.5, Task D, issues 2+3): a worker whose transcript
  // carries a discoverable session_uuid renders its own chat history
  // chat-shaped — via transformChatHistory + the real ChatMessage component —
  // instead of the flat transcript entries. Fixes "(no transcript yet)" for
  // running workers and the mismatched-rendering complaint (issue 3).
  describe('chat-shaped worker rendering (round 2, Task D)', () => {
    it('renders a user brief bubble + assistant reply from the mocked chat payload when session_uuid is present', async () => {
      getSubAgentTranscriptMock.mockResolvedValue(workerTranscriptWithSession());
      apiMock.mockImplementation((url: string) => {
        if (url.includes('/chat?')) return Promise.resolve(workerChatMessages());
        return Promise.resolve({ agents: [] });
      });

      render(
        <SubAgentDrillIn
          projectId="p1"
          sessionId="s1"
          handle="worker:a1b2c3d4-0"
          displayName="research task A"
          onBack={() => {}}
        />,
      );

      expect(await screen.findByText('Research topic X')).toBeInTheDocument();
      expect(await screen.findByText('Done. Found 3 sources.')).toBeInTheDocument();
      // Chat-shaped rendering replaces the flat entries rendering entirely.
      expect(screen.queryByTestId('drillin-entry')).not.toBeInTheDocument();

      // Composer stays disabled for workers regardless of rendering path
      // (resumable logic is untouched by this task).
      const composer = screen.getByPlaceholderText('Worker task — read-only');
      expect(composer).toBeDisabled();

      expect(apiMock).toHaveBeenCalledWith(expect.stringContaining('/api/v2/agents/p1/chat?session_id=sess-worker-uuid-1'));
    });

    it('renders a compact read-only capsule for agent_run items in the chat-shaped view', async () => {
      getSubAgentTranscriptMock.mockResolvedValue(workerTranscriptWithSession());
      apiMock.mockImplementation((url: string) => {
        if (url.includes('/chat?')) return Promise.resolve(workerChatMessagesWithToolCall());
        return Promise.resolve({ agents: [] });
      });

      render(
        <SubAgentDrillIn
          projectId="p1"
          sessionId="s1"
          handle="worker:a1b2c3d4-0"
          displayName="research task A"
          onBack={() => {}}
        />,
      );

      expect(await screen.findByText('Done. Found 3 sources.')).toBeInTheDocument();
      expect(screen.getByTestId('drillin-capsule')).toBeInTheDocument();
    });

    it('falls back to entries rendering when the transcript carries no session_uuid (old fanouts)', async () => {
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
      expect(screen.getAllByTestId('drillin-entry')).toHaveLength(2);
      // No worker chat fetch attempted — session_uuid is absent.
      expect(apiMock).not.toHaveBeenCalledWith(expect.stringContaining('/chat?'));
    });

    it('ignores a session_uuid on a CLI handle — only kind==="worker" gets chat-shaped rendering', async () => {
      getSubAgentTranscriptMock.mockResolvedValue({ ...cliTranscript(), session_uuid: 'sess-should-be-ignored' });

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
      expect(apiMock).not.toHaveBeenCalledWith(expect.stringContaining('/chat?'));
    });
  });
  // Round 3: live streaming. Worker chat.stream_delta events addressed to the
  // worker's session_uuid accumulate into a live bubble; the final delta (or a
  // terminal fanout.task_update for this handle) drops the buffer and refetches
  // so the persisted message supersedes it without duplication.
  describe('live worker streaming (round 3)', () => {
    async function renderStreamingWorker() {
      getSubAgentTranscriptMock.mockResolvedValue(workerTranscriptWithSession());
      apiMock.mockImplementation((url: string) => {
        if (url.includes('/chat?')) return Promise.resolve(workerChatMessages());
        return Promise.resolve({ agents: [] });
      });
      render(
        <SubAgentDrillIn
          projectId="p1"
          sessionId="s1"
          handle="worker:a1b2c3d4-0"
          displayName="research task A"
          onBack={() => {}}
        />,
      );
      await screen.findByText('Research topic X');
    }

    it('accumulates deltas for the worker session into a live bubble', async () => {
      await renderStreamingWorker();
      act(() => {
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'sess-worker-uuid-1', text: 'Live toke', is_final: false, seq: 1,
        });
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'sess-worker-uuid-1', text: 'ns', is_final: false, seq: 2,
        });
      });
      expect(await screen.findByTestId('drillin-live-stream')).toBeInTheDocument();
      expect(screen.getByText(/Live tokens/)).toBeInTheDocument();
    });

    it('ignores deltas addressed to a different session', async () => {
      await renderStreamingWorker();
      act(() => {
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'some-other-session', text: 'LEAKED', is_final: false, seq: 1,
        });
      });
      expect(screen.queryByText(/LEAKED/)).not.toBeInTheDocument();
      expect(screen.queryByTestId('drillin-live-stream')).not.toBeInTheDocument();
    });

    it('clears the live bubble on the final delta and refetches the persisted history', async () => {
      await renderStreamingWorker();
      act(() => {
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'sess-worker-uuid-1', text: 'Almost done', is_final: false, seq: 1,
        });
      });
      expect(await screen.findByTestId('drillin-live-stream')).toBeInTheDocument();
      const fetchesBefore = getSubAgentTranscriptMock.mock.calls.length;
      act(() => {
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'sess-worker-uuid-1', text: '', is_final: true, seq: 2,
        });
      });
      await waitFor(() =>
        expect(screen.queryByTestId('drillin-live-stream')).not.toBeInTheDocument(),
      );
      await waitFor(() =>
        expect(getSubAgentTranscriptMock.mock.calls.length).toBeGreaterThan(fetchesBefore),
      );
    });

    it('a terminal fanout.task_update for this handle clears the bubble and refetches', async () => {
      await renderStreamingWorker();
      act(() => {
        emitWs('chat.stream_delta', {
          type: 'chat.stream_delta', project_id: 'p1',
          session_id: 'sess-worker-uuid-1', text: 'partial', is_final: false, seq: 1,
        });
      });
      expect(await screen.findByTestId('drillin-live-stream')).toBeInTheDocument();
      const fetchesBefore = getSubAgentTranscriptMock.mock.calls.length;
      act(() => {
        emitWs('fanout.task_update', {
          type: 'fanout.task_update', project_id: 'p1', session_id: 's1',
          fanout_id: 'a1b2c3d4', handle: 'worker:a1b2c3d4-0', status: 'completed',
        });
      });
      await waitFor(() =>
        expect(screen.queryByTestId('drillin-live-stream')).not.toBeInTheDocument(),
      );
      await waitFor(() =>
        expect(getSubAgentTranscriptMock.mock.calls.length).toBeGreaterThan(fetchesBefore),
      );
    });
  });
});

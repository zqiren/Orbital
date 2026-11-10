// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Piece 3 Part D unit tests: badge states, stop control + honest warning.
// (The RULE-4 Playwright spec covers the same surface at 375x667 against the
// isolated-daemon harness; these run headlessly everywhere.)

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));
vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({ on: vi.fn(), off: vi.fn() }),
}));

import SubAgentStatusBar from './SubAgentStatusBar';

function statusPayload(status: string, commands: string[] = []) {
  return {
    session_id: 's1',
    agents: [{
      handle: 'claude-code',
      display_name: 'Claude Code',
      status,
      background_commands: commands,
    }],
  };
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('SubAgentStatusBar — three-state badge', () => {
  it('renders Working for an open turn', async () => {
    apiMock.mockResolvedValue(statusPayload('running'));
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    const badge = await screen.findByTestId('sub-agent-badge-claude-code');
    expect(badge).toHaveTextContent('Working');
    expect(badge.dataset.status).toBe('running');
    expect(screen.getByTestId('sub-agent-stop-claude-code')).toBeInTheDocument();
  });

  it('renders Background for turn-done-but-work-alive (the live-bug shape)', async () => {
    apiMock.mockResolvedValue(
      statusPayload('background-running', ['gh api search loop']));
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    const badge = await screen.findByTestId('sub-agent-badge-claude-code');
    expect(badge).toHaveTextContent('Background');
    expect(badge.dataset.status).toBe('background-running');
  });

  it('renders Idle with NO stop control (nothing real to stop)', async () => {
    apiMock.mockResolvedValue(statusPayload('idle'));
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    const badge = await screen.findByTestId('sub-agent-badge-claude-code');
    expect(badge).toHaveTextContent('Idle');
    expect(screen.queryByTestId('sub-agent-stop-claude-code')).toBeNull();
  });

  // Spec 145: the dot is a state colour at readable contrast, not the
  // agent's brand tint; running adds the pin switcher's moving ring.
  it.each([
    ['running', 'bg-success-text'],
    ['background-running', 'bg-warning-text'],
    ['idle', 'bg-idle'],
  ])('%s dot uses %s', async (status, cls) => {
    apiMock.mockResolvedValue(statusPayload(status));
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    const dot = await screen.findByTestId('sub-agent-dot-claude-code');
    expect(dot).toHaveClass(cls);
    expect(dot.className).not.toMatch(/agent-color|animate-pulse|bg-border\b/);
    expect(dot).toHaveAttribute('aria-hidden', 'true');
  });

  it('running dot carries the moving ring in its own box; other states do not', async () => {
    apiMock.mockResolvedValue(statusPayload('running'));
    const { unmount } = render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    const ring = await screen.findByTestId('sub-agent-ring-claude-code');
    expect(ring).toHaveClass('motion-safe:animate-ping', 'border-success-text');
    // The ring sits inside the dot's box, not on the badge or the chip, so
    // its growth never reaches the agent name.
    const box = screen.getByTestId('sub-agent-dot-claude-code').parentElement!;
    expect(box).toContainElement(ring);
    expect(box).toHaveClass('relative');
    unmount();

    apiMock.mockResolvedValue(statusPayload('background-running'));
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    await screen.findByTestId('sub-agent-dot-claude-code');
    expect(screen.queryByTestId('sub-agent-ring-claude-code')).toBeNull();
  });

  it('filters out fanout worker handles (they render in FanoutCard instead)', async () => {
    apiMock.mockResolvedValue({
      session_id: 's1',
      agents: [
        { handle: 'claude-code', display_name: 'Claude Code', status: 'running', background_commands: [] },
        { handle: 'worker:a1b2c3d4-0', display_name: 'research task A', status: 'running', background_commands: [] },
      ],
    });
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);
    await screen.findByTestId('sub-agent-chip-claude-code');
    expect(screen.queryByTestId('sub-agent-chip-worker:a1b2c3d4-0')).toBeNull();
  });

  it('renders nothing when no sub-agents are active', async () => {
    apiMock.mockResolvedValue({ session_id: 's1', agents: [] });
    const { container } = render(
      <SubAgentStatusBar projectId="p1" sessionId="s1" />);
    await waitFor(() => expect(apiMock).toHaveBeenCalled());
    expect(container.querySelector('[data-testid="sub-agent-status-bar"]')).toBeNull();
  });
});

describe('SubAgentStatusBar — needs you (awaiting approval)', () => {
  const live = (agents: unknown[]) =>
    ({ agents, refresh: vi.fn(async () => {}) }) as unknown as Parameters<typeof SubAgentStatusBar>[0]['live'];
  const base = { handle: 'claude-code', display_name: 'Claude Code', background_commands: [] as string[] };

  it('a running turn waiting on approval reads Needs you, with a still amber dot and the stop control', () => {
    render(<SubAgentStatusBar projectId="p1" sessionId="s1"
      live={live([{ ...base, status: 'running', awaiting_approval: true }])} />);
    const badge = screen.getByTestId('sub-agent-badge-claude-code');
    expect(badge).toHaveTextContent('Needs you');
    expect(badge.dataset.status).toBe('running');
    expect(badge.dataset.state).toBe('needs-you');
    expect(badge).toHaveClass('text-warning');
    expect(screen.getByTestId('sub-agent-dot-claude-code')).toHaveClass('bg-warning-text');
    // The moving ring means producing; waiting has none.
    expect(screen.queryByTestId('sub-agent-ring-claude-code')).toBeNull();
    expect(screen.getByTestId('sub-agent-stop-claude-code')).toBeInTheDocument();
  });

  it('awaiting_approval false or missing (an older daemon) reads Working', () => {
    const { unmount } = render(<SubAgentStatusBar projectId="p1" sessionId="s1"
      live={live([{ ...base, status: 'running', awaiting_approval: false }])} />);
    expect(screen.getByTestId('sub-agent-badge-claude-code')).toHaveTextContent('Working');
    unmount();
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([{ ...base, status: 'running' }])} />);
    expect(screen.getByTestId('sub-agent-badge-claude-code')).toHaveTextContent('Working');
    expect(screen.getByTestId('sub-agent-ring-claude-code')).toBeInTheDocument();
  });

  it('a status this build does not know renders as idle, without crashing', () => {
    render(<SubAgentStatusBar projectId="p1" sessionId="s1"
      live={live([{ ...base, status: 'paused-for-something-new' }, { ...base, handle: 'codex', status: undefined }])} />);
    expect(screen.getByTestId('sub-agent-badge-claude-code')).toHaveTextContent('Idle');
    expect(screen.getByTestId('sub-agent-dot-claude-code')).toHaveClass('bg-idle');
    expect(screen.getByTestId('sub-agent-badge-codex')).toHaveTextContent('Idle');
    expect(screen.queryByTestId('sub-agent-stop-claude-code')).toBeNull();
  });
});

describe('SubAgentStatusBar — stop flow with honest warning', () => {
  it('background-running stop shows the commands AND the raw-detach warning, then POSTs', async () => {
    apiMock.mockImplementation(async (_path: string, options?: RequestInit) => {
      if (options?.method === 'POST') {
        return {
          handle: 'claude-code', status: 'stopped',
          background_terminated: ['gh api search loop'], warning: 'w',
        };
      }
      return statusPayload('background-running', ['gh api search loop']);
    });
    const user = userEvent.setup();
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);

    await user.click(await screen.findByTestId('sub-agent-stop-claude-code'));

    // Honest dialog: terminated work listed + the limitation named.
    expect(screen.getByTestId('sub-agent-stop-confirm')).toBeInTheDocument();
    expect(screen.getByText('gh api search loop')).toBeInTheDocument();
    const warning = screen.getByTestId('sub-agent-stop-warning');
    expect(warning.textContent).toMatch(/nohup/);
    expect(warning.textContent).toMatch(/cannot be guaranteed stopped/);

    await user.click(screen.getByTestId('sub-agent-stop-confirm-button'));

    await waitFor(() => {
      const stopCall = apiMock.mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'POST');
      expect(stopCall?.[0]).toBe(
        '/api/v2/agents/p1/sub-agents/claude-code/stop?session_id=s1');
    });
    // Outcome surfaced.
    const result = await screen.findByTestId('sub-agent-stop-result');
    expect(result.textContent).toMatch(/1 background process/);
  });

  it('cancel leaves the agent untouched', async () => {
    apiMock.mockResolvedValue(statusPayload('running'));
    const user = userEvent.setup();
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" />);

    await user.click(await screen.findByTestId('sub-agent-stop-claude-code'));
    await user.click(screen.getByTestId('sub-agent-stop-cancel'));

    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
    const postCalls = apiMock.mock.calls.filter(
      (c) => (c[1] as RequestInit | undefined)?.method === 'POST');
    expect(postCalls).toHaveLength(0);
  });
});

describe('SubAgentStatusBar — spec 169 C: the stop dialog', () => {
  const agent = (status: 'running' | 'background-running' | 'idle') => ({
    handle: 'claude-code', display_name: 'Claude Code', status, background_commands: [] as string[],
  });
  const live = (agents: ReturnType<typeof agent>[]) => ({ agents, refresh: vi.fn(async () => {}) });

  it('renders under document.body, outside the chip line (which the app window clips)', async () => {
    const user = userEvent.setup();
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    await user.click(screen.getByTestId('sub-agent-stop-claude-code'));
    const dialog = screen.getByTestId('sub-agent-stop-confirm');
    expect(dialog.parentElement).toBe(document.body);
    expect(screen.getByTestId('sub-agent-status-bar').contains(dialog)).toBe(false);
    // Copy and theme context unchanged through the portal.
    expect(dialog).toHaveTextContent('Stop Claude Code?');
  });

  it('a click inside the panel does not dismiss it; the backdrop does', async () => {
    const user = userEvent.setup();
    render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    await user.click(screen.getByTestId('sub-agent-stop-claude-code'));
    await user.click(screen.getByTestId('sub-agent-stop-warning'));
    expect(screen.getByTestId('sub-agent-stop-confirm')).toBeInTheDocument();
    await user.click(screen.getByTestId('sub-agent-stop-confirm'));
    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
  });

  it('closes when the confirmed agent turns idle', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    await user.click(screen.getByTestId('sub-agent-stop-claude-code'));
    expect(screen.getByTestId('sub-agent-stop-confirm')).toBeInTheDocument();
    rerender(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('idle')])} />);
    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
    // And it does not come back when the agent runs again.
    rerender(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
  });

  it('closes when the confirmed agent leaves the line', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    await user.click(screen.getByTestId('sub-agent-stop-claude-code'));
    rerender(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([])} />);
    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
    rerender(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    expect(screen.queryByTestId('sub-agent-stop-confirm')).toBeNull();
  });

  it('stays open while the agent moves from running to background', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('running')])} />);
    await user.click(screen.getByTestId('sub-agent-stop-claude-code'));
    rerender(<SubAgentStatusBar projectId="p1" sessionId="s1" live={live([agent('background-running')])} />);
    expect(screen.getByTestId('sub-agent-stop-confirm')).toBeInTheDocument();
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Merged Project-Settings sub-agent card: per installed agent, one card with
// an enable toggle + auth badge + memory summary in the header, and the
// MEMORY.md editor as an expandable body. Replaces the two separate
// "Sub-Agents" (toggle list) and "Sub-Agent Memories" (memory card list)
// blocks.

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({
  api: apiMock,
  ApiError: class ApiError extends Error {
    detail: string;
    constructor(detail: string) {
      super(detail);
      this.detail = detail;
    }
  },
}));

import SubAgentCard from './SubAgentCard';
import type { SubAgentMemoryEntry } from './SubAgentMemoryCard';

function memEntry(
  overrides: Partial<SubAgentMemoryEntry> = {},
): SubAgentMemoryEntry {
  return {
    agent_slug: 'claude-code',
    agent_name: 'Claude Code',
    exists: false,
    content: '',
    last_modified: null,
    size_bytes: 0,
    ...overrides,
  };
}

beforeEach(() => {
  apiMock.mockReset();
});

describe('SubAgentCard — merged toggle + memory', () => {
  it('renders the agent name and slug in the header', () => {
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Claude Code"
        slug="claude-code"
        enabled
        ready
        memory={memEntry()}
        onToggle={() => {}}
        onMemorySaved={() => {}}
      />,
    );
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.getByText('claude-code')).toBeInTheDocument();
  });

  it('the enable checkbox reflects enabled state and calls onToggle when clicked', () => {
    const onToggle = vi.fn();
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Claude Code"
        slug="claude-code"
        enabled
        ready
        memory={memEntry()}
        onToggle={onToggle}
        onMemorySaved={() => {}}
      />,
    );
    const checkbox = screen.getByRole('checkbox', { name: /Claude Code/ });
    expect(checkbox).toBeChecked();
    fireEvent.click(checkbox);
    expect(onToggle).toHaveBeenCalledWith('claude-code', false);
  });

  it('disabled agent: checkbox unchecked, card is dimmed, body still expandable', () => {
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Gemini CLI"
        slug="gemini-cli"
        enabled={false}
        ready
        memory={memEntry({ agent_slug: 'gemini-cli', agent_name: 'Gemini CLI' })}
        onToggle={() => {}}
        onMemorySaved={() => {}}
      />,
    );
    expect(screen.getByRole('checkbox', { name: /Gemini CLI/ })).not.toBeChecked();
    // Expand body — the MEMORY.md editor textarea must appear even when disabled.
    fireEvent.click(screen.getByRole('button', { name: /memory/i }));
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('shows a "Sign in" auth badge when installed but not ready', () => {
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Codex"
        slug="codex"
        enabled
        ready={false}
        memory={memEntry({ agent_slug: 'codex', agent_name: 'Codex' })}
        onToggle={() => {}}
        onMemorySaved={() => {}}
      />,
    );
    expect(screen.getByText(/sign in/i)).toBeInTheDocument();
  });

  it('does NOT show the auth badge when the agent is ready', () => {
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Claude Code"
        slug="claude-code"
        enabled
        ready
        memory={memEntry()}
        onToggle={() => {}}
        onMemorySaved={() => {}}
      />,
    );
    expect(screen.queryByText(/sign in/i)).not.toBeInTheDocument();
  });

  it('expanding reveals the MEMORY.md editor with Save and Reset', () => {
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Claude Code"
        slug="claude-code"
        enabled
        ready
        memory={memEntry({ exists: true, content: 'hi', size_bytes: 2 })}
        onToggle={() => {}}
        onMemorySaved={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /memory/i }));
    expect(screen.getByRole('textbox')).toHaveValue('hi');
    expect(screen.getByRole('button', { name: /^save$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeInTheDocument();
  });

  it('saving the memory body PUTs to the memory endpoint and calls onMemorySaved', async () => {
    apiMock.mockResolvedValue({ size_bytes: 5 });
    const onMemorySaved = vi.fn();
    render(
      <SubAgentCard
        projectId="p1"
        agentName="Claude Code"
        slug="claude-code"
        enabled
        ready
        memory={memEntry()}
        onToggle={() => {}}
        onMemorySaved={onMemorySaved}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /memory/i }));
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'note' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^save$/i }));
    await waitFor(() => {
      expect(apiMock).toHaveBeenCalledWith(
        '/api/v2/projects/p1/sub-agent-memory/claude-code',
        expect.objectContaining({ method: 'PUT' }),
      );
    });
    await waitFor(() => expect(onMemorySaved).toHaveBeenCalled());
  });
});

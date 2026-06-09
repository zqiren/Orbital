// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SubAgentToggleList from './SubAgentToggleList';

const AGENTS = [
  { slug: 'claude-code', name: 'Claude Code' },
  { slug: 'gemini-cli', name: 'Gemini CLI' },
];

describe('SubAgentToggleList', () => {
  it('renders each installed sub-agent with a checkbox', () => {
    render(<SubAgentToggleList agents={AGENTS} disabled={[]} onToggle={() => {}} />);
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.getByText('Gemini CLI')).toBeInTheDocument();
  });

  it('checkbox is checked for enabled agents and unchecked for disabled ones', () => {
    render(
      <SubAgentToggleList agents={AGENTS} disabled={['gemini-cli']} onToggle={() => {}} />,
    );
    expect(screen.getByRole('checkbox', { name: /Claude Code/ })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /Gemini CLI/ })).not.toBeChecked();
  });

  it('toggling a disabled agent ON calls onToggle(slug, true)', () => {
    const onToggle = vi.fn();
    render(
      <SubAgentToggleList agents={AGENTS} disabled={['gemini-cli']} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole('checkbox', { name: /Gemini CLI/ }));
    expect(onToggle).toHaveBeenCalledWith('gemini-cli', true);
  });

  it('toggling an enabled agent OFF calls onToggle(slug, false)', () => {
    const onToggle = vi.fn();
    render(<SubAgentToggleList agents={AGENTS} disabled={[]} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole('checkbox', { name: /Claude Code/ }));
    expect(onToggle).toHaveBeenCalledWith('claude-code', false);
  });

  it('shows an empty-state message when no sub-agents are installed', () => {
    render(<SubAgentToggleList agents={[]} disabled={[]} onToggle={() => {}} />);
    expect(screen.getByText(/no sub-agents installed/i)).toBeInTheDocument();
  });
});

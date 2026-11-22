// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SubAgentToggleList from './SubAgentToggleList';

const AGENTS = [
  { slug: 'claude-code', name: 'Claude Code' },
  { slug: 'antigravity', name: 'Antigravity' },
];

describe('SubAgentToggleList', () => {
  it('renders each installed sub-agent with a checkbox', () => {
    render(<SubAgentToggleList agents={AGENTS} disabled={[]} onToggle={() => {}} />);
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.getByText('Antigravity')).toBeInTheDocument();
  });

  it('checkbox is checked for enabled agents and unchecked for disabled ones', () => {
    render(
      <SubAgentToggleList agents={AGENTS} disabled={['antigravity']} onToggle={() => {}} />,
    );
    expect(screen.getByRole('checkbox', { name: /Claude Code/ })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /Antigravity/ })).not.toBeChecked();
  });

  it('toggling a disabled agent ON calls onToggle(slug, true)', () => {
    const onToggle = vi.fn();
    render(
      <SubAgentToggleList agents={AGENTS} disabled={['antigravity']} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole('checkbox', { name: /Antigravity/ }));
    expect(onToggle).toHaveBeenCalledWith('antigravity', true);
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

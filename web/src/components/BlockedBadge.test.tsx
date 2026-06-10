// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mock useBlockedCount — control return value per test
// ---------------------------------------------------------------------------
const mockUseBlockedCount = vi.fn();

vi.mock('../hooks/useBlockedCount', () => ({
  useBlockedCount: () => mockUseBlockedCount(),
}));

import BlockedBadge from './BlockedBadge';

describe('BlockedBadge', () => {
  it('renders nothing while loading', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: true });
    const { container } = render(<BlockedBadge />);
    expect(container.firstChild).toBeNull();
  });

  it('renders "Blocked" label when count is 0', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    expect(screen.getByText('Blocked')).toBeInTheDocument();
  });

  it('hides the count pill when blockedCount is 0', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    expect(screen.queryByTestId('blocked-badge-pill')).toBeNull();
  });

  it('shows the count pill with correct value when blockedCount > 0', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 3, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    const pill = screen.getByTestId('blocked-badge-pill');
    expect(pill).toBeInTheDocument();
    expect(pill).toHaveTextContent('3');
  });

  it('renders with the warning design token on the pill when count > 0', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 2, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    const pill = screen.getByTestId('blocked-badge-pill');
    // Uses the Tailwind warning token (not hardcoded hex) — text-warning + bg-warning/10
    expect(pill.className).toContain('text-warning');
    expect(pill.className).toContain('bg-warning/10');
  });

  it('has correct aria-label with count when blockedCount > 0', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 5, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    expect(screen.getByRole('region')).toHaveAttribute(
      'aria-label',
      '5 sessions blocked across all projects',
    );
  });

  it('has correct singular aria-label when blockedCount is 1', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 1, blockedSessions: [], loading: false });
    render(<BlockedBadge />);
    expect(screen.getByRole('region')).toHaveAttribute(
      'aria-label',
      '1 session blocked across all projects',
    );
  });

  it('updates pill count when hook value increments (re-render simulation)', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 2, blockedSessions: [], loading: false });
    const { rerender } = render(<BlockedBadge />);
    expect(screen.getByTestId('blocked-badge-pill')).toHaveTextContent('2');

    // Simulate blocked-count-changed event causing hook to return higher value
    mockUseBlockedCount.mockReturnValue({ blockedCount: 4, blockedSessions: [], loading: false });
    rerender(<BlockedBadge />);
    expect(screen.getByTestId('blocked-badge-pill')).toHaveTextContent('4');
  });

  it('hides pill when count decrements back to 0 (re-render simulation)', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 3, blockedSessions: [], loading: false });
    const { rerender } = render(<BlockedBadge />);
    expect(screen.getByTestId('blocked-badge-pill')).toBeInTheDocument();

    // Simulate all sessions unblocked
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
    rerender(<BlockedBadge />);
    expect(screen.queryByTestId('blocked-badge-pill')).toBeNull();
  });

  it('shows pill again when count increments from 0 (re-render simulation)', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 0, blockedSessions: [], loading: false });
    const { rerender } = render(<BlockedBadge />);
    expect(screen.queryByTestId('blocked-badge-pill')).toBeNull();

    // Simulate a new blocked session arriving
    mockUseBlockedCount.mockReturnValue({ blockedCount: 1, blockedSessions: [], loading: false });
    rerender(<BlockedBadge />);
    expect(screen.getByTestId('blocked-badge-pill')).toHaveTextContent('1');
  });

  it('calls onClick when provided and the badge is clicked', () => {
    mockUseBlockedCount.mockReturnValue({ blockedCount: 2, blockedSessions: [], loading: false });
    const onClick = vi.fn();
    render(<BlockedBadge onClick={onClick} />);
    screen.getByRole('region').click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});

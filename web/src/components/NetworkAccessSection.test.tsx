// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * NetworkAccessSection unit tests — TOFU network allowlist Plan 2 Task 7.
 *
 * Covers the section's real logic (dedupe, pending→approved promotion):
 *  - removing an approved domain drops it from the list;
 *  - approving a pending request promotes it into approvedDomains and
 *    removes it from pendingRequests;
 *  - dismissing a pending request drops it from pendingRequests without
 *    touching approvedDomains.
 */

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { NetworkAccessSection } from './NetworkAccessSection';

const pending = [{ domain: 'x.com', reason: 'verify handles', requested_at: '2026-07-10T17:02:00Z' }];

describe('NetworkAccessSection', () => {
  it('renders approved domains with remove buttons', () => {
    const onChange = vi.fn();
    render(<NetworkAccessSection approvedDomains={['api.stripe.com']} pendingRequests={[]} onChange={onChange} />);
    expect(screen.getByText('api.stripe.com')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /remove api\.stripe\.com/i }));
    expect(onChange).toHaveBeenCalledWith({ approvedDomains: [], pendingRequests: [] });
  });

  it('approving a pending request promotes it', () => {
    const onChange = vi.fn();
    render(<NetworkAccessSection approvedDomains={[]} pendingRequests={pending} onChange={onChange} />);
    fireEvent.click(screen.getByRole('button', { name: /approve x\.com/i }));
    expect(onChange).toHaveBeenCalledWith({ approvedDomains: ['x.com'], pendingRequests: [] });
  });

  it('dismissing a pending request drops it without approving', () => {
    const onChange = vi.fn();
    render(<NetworkAccessSection approvedDomains={[]} pendingRequests={pending} onChange={onChange} />);
    fireEvent.click(screen.getByRole('button', { name: /dismiss x\.com/i }));
    expect(onChange).toHaveBeenCalledWith({ approvedDomains: [], pendingRequests: [] });
  });
});

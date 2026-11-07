// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for PendingInputNotice — pending-input queue (spec 006 · v3) §11e/§12.
 *
 * A message sent to a session whose own turn is running waits in that
 * session's queue. Spec 155 removed the cross-session kind: no "Waiting for
 * other sessions" line and no [Run now] — every session runs when sent to.
 * The waiting line is a tappable edit affordance (mobile equivalent of ↑),
 * labelled `pending.edit`, disabled when `canEdit===false` (attachments).
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import PendingInputNotice from './PendingInputNotice';

describe('PendingInputNotice', () => {
  it('renders the same-session waiting copy and no Run now', () => {
    render(<PendingInputNotice onEdit={vi.fn().mockResolvedValue(undefined)} />);
    const body = screen.getByTestId('pending-input-notice-body');
    expect(body).toHaveTextContent('Waiting for the current response to finish.');
    expect(screen.queryByText('Waiting for other sessions to finish.')).toBeNull();
    expect(screen.queryByTestId('pending-input-notice-run-now')).toBeNull();
    expect(screen.queryByTestId('pending-input-notice-stop')).toBeNull();
  });

  it('tapping the waiting line invokes onEdit (tap-to-edit; mobile ≈ ↑)', async () => {
    const onEdit = vi.fn().mockResolvedValue(undefined);
    render(<PendingInputNotice onEdit={onEdit} />);
    const edit = screen.getByTestId('pending-input-notice-edit');
    expect(edit).toHaveAttribute('aria-label', 'Edit queued message');
    await userEvent.click(edit);
    await waitFor(() => expect(onEdit).toHaveBeenCalledTimes(1));
  });

  it('disables tap-to-edit when canEdit is false (entry has attachments)', async () => {
    const onEdit = vi.fn().mockResolvedValue(undefined);
    render(<PendingInputNotice canEdit={false} onEdit={onEdit} />);
    const edit = screen.getByTestId('pending-input-notice-edit');
    expect(edit).toBeDisabled();
    await userEvent.click(edit);
    expect(onEdit).not.toHaveBeenCalled();
  });

  it('surfaces an inline error if onEdit rejects, and stays mounted', async () => {
    const onEdit = vi.fn().mockRejectedValue(new Error('boom'));
    render(<PendingInputNotice onEdit={onEdit} />);
    await userEvent.click(screen.getByTestId('pending-input-notice-edit'));
    await waitFor(() => {
      expect(screen.getByTestId('pending-input-notice-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('pending-input-notice')).toBeInTheDocument();
  });
});

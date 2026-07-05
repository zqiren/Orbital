// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for PendingInputNotice — pending-input queue (spec 006 · v3) §11e/§12.
 *
 * v3 contract changes from v2:
 *   - NO [Stop waiting] button (cancel/edit now happens via ↑ / tap-to-edit).
 *   - [Run now] renders ONLY for `kind === 'cross'` (slot held by another
 *     session). Same-session entries drain automatically — no Run-now.
 *   - Copy is placeholder-free: crossSession vs sameSession lines (no `{holder}`).
 *   - The waiting line is a tappable edit affordance (mobile equivalent of ↑)
 *     wired to `onEdit`, labelled `pending.edit`. Disabled when `canEdit===false`
 *     (the entry has attachments — recall can't restore chips).
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import PendingInputNotice from './PendingInputNotice';

describe('PendingInputNotice', () => {
  it('renders the cross-session waiting copy (placeholder-free, no holder id)', () => {
    render(
      <PendingInputNotice
        kind="cross"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onEdit={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('pending-input-notice')).toBeInTheDocument();
    const body = screen.getByTestId('pending-input-notice-body');
    expect(body).toHaveTextContent('Waiting for other sessions to finish.');
  });

  it('renders the same-session waiting copy', () => {
    render(<PendingInputNotice kind="same" onEdit={vi.fn().mockResolvedValue(undefined)} />);
    const body = screen.getByTestId('pending-input-notice-body');
    expect(body).toHaveTextContent('Waiting for the current response to finish.');
  });

  it('has NO Stop-waiting button (removed in v3)', () => {
    render(
      <PendingInputNotice
        kind="cross"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onEdit={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.queryByTestId('pending-input-notice-stop')).toBeNull();
  });

  it('renders [Run now] only for kind="cross"', () => {
    const { rerender } = render(
      <PendingInputNotice
        kind="cross"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onEdit={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('pending-input-notice-run-now')).toHaveTextContent('Run now');

    rerender(<PendingInputNotice kind="same" onEdit={vi.fn().mockResolvedValue(undefined)} />);
    expect(screen.queryByTestId('pending-input-notice-run-now')).toBeNull();
  });

  it('[Run now] invokes onRunNow only', async () => {
    const onRunNow = vi.fn().mockResolvedValue(undefined);
    const onEdit = vi.fn().mockResolvedValue(undefined);
    render(<PendingInputNotice kind="cross" onRunNow={onRunNow} onEdit={onEdit} />);
    await userEvent.click(screen.getByTestId('pending-input-notice-run-now'));
    await waitFor(() => expect(onRunNow).toHaveBeenCalledTimes(1));
    expect(onEdit).not.toHaveBeenCalled();
  });

  it('tapping the waiting line invokes onEdit (tap-to-edit; mobile ≈ ↑)', async () => {
    const onEdit = vi.fn().mockResolvedValue(undefined);
    render(<PendingInputNotice kind="same" onEdit={onEdit} />);
    const edit = screen.getByTestId('pending-input-notice-edit');
    expect(edit).toHaveAttribute('aria-label', 'Edit queued message');
    await userEvent.click(edit);
    await waitFor(() => expect(onEdit).toHaveBeenCalledTimes(1));
  });

  it('disables tap-to-edit when canEdit is false (entry has attachments)', async () => {
    const onEdit = vi.fn().mockResolvedValue(undefined);
    render(<PendingInputNotice kind="cross" canEdit={false} onRunNow={vi.fn()} onEdit={onEdit} />);
    const edit = screen.getByTestId('pending-input-notice-edit');
    expect(edit).toBeDisabled();
    await userEvent.click(edit);
    expect(onEdit).not.toHaveBeenCalled();
  });

  it('surfaces an inline error if onEdit rejects, and stays mounted', async () => {
    const onEdit = vi.fn().mockRejectedValue(new Error('boom'));
    render(<PendingInputNotice kind="same" onEdit={onEdit} />);
    await userEvent.click(screen.getByTestId('pending-input-notice-edit'));
    await waitFor(() => {
      expect(screen.getByTestId('pending-input-notice-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('pending-input-notice')).toBeInTheDocument();
  });

  it('surfaces an inline error if onRunNow rejects', async () => {
    const onRunNow = vi.fn().mockRejectedValue(new Error('nope'));
    render(
      <PendingInputNotice
        kind="cross"
        onRunNow={onRunNow}
        onEdit={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await userEvent.click(screen.getByTestId('pending-input-notice-run-now'));
    await waitFor(() => {
      expect(screen.getByTestId('pending-input-notice-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('pending-input-notice-run-now')).not.toBeDisabled();
  });
});

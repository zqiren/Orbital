// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for PendingInputNotice — pending-input queue (spec 006, Path A) §6.
 *
 * This is the NEW happy-path affordance that replaces SlotHeldNotice on the
 * 202 `queued_pending_slot` response (SlotHeldNotice is retained as the
 * enqueue-failure fallback). The component is callback-driven; these tests:
 *   1. Verify the wait copy mentions the holder verbatim.
 *   2. Verify both buttons render ([Run now], [Stop waiting]).
 *   3. Verify [Run now] invokes onRunNow ONLY (no onStopWaiting).
 *   4. Verify [Stop waiting] invokes onStopWaiting ONLY (no onRunNow).
 *   5. Verify a rejecting onStopWaiting / onRunNow surfaces the inline error.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import PendingInputNotice from './PendingInputNotice';

describe('PendingInputNotice', () => {
  it('renders the holder in the waiting copy', () => {
    render(
      <PendingInputNotice
        holder="sess-research-A"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onStopWaiting={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('pending-input-notice')).toBeInTheDocument();
    const body = screen.getByTestId('pending-input-notice-body');
    expect(body).toHaveTextContent('sess-research-A');
    expect(body.textContent ?? '').toContain('Waiting for');
  });

  it('renders both affordances [Run now] and [Stop waiting]', () => {
    render(
      <PendingInputNotice
        holder="sess-A"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onStopWaiting={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('pending-input-notice-run-now')).toHaveTextContent('Run now');
    expect(screen.getByTestId('pending-input-notice-stop')).toHaveTextContent('Stop waiting');
  });

  it('[Run now] invokes onRunNow and NOT onStopWaiting', async () => {
    const onRunNow = vi.fn().mockResolvedValue(undefined);
    const onStopWaiting = vi.fn().mockResolvedValue(undefined);
    render(
      <PendingInputNotice holder="sess-A" onRunNow={onRunNow} onStopWaiting={onStopWaiting} />,
    );
    await userEvent.click(screen.getByTestId('pending-input-notice-run-now'));
    await waitFor(() => expect(onRunNow).toHaveBeenCalledTimes(1));
    expect(onStopWaiting).not.toHaveBeenCalled();
  });

  it('[Stop waiting] invokes onStopWaiting and NOT onRunNow', async () => {
    const onRunNow = vi.fn().mockResolvedValue(undefined);
    const onStopWaiting = vi.fn().mockResolvedValue(undefined);
    render(
      <PendingInputNotice holder="sess-A" onRunNow={onRunNow} onStopWaiting={onStopWaiting} />,
    );
    await userEvent.click(screen.getByTestId('pending-input-notice-stop'));
    await waitFor(() => expect(onStopWaiting).toHaveBeenCalledTimes(1));
    expect(onRunNow).not.toHaveBeenCalled();
  });

  it('surfaces an inline error if onStopWaiting rejects', async () => {
    const onStopWaiting = vi.fn().mockRejectedValue(new Error('boom'));
    render(
      <PendingInputNotice
        holder="sess-A"
        onRunNow={vi.fn().mockResolvedValue(undefined)}
        onStopWaiting={onStopWaiting}
      />,
    );
    await userEvent.click(screen.getByTestId('pending-input-notice-stop'));
    await waitFor(() => {
      expect(screen.getByTestId('pending-input-notice-error')).toBeInTheDocument();
    });
    // The notice stays mounted so the user can retry.
    expect(screen.getByTestId('pending-input-notice')).toBeInTheDocument();
    // Stop button is re-enabled after the failure (busy cleared).
    expect(screen.getByTestId('pending-input-notice-stop')).not.toBeDisabled();
  });

  it('surfaces an inline error if onRunNow rejects', async () => {
    const onRunNow = vi.fn().mockRejectedValue(new Error('nope'));
    render(
      <PendingInputNotice
        holder="sess-A"
        onRunNow={onRunNow}
        onStopWaiting={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await userEvent.click(screen.getByTestId('pending-input-notice-run-now'));
    await waitFor(() => {
      expect(screen.getByTestId('pending-input-notice-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('pending-input-notice-run-now')).not.toBeDisabled();
  });
});

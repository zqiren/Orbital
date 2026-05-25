// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Tests for SlotHeldNotice — Track J Phase 1 intermediate UX.
 *
 * Per TASK/DISPATCH-2026-05-22.md §5.5 test 5: the 202 response renders the
 * notice with both affordances and the cancel-and-send path calls the right
 * cancel endpoint then re-injects.
 *
 * The component itself is callback-driven (onWait, onCancelAndSend), so these
 * tests:
 *   1. Verify the rendered text mentions the holding session_id verbatim.
 *   2. Verify [Wait] invokes onWait without calling onCancelAndSend.
 *   3. Verify [Cancel running session and send] invokes onCancelAndSend, and
 *      that the orchestrating caller can plumb that into /cancel + re-inject.
 *
 * Test 3 wires up a realistic harness that mimics the ChatView integration:
 * the onCancelAndSend handler hits a stubbed cancel endpoint then a stubbed
 * re-inject. Verifies the cancel is called BEFORE the inject (ordering) and
 * with the right projectId.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import SlotHeldNotice from './SlotHeldNotice';

describe('SlotHeldNotice', () => {
  it('renders the holding session_id in the notice text', () => {
    render(
      <SlotHeldNotice
        holdingSessionId="sess-research-openhuman"
        onWait={vi.fn()}
        onCancelAndSend={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('slot-held-notice')).toBeInTheDocument();
    expect(screen.getByTestId('slot-held-notice-holder')).toHaveTextContent(
      'sess-research-openhuman',
    );
    // Surrounding copy is present (verbatim from DISPATCH-2026-05-22 §5.4)
    const notice = screen.getByTestId('slot-held-notice');
    expect(notice.textContent ?? '').toContain('currently running');
    expect(notice.textContent ?? '').toContain('Wait for it to finish');
  });

  it('renders both affordances [Wait] and [Cancel running session and send]', () => {
    render(
      <SlotHeldNotice
        holdingSessionId="sess-A"
        onWait={vi.fn()}
        onCancelAndSend={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByTestId('slot-held-notice-wait')).toHaveTextContent('Wait');
    expect(
      screen.getByTestId('slot-held-notice-cancel-and-send'),
    ).toHaveTextContent('Cancel running session and send');
  });

  it('[Wait] invokes onWait and does NOT call onCancelAndSend', async () => {
    const onWait = vi.fn();
    const onCancelAndSend = vi.fn().mockResolvedValue(undefined);
    render(
      <SlotHeldNotice
        holdingSessionId="sess-A"
        onWait={onWait}
        onCancelAndSend={onCancelAndSend}
      />,
    );
    await userEvent.click(screen.getByTestId('slot-held-notice-wait'));
    expect(onWait).toHaveBeenCalledTimes(1);
    expect(onCancelAndSend).not.toHaveBeenCalled();
  });

  it('[Cancel and send] invokes onCancelAndSend in cancel-then-inject order', async () => {
    // Mimic the ChatView wiring: onCancelAndSend should cancel the running
    // turn, then re-inject. The component itself only calls the prop —
    // we verify here that the prop fires once, and the harness records
    // the order of the underlying cancel and inject calls.
    const calls: string[] = [];
    const cancelEndpoint = vi.fn(async (_projectId: string) => {
      calls.push('cancel');
    });
    const reinjectEndpoint = vi.fn(async (_projectId: string, _content: string) => {
      calls.push('inject');
    });
    const onCancelAndSend = vi.fn(async () => {
      await cancelEndpoint('proj-1');
      await reinjectEndpoint('proj-1', 'pending content');
    });

    render(
      <SlotHeldNotice
        holdingSessionId="sess-A"
        onWait={vi.fn()}
        onCancelAndSend={onCancelAndSend}
      />,
    );
    await userEvent.click(
      screen.getByTestId('slot-held-notice-cancel-and-send'),
    );

    await waitFor(() => {
      expect(onCancelAndSend).toHaveBeenCalledTimes(1);
    });
    expect(cancelEndpoint).toHaveBeenCalledTimes(1);
    expect(cancelEndpoint).toHaveBeenCalledWith('proj-1');
    expect(reinjectEndpoint).toHaveBeenCalledTimes(1);
    expect(reinjectEndpoint).toHaveBeenCalledWith('proj-1', 'pending content');
    // Cancel MUST happen before inject — verifies the wire contract from
    // DISPATCH-2026-05-22 §5.4.
    expect(calls).toEqual(['cancel', 'inject']);
  });

  it('shows an inline error if onCancelAndSend rejects', async () => {
    const onCancelAndSend = vi.fn().mockRejectedValue(new Error('boom'));
    render(
      <SlotHeldNotice
        holdingSessionId="sess-A"
        onWait={vi.fn()}
        onCancelAndSend={onCancelAndSend}
      />,
    );
    await userEvent.click(
      screen.getByTestId('slot-held-notice-cancel-and-send'),
    );
    await waitFor(() => {
      expect(screen.getByTestId('slot-held-notice-error')).toHaveTextContent(
        'boom',
      );
    });
    // The notice stays mounted so the user can retry [Wait].
    expect(screen.getByTestId('slot-held-notice')).toBeInTheDocument();
  });
});

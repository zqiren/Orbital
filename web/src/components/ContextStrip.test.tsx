// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ContextStrip from './ContextStrip';
import { LocaleProvider } from '../i18n/LocaleContext';
import type { ContextUsage } from '../hooks/useContextUsage';

function usage(over: Partial<ContextUsage> = {}): ContextUsage {
  return {
    used: 0,
    window: 200_000,
    threshold: 160_000,
    provider: 'anthropic',
    model: 'claude-x',
    ...over,
  };
}

function renderStrip(u: ContextUsage | null, onNewSession = vi.fn()) {
  return render(
    <LocaleProvider>
      <ContextStrip usage={u} onNewSession={onNewSession} />
    </LocaleProvider>,
  );
}

const track = () => screen.queryByTestId('context-strip');
const fill = () => screen.getByTestId('context-fill');
const tick = () => screen.getByTestId('context-tick');

describe('ContextStrip', () => {
  it('renders nothing without a measurement', () => {
    // A session that has never called the model is not "0% full" — it is
    // unmeasured. Painting an empty meter would be a claim we cannot make.
    renderStrip(null);
    expect(track()).toBeNull();
  });

  it('renders nothing when used is null', () => {
    renderStrip(usage({ used: null }));
    expect(track()).toBeNull();
  });

  it('stays hidden while the context is nowhere near compaction', () => {
    // Ambient, not a dashboard: the user can do nothing useful at 20%, so it
    // should not occupy the eye.
    renderStrip(usage({ used: 40_000 }));
    expect(track()).toBeNull();
  });

  it('appears as the context approaches the compaction point', () => {
    renderStrip(usage({ used: 150_000 }));
    expect(track()).not.toBeNull();
  });

  it('measures the bar against the FULL window, not the threshold', () => {
    // The window size is what the user recognises ("200k model"), so the bar
    // has to span it — otherwise the number and the geometry disagree.
    renderStrip(usage({ used: 150_000 }));
    expect(fill().style.width).toBe('75%');
  });

  it('puts the tick where the agent will actually compact', () => {
    renderStrip(usage({ used: 150_000 }));
    expect(tick().style.left).toBe('80%');
  });

  it('moves the tick with the threshold the server reports', () => {
    // Small-window models keep the old budget-derived trigger, so the mark
    // must follow the server rather than assume a flat 80%.
    renderStrip(usage({ used: 9_000, window: 32_768, threshold: 10_214 }));
    expect(parseFloat(tick().style.left)).toBeCloseTo(31.2, 1);
  });

  it('shows used and window in the label', () => {
    renderStrip(usage({ used: 150_000 }));
    expect(screen.getByTestId('context-label').textContent).toBe('150K / 200K');
  });

  it('offers a new session only in the window where it still helps', () => {
    const onNew = vi.fn();
    // Well before the mark: nothing to decide yet.
    const { unmount } = renderStrip(usage({ used: 120_000 }), onNew);
    expect(screen.queryByTestId('context-new-session')).toBeNull();
    unmount();
    // Closing on the mark: this is the moment a fresh session avoids the
    // summarization entirely.
    renderStrip(usage({ used: 155_000 }), onNew);
    expect(screen.getByTestId('context-new-session')).toBeTruthy();
  });

  it('withdraws the offer once compaction has already happened', () => {
    // Past the threshold the summarization has run; offering "New session"
    // there advertises a choice the user no longer has.
    renderStrip(usage({ used: 165_000 }));
    expect(screen.queryByTestId('context-new-session')).toBeNull();
  });

  it('calls back when the new-session offer is taken', async () => {
    const onNew = vi.fn();
    renderStrip(usage({ used: 155_000 }), onNew);
    screen.getByTestId('context-new-session').click();
    expect(onNew).toHaveBeenCalledTimes(1);
  });

  it('warns in colour once the threshold is crossed', () => {
    const { unmount } = renderStrip(usage({ used: 150_000 }));
    const before = fill().getAttribute('data-tone');
    unmount();
    renderStrip(usage({ used: 165_000 }));
    expect(fill().getAttribute('data-tone')).not.toBe(before);
    expect(fill().getAttribute('data-tone')).toBe('over');
  });

  it('never overflows the track past 100%', () => {
    renderStrip(usage({ used: 400_000 }));
    expect(fill().style.width).toBe('100%');
  });

  it('survives a zero window without dividing by zero', () => {
    renderStrip(usage({ used: 100, window: 0, threshold: 0 }));
    expect(track()).toBeNull();
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionListItem } from './SessionListItem';
import type { SessionListEntry } from '../types';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSession(overrides: Partial<SessionListEntry> = {}): SessionListEntry {
  return {
    session_id: 'sess-test',
    status: 'idle',
    session_uuid: 'uuid-test',
    last_terminal_event: null,
    last_activity_at: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Status glyph + color state coverage
// ---------------------------------------------------------------------------

describe('SessionListItem — status glyph rendering', () => {
  it('running → glyph ◐', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-running', status: 'running' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph.textContent).toBe('◐');
    expect(glyph).toHaveStyle({ color: '#22C55E' });
  });

  it('waiting → glyph ⟳', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-waiting', status: 'waiting' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph.textContent).toBe('⟳');
    expect(glyph).toHaveStyle({ color: '#6366F1' });
  });

  it('pending_approval (Blocked) → glyph ⚠ with warning color', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-blocked', status: 'pending_approval' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph.textContent).toBe('⚠');
    expect(glyph).toHaveStyle({ color: '#F59E0B' });
  });

  it('idle → glyph ⏸ with muted color', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-idle', status: 'idle' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph.textContent).toBe('⏸');
    expect(glyph).toHaveStyle({ color: '#A1A1AA' });
  });
});

// ---------------------------------------------------------------------------
// Session name + last_activity_at rendering
// ---------------------------------------------------------------------------

describe('SessionListItem — name and time', () => {
  it('renders session_id as the display name', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'my-cool-session' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-name')).toHaveTextContent('my-cool-session');
  });

  it('renders "—" when last_activity_at is null', () => {
    render(
      <SessionListItem
        session={makeSession({ last_activity_at: null })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-time')).toHaveTextContent('—');
  });

  it('renders "—" when last_activity_at is undefined', () => {
    const session = makeSession();
    delete session.last_activity_at;
    render(
      <SessionListItem session={session} selected={false} onSelect={vi.fn()} />,
    );
    expect(screen.getByTestId('session-time')).toHaveTextContent('—');
  });

  it('renders a relative time string when last_activity_at is set', () => {
    // Use a fixed recent timestamp: 5 minutes ago
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    render(
      <SessionListItem
        session={makeSession({ last_activity_at: fiveMinAgo })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-time')).toHaveTextContent('m ago');
  });
});

// ---------------------------------------------------------------------------
// Error indicator (SessionStatusGlyph)
// ---------------------------------------------------------------------------

describe('SessionListItem — error indicator', () => {
  it('shows the error glyph when last_terminal_event.type === "error"', () => {
    render(
      <SessionListItem
        session={makeSession({
          last_terminal_event: {
            type: 'error',
            timestamp: '2026-05-24T12:00:00Z',
            details: 'Something went wrong',
          },
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-error-glyph')).toBeInTheDocument();
  });

  it('does NOT show error glyph when last_terminal_event is null', () => {
    render(
      <SessionListItem
        session={makeSession({ last_terminal_event: null })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('session-error-glyph')).toBeNull();
  });

  it('does NOT show error glyph when last_terminal_event.type === "stopped"', () => {
    render(
      <SessionListItem
        session={makeSession({
          last_terminal_event: { type: 'stopped', timestamp: '2026-05-24T12:00:00Z', details: null },
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('session-error-glyph')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Selected / active highlight
// ---------------------------------------------------------------------------

describe('SessionListItem — active highlight', () => {
  it('applies font-medium class when selected', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-sel' })}
        selected={true}
        onSelect={vi.fn()}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-sel');
    expect(row.className).toContain('font-medium');
  });

  it('does not apply font-medium when not selected', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-unsel' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-unsel');
    expect(row.className).not.toContain('font-medium');
  });

  it('applies white background style when selected', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-bg' })}
        selected={true}
        onSelect={vi.fn()}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-bg');
    expect(row).toHaveStyle({ background: '#fff' });
  });
});

// ---------------------------------------------------------------------------
// Queue-origin hue variation
// ---------------------------------------------------------------------------

describe('SessionListItem — queue-origin hue variation', () => {
  it('queue-origin session: status dot has data-origin="queue"', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-q', status: 'running', origin: 'queue' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph).toHaveAttribute('data-origin', 'queue');
  });

  it('manual-origin session: status dot has data-origin="manual"', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-m', status: 'running', origin: 'manual' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph).toHaveAttribute('data-origin', 'manual');
  });

  it('undefined-origin session: treated as manual (data-origin="manual")', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-u', status: 'running' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph).toHaveAttribute('data-origin', 'manual');
  });

  it('queue-origin dot color differs from manual-origin dot color (desaturated toward gray)', () => {
    const { unmount } = render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-qc', status: 'running', origin: 'queue' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const queueGlyph = screen.getByTestId('session-status-glyph');
    const queueColor = queueGlyph.style.color;
    unmount();

    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-mc', status: 'running', origin: 'manual' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const manualGlyph = screen.getByTestId('session-status-glyph');
    const manualColor = manualGlyph.style.color;

    // Colors must differ (queue is desaturated toward gray)
    expect(queueColor).not.toBe(manualColor);
  });

  it('queue-origin hue variation also applies to a non-running status (waiting)', () => {
    const { unmount } = render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-qw', status: 'waiting', origin: 'queue' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const queueColor = screen.getByTestId('session-status-glyph').style.color;
    unmount();

    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-mw', status: 'waiting', origin: 'manual' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const manualColor = screen.getByTestId('session-status-glyph').style.color;

    expect(queueColor).not.toBe(manualColor);
  });
});

// ---------------------------------------------------------------------------
// onSelect fires on click
// ---------------------------------------------------------------------------

describe('SessionListItem — interaction', () => {
  it('calls onSelect when clicked', async () => {
    const onSelect = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-click' })}
        selected={false}
        onSelect={onSelect}
      />,
    );
    await userEvent.click(screen.getByTestId('session-list-item-sess-click'));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('calls onSelect on Enter keydown', async () => {
    const onSelect = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-enter' })}
        selected={false}
        onSelect={onSelect}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-enter');
    row.focus();
    await userEvent.keyboard('{Enter}');
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('calls onSelect on Space keydown', async () => {
    const onSelect = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-space' })}
        selected={false}
        onSelect={onSelect}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-space');
    row.focus();
    await userEvent.keyboard(' ');
    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});

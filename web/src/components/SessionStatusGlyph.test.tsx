// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Regression: SessionStatusGlyph renders the warning glyph only when the
// session's last_terminal_event.type === 'error'. Other terminal types
// (stopped, new_session) are silent today; null / undefined renders null.
//
// See: TASK/TASK-state-model-alignment-fixes.md §2.4 + §3 (test 4
// frontend half).

import { describe, it, expect, beforeEach } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { SessionStatusGlyph } from './SessionStatusGlyph';
import type { LastTerminalEvent } from '../types';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

describe('SessionStatusGlyph', () => {
  it('renders the warning glyph when last_terminal_event.type === "error"', async () => {
    const evt: LastTerminalEvent = {
      type: 'error',
      timestamp: '2026-05-24T12:00:00Z',
      details: 'LLM provider 500: connection reset',
    };
    await act(async () => {
      root.render(<SessionStatusGlyph lastTerminalEvent={evt} />);
    });
    const glyph = container.querySelector('[data-testid="session-error-glyph"]');
    expect(glyph).toBeTruthy();
    expect(glyph?.getAttribute('aria-label')).toBe('Session errored');
    // Title (lucide wraps children inside the svg) carries the details.
    expect(container.textContent ?? '').toContain('LLM provider 500');
  });

  it('renders nothing when last_terminal_event is null', async () => {
    await act(async () => {
      root.render(<SessionStatusGlyph lastTerminalEvent={null} />);
    });
    expect(container.querySelector('[data-testid="session-error-glyph"]')).toBeNull();
  });

  it('renders nothing when last_terminal_event.type === "stopped"', async () => {
    const evt: LastTerminalEvent = {
      type: 'stopped',
      timestamp: '2026-05-24T12:00:00Z',
      details: null,
    };
    await act(async () => {
      root.render(<SessionStatusGlyph lastTerminalEvent={evt} />);
    });
    expect(container.querySelector('[data-testid="session-error-glyph"]')).toBeNull();
  });

  it('renders nothing when last_terminal_event.type === "new_session"', async () => {
    const evt: LastTerminalEvent = {
      type: 'new_session',
      timestamp: '2026-05-24T12:00:00Z',
      details: null,
    };
    await act(async () => {
      root.render(<SessionStatusGlyph lastTerminalEvent={evt} />);
    });
    expect(container.querySelector('[data-testid="session-error-glyph"]')).toBeNull();
  });

  it('uses a generic title when details are null', async () => {
    const evt: LastTerminalEvent = {
      type: 'error',
      timestamp: '2026-05-24T12:00:00Z',
      details: null,
    };
    await act(async () => {
      root.render(<SessionStatusGlyph lastTerminalEvent={evt} />);
    });
    // Falls back to the bare label when no details available.
    expect(container.textContent ?? '').toContain('Session errored');
  });
});

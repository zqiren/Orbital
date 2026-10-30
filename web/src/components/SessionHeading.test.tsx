// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import SessionHeading from './SessionHeading';
import type { SessionListEntry } from '../types';

function entry(over: Partial<SessionListEntry>): SessionListEntry {
  return {
    session_id: 'proj_sess0001',
    status: 'idle',
    session_uuid: null,
    last_terminal_event: null,
    ...over,
  } as SessionListEntry;
}

// Spec 137 follow-up: the chat beside the project column starts with the
// session's name as a heading — same type as PageHeading, no bar, no line.
describe('SessionHeading', () => {
  it('a named chat: the name as a 15px heading, nothing above it', () => {
    const { container } = render(<SessionHeading session={entry({ name: '看看tutti的近况' })} />);
    const h1 = screen.getByRole('heading', { level: 1 });
    expect(h1.textContent).toBe('看看tutti的近况');
    expect(h1.className).toContain('text-[15px]');
    expect(screen.queryByTestId('session-heading-eyebrow')).toBeNull();
    expect(container.innerHTML).not.toMatch(/border/);
  });

  it('an automation run: the automation name, with its kind small above', () => {
    render(<SessionHeading session={entry({ name: "[Triggered by schedule 'index-1027F' (every minute)]\n\nRead orbital/INDEX.md…", origin: 'chat', trigger_type: 'schedule' })} />);
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('index-1027F');
    expect(screen.getByTestId('session-heading-eyebrow').textContent).toBe('Scheduled');
  });

  it('an imported native session names the agent it came from', () => {
    render(
      <SessionHeading
        session={entry({
          name: 'Fix the login bug',
          origin: 'imported',
          imported: { agent: 'codex', native_id: 'x', materialized: true, size_bytes: 1 },
        })}
      />,
    );
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Fix the login bug');
    expect(screen.getByTestId('session-heading-eyebrow').textContent).toBe('Imported · codex');
  });

  it('a queue session with no name of its own: the kind is the title', () => {
    render(<SessionHeading session={entry({ name: null, origin: 'queue' })} />);
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Queue task');
    expect(screen.queryByTestId('session-heading-eyebrow')).toBeNull();
  });

  it('a chat with no name yet never shows its raw id', () => {
    render(<SessionHeading session={entry({ name: '', origin: 'chat' })} />);
    expect(screen.getByTestId('session-heading-title').textContent).toBe('New session');
  });

  it('no session yet: "New session"', () => {
    render(<SessionHeading session={undefined} />);
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('New session');
  });
});

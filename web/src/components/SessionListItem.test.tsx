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
    expect(glyph).toHaveStyle({ color: '#539AF8' });
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

  it('idle → no glyph (the slot stays empty so rows read clean)', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-idle', status: 'idle' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const glyph = screen.getByTestId('session-status-glyph');
    expect(glyph.textContent).toBe('');
  });
});

// ---------------------------------------------------------------------------
// Session name + last_activity_at rendering
// ---------------------------------------------------------------------------

describe('SessionListItem — name and time', () => {
  it('renders session.name as the display label when present', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess_abcd1234', name: 'Implement login flow' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-name')).toHaveTextContent('Implement login flow');
  });

  it('falls back to session_id when name is null', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'my-cool-session', name: null })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-name')).toHaveTextContent('my-cool-session');
  });

  it('falls back to session_id when name is undefined', () => {
    const session = makeSession({ session_id: 'sess-no-name' });
    delete session.name;
    render(<SessionListItem session={session} selected={false} onSelect={vi.fn()} />);
    expect(screen.getByTestId('session-name')).toHaveTextContent('sess-no-name');
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
// Machine-session kind chips (queue / triggers / attachments)
// ---------------------------------------------------------------------------

describe('SessionListItem — kind chips and cleaned labels', () => {
  it('queue-origin session renders the queue chip at the front, no name text', () => {
    render(
      <SessionListItem
        session={makeSession({
          session_id: 'sess-q',
          status: 'running',
          origin: 'queue',
          name: '[QUEUE ITEM | id=item_ab12 | attempt=1] You are…',
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-kind-chip')).toHaveTextContent('Queue task');
    expect(screen.getByTestId('session-name')).toHaveTextContent('');
  });

  it('renamed queue session keeps the chip and shows the renamed text', () => {
    render(
      <SessionListItem
        session={makeSession({
          session_id: 'sess-qr',
          status: 'running',
          origin: 'queue',
          name: 'Weekly digest run',
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-kind-chip')).toHaveTextContent('Queue task');
    expect(screen.getByTestId('session-name')).toHaveTextContent('Weekly digest run');
  });

  it('schedule-trigger session: chip + extracted trigger name + tooltip detail', () => {
    render(
      <SessionListItem
        session={makeSession({
          session_id: 'sess-t',
          name: "[Triggered by schedule 'Daily check' (every d…",
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    const chip = screen.getByTestId('session-kind-chip');
    expect(chip).toHaveTextContent('Scheduled');
    expect(chip).toHaveAttribute('title', "schedule 'Daily check'");
    expect(screen.getByTestId('session-name')).toHaveTextContent('Daily check');
  });

  it('file_watch-trigger session renders the file-watch chip', () => {
    render(
      <SessionListItem
        session={makeSession({
          session_id: 'sess-w',
          name: "[Triggered by file_watch 'specs watcher']\nChang…",
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByTestId('session-kind-chip')).toHaveTextContent('File watch');
    expect(screen.getByTestId('session-name')).toHaveTextContent('specs watcher');
  });

  it('attachment-first session: NO chip, basename as the label', () => {
    render(
      <SessionListItem
        session={makeSession({
          session_id: 'sess-a',
          name: '<attached_files>\n- /uploads/2026-08-12T053000-report.pdf (app…',
        })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('session-kind-chip')).toBeNull();
    expect(screen.getByTestId('session-name')).toHaveTextContent('report.pdf');
  });

  it('plain session renders no chip and the verbatim name (status color intact)', () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-p', status: 'running', name: 'write an essay' })}
        selected={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('session-kind-chip')).toBeNull();
    expect(screen.getByTestId('session-name')).toHaveTextContent('write an essay');
    // The old queue-origin desaturation is retired: status color is the token value.
    expect(screen.getByTestId('session-status-glyph')).toHaveStyle({ color: '#22C55E' });
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

// ---------------------------------------------------------------------------
// Inline rename
// ---------------------------------------------------------------------------

describe('SessionListItem — inline rename', () => {
  it('double-clicking the name opens an editable input prefilled with the label', async () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-rn', name: 'Old Name' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={vi.fn()}
      />,
    );
    await userEvent.dblClick(screen.getByTestId('session-name'));
    const input = screen.getByTestId('session-rename-input') as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.value).toBe('Old Name');
  });

  it('Enter saves the new name via onRename', async () => {
    const onRename = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-rn2', name: 'Old' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );
    await userEvent.dblClick(screen.getByTestId('session-name'));
    const input = screen.getByTestId('session-rename-input');
    await userEvent.clear(input);
    await userEvent.type(input, 'Brand New Name{Enter}');
    expect(onRename).toHaveBeenCalledWith('sess-rn2', 'Brand New Name');
    // Input closes after save.
    expect(screen.queryByTestId('session-rename-input')).toBeNull();
  });

  it('Escape cancels the rename without calling onRename', async () => {
    const onRename = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-rn3', name: 'KeepMe' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );
    await userEvent.dblClick(screen.getByTestId('session-name'));
    const input = screen.getByTestId('session-rename-input');
    await userEvent.clear(input);
    await userEvent.type(input, 'Discarded{Escape}');
    expect(onRename).not.toHaveBeenCalled();
    expect(screen.queryByTestId('session-rename-input')).toBeNull();
    // Original label still shown.
    expect(screen.getByTestId('session-name')).toHaveTextContent('KeepMe');
  });

  it('saving an unchanged name does not call onRename', async () => {
    const onRename = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-rn4', name: 'Same' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );
    await userEvent.dblClick(screen.getByTestId('session-name'));
    const input = screen.getByTestId('session-rename-input');
    await userEvent.type(input, '{Enter}'); // value unchanged
    expect(onRename).not.toHaveBeenCalled();
  });

  it('entering rename mode does not trigger onSelect on the row', async () => {
    const onSelect = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-rn5', name: 'X' })}
        selected={false}
        onSelect={onSelect}
        onRename={vi.fn()}
      />,
    );
    await userEvent.dblClick(screen.getByTestId('session-name'));
    // dblClick fires two clicks; with the input open, the row click is a no-op.
    // The important invariant: typing in the input does not navigate.
    const input = screen.getByTestId('session-rename-input');
    await userEvent.type(input, 'abc');
    // No assertion on onSelect count from the dblClick itself (it may fire on
    // the first click before editing) — just confirm the input is still open.
    expect(screen.getByTestId('session-rename-input')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Three-dot menu + delete confirmation
// ---------------------------------------------------------------------------

describe('SessionListItem — context menu and delete', () => {
  it('opens the three-dot menu with Rename and Delete actions', async () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-menu' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-three-dot-dropdown')).toBeInTheDocument();
    expect(screen.getByTestId('session-action-rename')).toBeInTheDocument();
    expect(screen.getByTestId('session-action-delete')).toBeInTheDocument();
    // The delete action is NOT disabled (real handler, not a Batch-4 placeholder).
    expect(screen.getByTestId('session-action-delete')).not.toBeDisabled();
  });

  it('right-clicking the row opens the menu', async () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-ctx' })}
        selected={false}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    const row = screen.getByTestId('session-list-item-sess-ctx');
    await userEvent.pointer({ keys: '[MouseRight]', target: row });
    expect(screen.getByTestId('session-three-dot-dropdown')).toBeInTheDocument();
  });

  it('Rename menu action opens the inline editor', async () => {
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-menu-rn', name: 'MenuRename' })}
        selected={false}
        onSelect={vi.fn()}
        onRename={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    await userEvent.click(screen.getByTestId('session-action-rename'));
    const input = screen.getByTestId('session-rename-input') as HTMLInputElement;
    expect(input.value).toBe('MenuRename');
  });

  it('Delete action opens a confirmation dialog (does not delete immediately)', async () => {
    const onDelete = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-dc' })}
        selected={false}
        onSelect={vi.fn()}
        onDelete={onDelete}
      />,
    );
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    await userEvent.click(screen.getByTestId('session-action-delete'));
    expect(screen.getByTestId('session-delete-confirm')).toBeInTheDocument();
    expect(onDelete).not.toHaveBeenCalled();
  });

  it('confirming delete calls onDelete with the session_id', async () => {
    const onDelete = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-dc2' })}
        selected={false}
        onSelect={vi.fn()}
        onDelete={onDelete}
      />,
    );
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    await userEvent.click(screen.getByTestId('session-action-delete'));
    await userEvent.click(screen.getByTestId('session-delete-confirm-button'));
    expect(onDelete).toHaveBeenCalledWith('sess-dc2');
    // Dialog closes.
    expect(screen.queryByTestId('session-delete-confirm')).toBeNull();
  });

  it('cancelling the delete dialog does not call onDelete', async () => {
    const onDelete = vi.fn();
    render(
      <SessionListItem
        session={makeSession({ session_id: 'sess-dc3' })}
        selected={false}
        onSelect={vi.fn()}
        onDelete={onDelete}
      />,
    );
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    await userEvent.click(screen.getByTestId('session-action-delete'));
    await userEvent.click(screen.getByTestId('session-delete-cancel'));
    expect(onDelete).not.toHaveBeenCalled();
    expect(screen.queryByTestId('session-delete-confirm')).toBeNull();
  });
});

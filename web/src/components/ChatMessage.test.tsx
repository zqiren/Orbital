// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ChatMessage from './ChatMessage';
import type { DisplayItem } from '../utils/chatTransform';

type UserMessage = Extract<DisplayItem, { type: 'user_message' }>;
type AgentMessage = Extract<DisplayItem, { type: 'agent_message' }>;

function userMsg(content: string, target?: string): UserMessage {
  return {
    type: 'user_message',
    content,
    timestamp: '2026-04-30T10:23:00Z',
    ...(target && { target }),
  };
}

function agentMsg(content: string): AgentMessage {
  return {
    type: 'agent_message',
    content,
    source: 'management',
    timestamp: '2026-04-30T10:23:00Z',
  };
}

describe('ChatMessage — user_message with <attached_files> block', () => {
  it('renders chips for each attachment and the stripped user text below', () => {
    const content =
      '<attached_files>\n' +
      '- uploads/2026-04-30T1023-screenshot.png (image/png, 1.0 KB)\n' +
      '- uploads/notes.txt (text/plain, 512 B)\n' +
      '</attached_files>\n\n' +
      "what's in these?";
    const { container } = render(<ChatMessage message={userMsg(content)} />);

    const chips = container.querySelectorAll('[data-testid="attachment-chip"]');
    expect(chips.length).toBe(2);

    // basename rendering — keep the timestamp prefix, matching what's on disk.
    expect(screen.getByText('2026-04-30T1023-screenshot.png')).toBeInTheDocument();
    expect(screen.getByText('notes.txt')).toBeInTheDocument();

    // stripped text is visible
    expect(screen.getByText("what's in these?")).toBeInTheDocument();

    // raw block must NOT appear
    expect(container.textContent ?? '').not.toContain('<attached_files>');
    expect(container.textContent ?? '').not.toContain('</attached_files>');
  });

  it('renders chips only (no empty text node) when there is no user text after the block', () => {
    const content =
      '<attached_files>\n- uploads/x.txt (text/plain, 11 B)\n</attached_files>\n\n';
    const { container } = render(<ChatMessage message={userMsg(content)} />);

    expect(container.querySelectorAll('[data-testid="attachment-chip"]').length).toBe(1);
    // The bubble has no "loose text" outside of the chips.
    const bubble = container.querySelector('.bg-card-hover');
    expect(bubble).not.toBeNull();
    // Filter to direct text nodes (chips wrap their own text).
    let directText = '';
    bubble?.childNodes.forEach((n) => {
      if (n.nodeType === Node.TEXT_NODE) directText += n.textContent ?? '';
    });
    expect(directText.trim()).toBe('');
  });
});

describe('ChatMessage — user_message without block', () => {
  it('renders content unchanged (regression: existing rendering still works)', () => {
    render(<ChatMessage message={userMsg('hello world')} />);
    expect(screen.getByText('hello world')).toBeInTheDocument();
  });
});

describe('ChatMessage — assistant/agent messages do not parse the block', () => {
  it('an agent_message with <attached_files> text in content is rendered as-is, no chips', () => {
    const content =
      '<attached_files>\n- uploads/x.txt (text/plain, 1 B)\n</attached_files>\n\nthe agent literally typed this';
    const { container } = render(<ChatMessage message={agentMsg(content)} />);
    expect(container.querySelectorAll('[data-testid="attachment-chip"]').length).toBe(0);
    // The raw block survives in the rendered bubble — MarkdownContent shows it as text.
    // We assert at least that no chips appeared (the key invariant).
  });
});

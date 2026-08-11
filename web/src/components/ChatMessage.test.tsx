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

  it('renders chips only (no loose text) when there is no user text after the block', () => {
    const content =
      '<attached_files>\n- uploads/x.txt (text/plain, 11 B)\n</attached_files>\n\n';
    const { container } = render(<ChatMessage message={userMsg(content)} />);

    expect(container.querySelectorAll('[data-testid="attachment-chip"]').length).toBe(1);
    // The content block holds the chips and no loose text.
    const chip = container.querySelector('[data-testid="attachment-chip"]');
    const contentBlock = chip?.parentElement?.parentElement;
    expect(contentBlock).not.toBeNull();
    // Filter to direct text nodes (chips wrap their own text).
    let directText = '';
    contentBlock?.childNodes.forEach((n) => {
      if (n.nodeType === Node.TEXT_NODE) directText += n.textContent ?? '';
    });
    expect(directText.trim()).toBe('');
  });
});

describe('ChatMessage — flat avatar-log layout (Design §5)', () => {
  it('user message renders a user avatar (initials), sender·time, and content with NO bubble', () => {
    const { container } = render(<ChatMessage message={userMsg('hello world')} />);

    // Avatar: user variant, shows the "ME" placeholder initials.
    const avatar = container.querySelector('[data-testid="message-avatar"]');
    expect(avatar).not.toBeNull();
    expect(avatar?.getAttribute('data-variant')).toBe('user');
    expect(avatar?.textContent).toBe('ME');
    expect(avatar?.className).toContain('bg-primary');

    // Sender label + 24h time (timestamp is 10:23:00Z → local HH:MM).
    expect(screen.getByText('you')).toBeInTheDocument();
    expect(screen.getByText(/^· \d{2}:\d{2}$/)).toBeInTheDocument();

    // Content present.
    expect(screen.getByText('hello world')).toBeInTheDocument();

    // No bubble bg / border / rounded box from the old layout.
    expect(container.querySelector('.bg-card-hover')).toBeNull();
    expect(container.querySelector('.justify-end')).toBeNull();
    expect(container.querySelector('.rounded-lg')).toBeNull();
  });

  it('user message targeting a sub-agent shows "you → @target" in the label', () => {
    render(<ChatMessage message={userMsg('do the thing', 'researcher')} />);
    expect(screen.getByText('you → @researcher')).toBeInTheDocument();
  });

  it('agent message renders the Orbital avatar, "agent" sender·time, and content with NO bubble', () => {
    const { container } = render(<ChatMessage message={agentMsg('agent reply here')} />);

    const avatar = container.querySelector('[data-testid="message-avatar"]');
    expect(avatar?.getAttribute('data-variant')).toBe('agent');
    expect(avatar?.tagName).toBe('IMG');
    expect(avatar?.getAttribute('src')).toBe('/icon-192.png');

    expect(screen.getByText('agent')).toBeInTheDocument();
    expect(screen.getByText('agent reply here')).toBeInTheDocument();

    // No bubble background/border box wrapping the content.
    expect(container.querySelector('.bg-background.border.rounded-lg')).toBeNull();
  });

  it('sub-agent message uses the source as the sender label', () => {
    const subMsg: DisplayItem = {
      type: 'sub_agent_message',
      content: 'sub agent output',
      source: 'researcher',
      timestamp: '2026-04-30T10:23:00Z',
    };
    render(<ChatMessage message={subMsg as Extract<DisplayItem, { type: 'sub_agent_message' }>} />);
    expect(screen.getByText('researcher')).toBeInTheDocument();
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

describe('ChatMessage — configured agent name', () => {
  const subAgentMsg: Extract<DisplayItem, { type: 'sub_agent_message' }> = {
    type: 'sub_agent_message',
    content: 'sub reply',
    source: 'claude-code',
    timestamp: '2026-04-30T10:23:00Z',
  };

  it('Test 1: a management message shows the configured agent_name', () => {
    render(<ChatMessage message={agentMsg('mgmt reply')} agentName="Research Bot" />);
    expect(screen.getByText('Research Bot')).toBeInTheDocument();
    expect(screen.queryByText('agent')).not.toBeInTheDocument();
  });

  it('Test 2: a management message with empty agent_name falls back to "agent"', () => {
    render(<ChatMessage message={agentMsg('mgmt reply')} agentName="" />);
    expect(screen.getByText('agent')).toBeInTheDocument();
  });

  it('Test 2b: a whitespace-only agent_name also falls back to "agent"', () => {
    render(<ChatMessage message={agentMsg('mgmt reply')} agentName="   " />);
    expect(screen.getByText('agent')).toBeInTheDocument();
  });

  it('Test 3: a sub-agent message shows its handle regardless of agentName', () => {
    render(<ChatMessage message={subAgentMsg} agentName="Research Bot" />);
    expect(screen.getByText('claude-code')).toBeInTheDocument();
    expect(screen.queryByText('Research Bot')).not.toBeInTheDocument();
  });
});

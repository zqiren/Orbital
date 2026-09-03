// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
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
  it('user message is right-anchored with NO avatar, a sender·time label, and a width-capped tint block', () => {
    const { container } = render(<ChatMessage message={userMsg('hello world')} />);

    // No avatar on the user side: once the row is on its own edge, position
    // encodes identity and a "ME" square is redundant. (Reverses 33be98b's
    // flat avatar-log, which put both speakers on one left edge.)
    expect(container.querySelector('[data-testid="message-avatar"]')).toBeNull();

    // Right-anchored row.
    const row = container.querySelector('[data-testid="user-message"]');
    expect(row).not.toBeNull();
    expect(row?.className).toContain('items-end');

    // Sender label + 24h time survive the avatar's removal — the label line is
    // the only carrier of "you → @target" on @mention sends.
    expect(screen.getByText('you')).toBeInTheDocument();
    expect(screen.getByText(/^· \d{2}:\d{2}$/)).toBeInTheDocument();

    // Content sits in a tint block that is CAPPED, not full-bleed: an
    // uncapped right-aligned "ok" would sit ~1200px from the reply it answers
    // on a maximised window. Notch flips tl→tr to point back at the label.
    const content = screen.getByText('hello world');
    expect(content.className).toContain('bg-sidebar');
    expect(content.className).toContain('max-w-[min(70%,680px)]');
    expect(content.className).toContain('rounded-tr-sm');
    expect(container.querySelector('.border.rounded-lg')).toBeNull();
  });

  it('agent rows stay LEFT-anchored and keep their avatar (the asymmetry is the point)', () => {
    const { container } = render(<ChatMessage message={agentMsg('sure thing')} />);
    const avatar = container.querySelector('[data-testid="message-avatar"]');
    expect(avatar?.getAttribute('data-variant')).toBe('agent');
    expect(container.querySelector('[data-testid="user-message"]')).toBeNull();
    expect(container.querySelector('.items-end')).toBeNull();
  });

  it('user message targeting a sub-agent shows "you → @target" in the label', () => {
    render(<ChatMessage message={userMsg('do the thing', 'researcher')} />);
    expect(screen.getByText('you → @researcher')).toBeInTheDocument();
  });

  it('agent message renders the Orbital avatar, "agent" sender·time, and content with NO bubble', () => {
    const { container } = render(<ChatMessage message={agentMsg('agent reply here')} />);

    const avatar = container.querySelector('[data-testid="message-avatar"]');
    expect(avatar?.getAttribute('data-variant')).toBe('agent');
    expect(avatar?.querySelector('img')?.getAttribute('src')).toBe('/icon-192.png');

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

// ---------------------------------------------------------------------------
// Copy (BACKLOG spec 068)
// ---------------------------------------------------------------------------

describe('ChatMessage — copy', () => {
  // CopyButton renders nothing without a clipboard API (that is its contract on
  // the insecure LAN surface), and jsdom ships none — so stub one, or these
  // assert the absence of a button that was never going to render.
  beforeEach(() => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async () => undefined },
      configurable: true,
      writable: true,
    });
  });

  it('user copy carries the stripped text, NOT the <attached_files> markup', () => {
    const raw =
      '<attached_files>\n- /w/report.pdf (2.0 MB)\n</attached_files>\n\nsummarise this';
    render(<ChatMessage message={userMsg(raw)} />);
    const btn = screen.getByTestId('user-message-copy');
    expect(btn).toBeInTheDocument();
    // The button holds no text itself; assert via the component contract that
    // the visible bubble text is what a copy would carry.
    expect(screen.getByText('summarise this')).toBeInTheDocument();
    expect(screen.queryByText(/attached_files/)).toBeNull();
  });

  it('agent rows get a copy button too', () => {
    render(<ChatMessage message={agentMsg('here is the answer')} />);
    expect(screen.getByTestId('agent-message-copy')).toBeInTheDocument();
  });

  it('a header-only agent row has NO copy button (it has no body to copy)', () => {
    const headerOnly = { ...agentMsg(''), isHeaderOnly: true } as ReturnType<typeof agentMsg>;
    render(<ChatMessage message={headerOnly} />);
    expect(screen.queryByTestId('agent-message-copy')).toBeNull();
  });
});

// ─── Spec 078 §5.4 step 3: the quotes block folds behind a chip ────────────

describe('ChatMessage — user_message with an appended Annotations block', () => {
  const QUOTES = [
    'Annotations:',
    '[1] Browser · "Queue" · box 10,20 100×40 (page pixels) — see attached annotation-1.png',
    '    note: click this one, not the ad',
    '[2] web/src/App.tsx lines 3–4',
    '    > const x = 1;',
  ].join('\n');

  it('shows the count as a chip and keeps the block collapsed', () => {
    render(<ChatMessage message={userMsg(`which one?\n\n${QUOTES}`)} />);
    expect(screen.getByText('which one?')).toBeInTheDocument();
    expect(screen.getByTestId('message-annotation-chip')).toHaveTextContent('2 annotations');
    // Collapsed: none of the machine markup is on screen.
    expect(screen.queryByTestId('message-annotation-quotes')).toBeNull();
    expect(screen.queryByText(/Annotations:/)).toBeNull();
  });

  it('expands the verbatim block on click', () => {
    render(<ChatMessage message={userMsg(`which one?\n\n${QUOTES}`)} />);
    fireEvent.click(screen.getByTestId('message-annotation-chip'));
    expect(screen.getByTestId('message-annotation-quotes')).toHaveTextContent(
      'click this one, not the ad',
    );
  });

  it('renders a chip on an annotation-only message (no typed text)', () => {
    render(<ChatMessage message={userMsg(QUOTES)} />);
    expect(screen.getByTestId('message-annotation-chip')).toHaveTextContent('2 annotations');
  });

  it('works alongside an <attached_files> prefix', () => {
    const content =
      '<attached_files>\n' +
      '- uploads/annotation-1.png (image/png, 12.0 KB)\n' +
      '</attached_files>\n\n' +
      `which one?\n\n${QUOTES}`;
    const { container } = render(<ChatMessage message={userMsg(content)} />);
    expect(container.textContent).toContain('annotation-1.png');
    expect(screen.getByTestId('message-annotation-chip')).toHaveTextContent('2 annotations');
    expect(screen.getByText('which one?')).toBeInTheDocument();
  });

  it('leaves a message with no block completely untouched', () => {
    render(<ChatMessage message={userMsg('Annotations: are a nice idea')} />);
    expect(screen.queryByTestId('message-annotation-chip')).toBeNull();
    expect(screen.getByText('Annotations: are a nice idea')).toBeInTheDocument();
  });
});

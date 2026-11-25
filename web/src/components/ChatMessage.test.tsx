// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ChatMessage from './ChatMessage';
import type { DisplayItem } from '../utils/chatTransform';
import { QUOTES_HEADING } from '../utils/annotations';
import { LocaleProvider } from '../i18n/LocaleContext';
import sourceSamples from '../../../tests/fixtures/source_task_samples.json';

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

describe('ChatMessage — an automation run\'s first message', () => {
  it('shows the task without the run header', () => {
    render(<ChatMessage message={userMsg("[Triggered by schedule 'Daily' (every day at 9)]\n\nSummarise the inbox")} />);
    expect(screen.getByText('Summarise the inbox')).toBeInTheDocument();
    expect(screen.queryByText(/Triggered by/)).not.toBeInTheDocument();
  });

  it('keeps the changed files of a file_watch run', () => {
    render(<ChatMessage message={userMsg("[Triggered by file_watch 'Inbox']\n\nChanged files: a.md\n\nFile them")} />);
    expect(screen.getByText(/Changed files: a\.md\s+File them/)).toBeInTheDocument();
    expect(screen.queryByText(/Triggered by/)).not.toBeInTheDocument();
  });
});

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

describe('ChatMessage — user_message with a <use_skill> block (spec 132)', () => {
  const SKILL =
    '<use_skill name="task-planning" path="/w/orbital/skills/task-planning/SKILL.md">\n' +
    'The user chose this skill for this message. Read the file at `path` before acting and follow it.\n' +
    '</use_skill>\n\n';

  it('renders a skill chip and only the typed text', () => {
    const { container } = render(<ChatMessage message={userMsg(SKILL + 'plan my week')} />);
    const chip = container.querySelector('[data-testid="message-skill-chip"]');
    expect(chip?.textContent).toBe('task-planning');
    expect(chip?.getAttribute('title')).toBe('/w/orbital/skills/task-planning/SKILL.md');
    expect(screen.getByText('plan my week')).toBeInTheDocument();
    expect(container.textContent ?? '').not.toContain('use_skill');
    expect(container.textContent ?? '').not.toContain('Read the file');
  });

  it('shows the skill chip next to attachment chips (skill block after attachments)', () => {
    const content =
      '<attached_files>\n- uploads/x.txt (text/plain, 11 B)\n</attached_files>\n\n' + SKILL + 'go';
    const { container } = render(<ChatMessage message={userMsg(content)} />);
    expect(container.querySelectorAll('[data-testid="attachment-chip"]').length).toBe(1);
    expect(container.querySelector('[data-testid="message-skill-chip"]')).not.toBeNull();
    expect(screen.getByText('go')).toBeInTheDocument();
  });

  it('copy carries the typed text, not the skill markup', async () => {
    const writes: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async (text: string) => { writes.push(text); } },
      configurable: true,
      writable: true,
    });
    render(<ChatMessage message={userMsg(SKILL + 'plan it')} />);
    fireEvent.click(screen.getByTestId('user-message-copy'));
    await Promise.resolve();
    expect(writes).toEqual(['plan it']);
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
    expect(content.className).toContain('bg-bubble');
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

    expect(screen.getByText('Agent')).toBeInTheDocument();
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
    expect(screen.queryByText('Agent')).not.toBeInTheDocument();
  });

  it('Test 2: a management message with empty agent_name falls back to "agent"', () => {
    render(<ChatMessage message={agentMsg('mgmt reply')} agentName="" />);
    expect(screen.getByText('Agent')).toBeInTheDocument();
  });

  it('Test 2b: a whitespace-only agent_name also falls back to "agent"', () => {
    render(<ChatMessage message={agentMsg('mgmt reply')} agentName="   " />);
    expect(screen.getByText('Agent')).toBeInTheDocument();
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

  it('an agent reply shows and copies its text without the [STATUS: …] tag', async () => {
    const writes: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async (text: string) => { writes.push(text); } },
      configurable: true,
      writable: true,
    });
    const { container } = render(<ChatMessage message={agentMsg('日报已写入。\n\n[STATUS: 完成]')} />);
    expect(container.textContent).toContain('日报已写入。');
    expect(container.textContent).not.toContain('STATUS');
    fireEvent.click(screen.getByTestId('agent-message-copy'));
    await Promise.resolve();
    expect(writes).toEqual(['日报已写入。']);
  });

  it('a reply that is only a tag shows its words, never the brackets', () => {
    const { container } = render(<ChatMessage message={agentMsg('[STATUS: 完成]')} />);
    expect(container.textContent).toContain('完成');
    expect(container.textContent).not.toContain('[');
  });

  it('a tag inside a code block stays', () => {
    const { container } = render(<ChatMessage message={agentMsg('Use `[STATUS: x]` in replies.')} />);
    expect(container.textContent).toContain('[STATUS: x]');
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
    QUOTES_HEADING,
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
    expect(screen.queryByText(QUOTES_HEADING)).toBeNull();
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

describe('ChatMessage — the agent body leaves room for card shadows', () => {
  it('widens its clip box with a matching bleed so the text does not move', () => {
    // overflow-x-auto clips a child's box-shadow at the body's edge; the
    // negative margin + equal padding moves that edge out without moving text.
    const { container } = render(<ChatMessage message={agentMsg('Wrote [about](about.html).')} />);
    const body = [...container.querySelectorAll('div')].find((d) => d.className.includes('overflow-x-auto'));
    expect(body).toBeTruthy();
    const cls = body!.className.split(/\s+/);
    for (const c of ['-mx-4', 'px-4', '-my-1.5', 'py-1.5']) expect(cls).toContain(c);
  });
});

// ─── Spec 156 §5.1/§5.2 + 164: the action row under the message ────────────

describe('ChatMessage — action row (spec 156 §5.1, 164)', () => {
  beforeEach(() => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async () => undefined },
      configurable: true,
      writable: true,
    });
  });

  it('copy sits in the row under the message, not on the label line', () => {
    const { container } = render(<ChatMessage message={userMsg('hello')} />);
    const row = screen.getByTestId('message-actions');
    expect(row).toContainElement(screen.getByTestId('user-message-copy'));
    // The label line keeps "you · HH:MM" and nothing else.
    const label = container.querySelector('.font-mono') as HTMLElement;
    expect(label.querySelector('button')).toBeNull();
    // Out of flow in the gap, so hover moves nothing; always on below md.
    expect(row.className).toContain('absolute');
    expect(row.className).toContain('top-full');
    expect(row.className).toContain('group-hover:opacity-100');
    expect(row.className).toContain('max-md:opacity-100');
  });

  it('copy on every message with a body: user, agent, sub-agent', () => {
    render(
      <>
        <ChatMessage message={userMsg('q')} />
        <ChatMessage message={agentMsg('a')} />
        <ChatMessage message={{ type: 'sub_agent_message', content: 'w', source: 'codex', timestamp: '2026-04-30T10:23:00Z' }} />
      </>,
    );
    expect(screen.getAllByTestId('user-message-copy')).toHaveLength(1);
    expect(screen.getAllByTestId('agent-message-copy')).toHaveLength(2);
  });

  it('an attachment-only message has no copy (no body)', () => {
    render(<ChatMessage message={userMsg('<attached_files>\n- uploads/x.txt (text/plain, 11 B)\n</attached_files>\n\n')} />);
    expect(screen.queryByTestId('user-message-copy')).toBeNull();
    expect(screen.queryByTestId('message-actions')).toBeNull();
  });

  it('修改 only on the user row, and only when ChatView allows it', () => {
    const onEdit = vi.fn();
    const { rerender } = render(<ChatMessage message={userMsg('q')} canEdit onEdit={onEdit} />);
    fireEvent.click(screen.getByTestId('message-edit'));
    expect(onEdit).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('message-edit')).toHaveAttribute('aria-label', 'Edit');
    // Running (or any other gate): ChatView passes canEdit=false; copy stays.
    rerender(<ChatMessage message={userMsg('q')} canEdit={false} onEdit={onEdit} />);
    expect(screen.queryByTestId('message-edit')).toBeNull();
    expect(screen.getByTestId('user-message-copy')).toBeInTheDocument();
    // An agent row never shows edit, whatever it is passed.
    rerender(<ChatMessage message={agentMsg('a')} canEdit onEdit={onEdit} />);
    expect(screen.queryByTestId('message-edit')).toBeNull();
  });

  it('order is copy · 重试 · 修改', () => {
    render(<ChatMessage message={userMsg('q')} canEdit onEdit={() => {}} canRetry onRetry={() => {}} />);
    const ids = Array.from(screen.getByTestId('message-actions').querySelectorAll('button')).map((b) =>
      b.getAttribute('data-testid'),
    );
    expect(ids).toEqual(['user-message-copy', 'message-retry', 'message-edit']);
  });

  it('spec 196 D1: the undo-since glyph only on the user row, last, when ChatView allows it', () => {
    const onUndoSince = vi.fn();
    const msg = userMsg('q');
    const { rerender } = render(
      <ChatMessage message={msg} canEdit onEdit={() => {}} canUndoSince onUndoSince={onUndoSince} />,
    );
    const ids = Array.from(screen.getByTestId('message-actions').querySelectorAll('button')).map((b) =>
      b.getAttribute('data-testid'),
    );
    expect(ids).toEqual(['user-message-copy', 'message-edit', 'message-undo-since']);
    fireEvent.click(screen.getByTestId('message-undo-since'));
    expect(onUndoSince).toHaveBeenCalledWith(msg);
    expect(screen.getByTestId('message-undo-since')).toHaveAttribute('title', 'Undo changes since this message');
    rerender(<ChatMessage message={msg} canUndoSince={false} onUndoSince={onUndoSince} />);
    expect(screen.queryByTestId('message-undo-since')).toBeNull();
    rerender(<ChatMessage message={agentMsg('a')} canUndoSince onUndoSince={onUndoSince} />);
    expect(screen.queryByTestId('message-undo-since')).toBeNull();
  });

  it('a header-only agent row gets no row at all, even as the retry anchor', () => {
    const headerOnly = { ...agentMsg(''), isHeaderOnly: true } as AgentMessage;
    render(<ChatMessage message={headerOnly} canRetry onRetry={() => {}} />);
    expect(screen.queryByTestId('message-actions')).toBeNull();
  });

  it('重试 waits for its send: a double click sends once', async () => {
    let release!: () => void;
    const onRetry = vi.fn(() => new Promise<void>((r) => { release = r; }));
    render(<ChatMessage message={agentMsg('a')} canRetry onRetry={onRetry} />);
    const btn = screen.getByTestId('message-retry');
    fireEvent.click(btn);
    fireEvent.click(btn);
    await Promise.resolve();
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(btn).toBeDisabled();
    release();
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.getByTestId('message-retry')).not.toBeDisabled();
  });
});

describe('ChatMessage — inline editor (spec 156 §5.2)', () => {
  const ATT = '<attached_files>\n- uploads/x.txt (text/plain, 11 B)\n</attached_files>\n\n';

  function renderEditor(content: string, extra: Partial<Parameters<typeof ChatMessage>[0]> = {}) {
    const onEditSend = vi.fn();
    const onEditCancel = vi.fn();
    render(
      <ChatMessage
        message={userMsg(content)}
        canEdit
        editing
        onEdit={() => {}}
        onEditSend={onEditSend}
        onEditCancel={onEditCancel}
        {...extra}
      />,
    );
    const input = screen.getByTestId('message-editor-input') as HTMLTextAreaElement;
    return { onEditSend, onEditCancel, input };
  }

  it('holds the body only, chips above it read-only, and no action row', () => {
    const { input } = renderEditor(ATT + 'read this');
    expect(input.value).toBe('read this');
    expect(screen.getByTestId('message-editor').querySelector('[data-testid="attachment-chip"]')).not.toBeNull();
    expect(screen.queryByTestId('message-actions')).toBeNull();
    expect(document.activeElement).toBe(input);
    expect(input.selectionStart).toBe('read this'.length);
  });

  it('Enter sends the edited text; Shift+Enter does not', () => {
    const { input, onEditSend } = renderEditor('old');
    fireEvent.change(input, { target: { value: 'new words ' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(onEditSend).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onEditSend).toHaveBeenCalledWith(expect.objectContaining({ content: 'old' }), 'new words');
  });

  it('Enter during IME composition never sends (isComposing or keyCode 229)', () => {
    const { input, onEditSend } = renderEditor('拼音');
    fireEvent.keyDown(input, { key: 'Enter', isComposing: true });
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 });
    expect(onEditSend).not.toHaveBeenCalled();
  });

  it('Esc and 取消 cancel', () => {
    const { input, onEditCancel } = renderEditor('old');
    fireEvent.keyDown(input, { key: 'Escape' });
    fireEvent.click(screen.getByTestId('message-editor-cancel'));
    expect(onEditCancel).toHaveBeenCalledTimes(2);
  });

  it('发送 is disabled on empty text with no chips; unchanged text still sends', () => {
    const { input, onEditSend } = renderEditor('same');
    fireEvent.click(screen.getByTestId('message-editor-send'));
    expect(onEditSend).toHaveBeenCalledWith(expect.anything(), 'same');
    fireEvent.change(input, { target: { value: '   ' } });
    expect(screen.getByTestId('message-editor-send')).toBeDisabled();
  });

  it('empty text with an attachment chip can still send', () => {
    const { input } = renderEditor(ATT + 'x');
    fireEvent.change(input, { target: { value: '' } });
    expect(screen.getByTestId('message-editor-send')).not.toBeDisabled();
  });

  it('发送 waits while the edit is in flight', () => {
    renderEditor('x', { editBusy: true });
    expect(screen.getByTestId('message-editor-send')).toBeDisabled();
  });

  it('desktop keeps the bubble cap; the phone gets the full column', () => {
    renderEditor('x');
    const cls = screen.getByTestId('message-editor').className;
    expect(cls).toContain('w-[min(70%,680px)]');
    expect(cls).toContain('max-md:w-full');
  });
});

// ---------------------------------------------------------------------------
// Spec 188 §2.5.2 — a routed task's first message shows the item
// ---------------------------------------------------------------------------

describe('ChatMessage — a routed task from a source', () => {
  const sample = (name: string) => {
    const s = (sourceSamples as { samples: Array<{ name: string; text: string }> }).samples.find((x) => x.name === name);
    if (!s) throw new Error(name);
    return s.text;
  };

  beforeEach(() => {
    localStorage.setItem('orbital.locale', 'zh');
  });
  afterEach(() => {
    localStorage.removeItem('orbital.locale');
  });

  const renderZh = (content: string, props: Partial<Parameters<typeof ChatMessage>[0]> = {}) =>
    render(<LocaleProvider><ChatMessage message={userMsg(content)} {...props} /></LocaleProvider>);

  it('shows the mark, the zh kind, the title once, the sender and the link; no wrapper', () => {
    const text = sample('github_label_long_zh_title');
    const { container } = renderZh(text);
    expect(container.querySelector('.src-mk[data-type="github"]')).not.toBeNull();
    const label = screen.getByTestId('source-task-label');
    expect(label.textContent).toContain('GitHub · 带标签');
    expect(label.textContent).not.toContain('你');
    const title = text.split('\n')[0];
    expect(screen.getAllByText(title)).toHaveLength(1);
    expect(screen.getByTestId('source-task-from').textContent).toBe('zqiren');
    const link = screen.getByTestId('source-task-link').querySelector('a');
    expect(link?.getAttribute('href')).toBe('https://github.com/zqiren/xiaoman-budget/issues/3');
    const dom = container.textContent ?? '';
    for (const wire of ['<source_item>', 'source_item', '[Task source', 'From:', 'At:', 'Title:', 'outside material']) {
      expect(dom).not.toContain(wire);
    }
  });

  it('a Feishu DM without a sender: no 你, no "direct message"', () => {
    const { container } = renderZh(sample('feishu_dm_no_sender'));
    expect(container.querySelector('.src-mk[data-type="feishu"]')).not.toBeNull();
    expect(screen.getByTestId('source-task-label').textContent).toContain('飞书 · 私聊机器人');
    expect(container.textContent).not.toContain('direct message');
    // The kind already says DM: no attribution line restating it.
    expect(screen.queryByTestId('source-task-from')).toBeNull();
  });

  it('a group @ keeps the attribution string as written', () => {
    renderZh(sample('feishu_group_at_with_sender'));
    expect(screen.getByTestId('source-task-from').textContent).toBe('王小明 · 小满产品群');
  });

  it('a DM sender under an unknown kind reads 私聊', () => {
    // The source's name is a kind word too, so the kind is ambiguous.
    const text = sample('telegram_dm_with_sender').replace('[Task source: Telegram · @acme_bot ·', '[Task source: Telegram · New mail ·');
    renderZh(text);
    expect(screen.getByTestId('source-task-from').textContent).toBe('Alice · 私聊');
  });

  it('merged updates render in order with their own time', () => {
    renderZh(sample('three_merged_parts'));
    const parts = screen.getAllByTestId('source-task-part');
    expect(parts).toHaveLength(3);
    const froms = screen.getAllByTestId('source-task-from').map((n) => n.textContent ?? '');
    expect(froms).toHaveLength(3);
    for (const f of froms) expect(f).toMatch(/^李雷 · (\d+\/\d+ )?\d\d:\d\d$/);
    expect(parts[1].textContent).toContain('控制台显示 500');
  });

  it('copy gives title + body + link', async () => {
    const writes: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async (s: string) => { writes.push(s); } },
      configurable: true,
      writable: true,
    });
    const text = sample('github_label_long_zh_title');
    renderZh(text);
    fireEvent.click(screen.getByTestId('user-message-copy'));
    await Promise.resolve();
    expect(writes).toHaveLength(1);
    expect(writes[0].startsWith(text.split('\n')[0] + '\n\nIssue in zqiren/xiaoman-budget')).toBe(true);
    expect(writes[0].endsWith('https://github.com/zqiren/xiaoman-budget/issues/3')).toBe(true);
    expect(writes[0]).not.toContain('source_item');
  });

  it('never offers edit or retry, even when the caller would allow them', () => {
    renderZh(sample('github_assigned_with_link'), { canEdit: true, canRetry: true, onEdit: () => {}, onRetry: () => {} });
    expect(screen.queryByTestId('message-edit')).toBeNull();
    expect(screen.queryByTestId('message-retry')).toBeNull();
  });

  it('English follows the UI language', () => {
    localStorage.setItem('orbital.locale', 'en');
    renderZh(sample('calendar_before_zh'));
    expect(screen.getByTestId('source-task-label').textContent).toContain('Calendar · Starting soon');
  });

  it('plain text that starts like the header renders as typed', () => {
    const text = sample('plain_text_starting_like_header');
    renderZh(text);
    expect(screen.queryByTestId('source-task-label')).toBeNull();
    expect(screen.getByText(/\[Task source: GitHub · mine\]/)).toBeInTheDocument();
  });
});

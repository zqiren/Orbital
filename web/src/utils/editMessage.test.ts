// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect, beforeEach } from 'vitest';
import {
  rebuildEditedContent,
  splitUserWire,
  forkSessionName,
  stashForkSend,
  takeForkSend,
  __resetForkSendsForTests,
} from './editMessage';
import { buildAttachmentsBlock } from '../lib/attachment-parsing';
import { buildInlineSkillBlock, buildSkillBlock } from '../lib/skill-parsing';
import { formatAskAnswer } from './askAnswer';
import { QUOTES_HEADING } from './annotations';
import { translate } from '../i18n/useT';

const ATT = buildAttachmentsBlock([
  { path: 'uploads/2026-11-10T080000-合同.pdf', mime: 'application/pdf', size: 20480 },
  { path: '/Users/me/notes.md', mime: 'text/markdown', size: 12, local: true },
]);
const SKILL = buildSkillBlock({ name: 'contract-review', path: '/w/orbital/skills/c/SKILL.md' });
const INLINE_SKILL = buildInlineSkillBlock('automate', '(built-in)', '# Automate\nUse create_trigger.');
const QUOTES = QUOTES_HEADING + '\n[1] notes.md lines 3–4\n    > first\n    >\n[2] Browser · "Docs" · box 1,2 3x4 (page pixels) — see attached annotation-2.png';

describe('rebuildEditedContent (spec 156 §5.2)', () => {
  it('plain text: the body is all there is', () => {
    expect(rebuildEditedContent('hello', 'bye')).toBe('bye');
    expect(splitUserWire('hello')).toEqual({ prefix: '', body: 'hello', quotesSep: '', quotes: null });
  });

  it('keeps the attachments block byte-for-byte', () => {
    const original = `${ATT}read this`;
    expect(rebuildEditedContent(original, 'read only page 2')).toBe(`${ATT}read only page 2`);
  });

  it('keeps a catalog skill pointer block', () => {
    expect(rebuildEditedContent(`${SKILL}review it`, 'review clause 9')).toBe(`${SKILL}review clause 9`);
  });

  it('keeps an inline (built-in / safe-mode) skill block', () => {
    expect(rebuildEditedContent(`${INLINE_SKILL}every day at 9`, 'every day at 10')).toBe(
      `${INLINE_SKILL}every day at 10`,
    );
  });

  it('keeps the ask-answer prefix', () => {
    const original = formatAskAnswer('a1b2c3', 'Which clause?', 'clause 4');
    expect(rebuildEditedContent(original, 'clause 9')).toBe(formatAskAnswer('a1b2c3', 'Which clause?', 'clause 9'));
  });

  it('keeps the quotes block after the body, with its separator', () => {
    const original = `look at these\n\n${QUOTES}`;
    expect(rebuildEditedContent(original, 'look again')).toBe(`look again\n\n${QUOTES}`);
  });

  it('quotes with no text: a new body joins them the way the composer does', () => {
    expect(rebuildEditedContent(QUOTES, 'why?')).toBe(`why?\n\n${QUOTES}`);
    expect(rebuildEditedContent(`text\n\n${QUOTES}`, '')).toBe(QUOTES);
  });

  it('all blocks together: only the body changes', () => {
    const body = '第一期 30%，第二期 40%。\n\n违约金 15%。';
    const original = ATT + SKILL + formatAskAnswer('abcdef', 'Q?', body) + `\n\n${QUOTES}`;
    const rebuilt = rebuildEditedContent(original, '只分两期。');
    expect(rebuilt).toBe(ATT + SKILL + formatAskAnswer('abcdef', 'Q?', '只分两期。') + `\n\n${QUOTES}`);
    expect(splitUserWire(original).body).toBe(body);
  });

  it('unchanged text gives back the original', () => {
    const original = ATT + SKILL + formatAskAnswer('abcdef', 'Q?', 'same words') + `\n\n${QUOTES}`;
    expect(rebuildEditedContent(original, splitUserWire(original).body)).toBe(original);
  });

  it('keeps a trigger header, drops a daemon turn note', () => {
    const header = "[Triggered by schedule 'Daily' (every day at 9)]\n\n";
    expect(rebuildEditedContent(`${header}do it`, 'do it twice')).toBe(`${header}do it twice`);
    expect(rebuildEditedContent('yes\n<orbital_turn_note>scan said X</orbital_turn_note>', 'no')).toBe('no');
  });
});

describe('forkSessionName (spec 156 §5.8)', () => {
  const at = new Date(2026, 10, 10, 14, 32).toISOString();
  const zh = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('zh', k, v);

  it('names the fork after the parent and the edited message time', () => {
    expect(forkSessionName('合同审阅', at, zh)).toBe('合同审阅 · 改自 14:32');
    expect(forkSessionName('Contract review', at)).toBe('Contract review · edited 14:32');
  });

  it('a fork of a fork keeps one suffix, in either language', () => {
    expect(forkSessionName('合同审阅 · 改自 09:05', at, zh)).toBe('合同审阅 · 改自 14:32');
    expect(forkSessionName('合同审阅 · edited 09:05', at, zh)).toBe('合同审阅 · 改自 14:32');
    expect(forkSessionName('Review · 改自 09:05 · edited 10:00', at)).toBe('Review · edited 14:32');
  });

  it('a name that merely contains the word is left alone', () => {
    expect(forkSessionName('edited 14:00 notes', at)).toBe('edited 14:00 notes · edited 14:32');
  });
});

describe('fork send stash', () => {
  beforeEach(() => __resetForkSendsForTests());

  it('is read once, per project and session', () => {
    stashForkSend('p1', 's2', { content: 'c', target: 'codex', body: 'b' });
    expect(takeForkSend('p2', 's2')).toBeNull();
    expect(takeForkSend('p1', 's2')).toEqual({ content: 'c', target: 'codex', body: 'b' });
    expect(takeForkSend('p1', 's2')).toBeNull();
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { formatAskAnswer, parseAskAnswer, stripTurnNote, displayAnswerText } from './askAnswer';

describe('stripTurnNote (spec 160 round 4)', () => {
  it('hides the one-turn daemon note; the answer still parses', () => {
    const wire = `${formatAskAnswer('abc123', '这样理解对吗？', '不对，是公众号连载')}\n\n`
      + '<orbital_turn_note>用户在回答你对这个项目的理解。只改 orbital/ 下的文件。</orbital_turn_note>';
    const shown = stripTurnNote(wire);
    expect(shown).not.toContain('orbital_turn_note');
    expect(parseAskAnswer(shown)).toEqual({ question: '这样理解对吗？', rest: '不对，是公众号连载' });
  });

  it('hides the picked-step approval behind a tap (round 5); the bubble shows the tap', () => {
    const wire = `${formatAskAnswer('abc123', '这样理解对吗？', '对，做第 1 步')}\n\n`
      + '<orbital_picked_step>对，按第 1 步开始：实现 dumps() 并补测试</orbital_picked_step>';
    const shown = stripTurnNote(wire);
    expect(shown).not.toContain('orbital_picked_step');
    expect(shown).not.toContain('dumps()');
    expect(parseAskAnswer(shown)).toEqual({ question: '这样理解对吗？', rest: '对，做第 1 步' });
  });

  it('leaves every other message byte-identical', () => {
    const plain = 'Fix the <b>tag</b> in README\n\nthanks';
    expect(stripTurnNote(plain)).toBe(plain);
  });
});

describe('displayAnswerText (batch 1110, left-off cards)', () => {
  it('shows the tapped word, not the wire prefix', () => {
    expect(displayAnswerText('[Answer to ask d0425a, already closed: "Which one: red or blue?"]\nred')).toBe('red');
  });
  it('drops a daemon turn note and keeps plain text as is', () => {
    expect(displayAnswerText('需要改 <orbital_turn_note>guard</orbital_turn_note>')).toBe('需要改');
    expect(displayAnswerText('Fix the login bug')).toBe('Fix the login bug');
  });
  it('never returns an empty quote', () => {
    expect(displayAnswerText('[Answer to ask d0425a, already closed: "Q?"]\n')).toBe('[Answer to ask d0425a, already closed: "Q?"]\n');
  });
});

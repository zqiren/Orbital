// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { formatAskAnswer, parseAskAnswer, stripTurnNote } from './askAnswer';

describe('stripTurnNote (spec 160 round 4)', () => {
  it('hides the one-turn daemon note; the answer still parses', () => {
    const wire = `${formatAskAnswer('abc123', '这样理解对吗？', '不对，是公众号连载')}\n\n`
      + '<orbital_turn_note>用户在回答你对这个项目的理解。只改 orbital/ 下的文件。</orbital_turn_note>';
    const shown = stripTurnNote(wire);
    expect(shown).not.toContain('orbital_turn_note');
    expect(parseAskAnswer(shown)).toEqual({ question: '这样理解对吗？', rest: '不对，是公众号连载' });
  });

  it('leaves every other message byte-identical', () => {
    const plain = 'Fix the <b>tag</b> in README\n\nthanks';
    expect(stripTurnNote(plain)).toBe(plain);
  });
});

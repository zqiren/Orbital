// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 188 §2.5.1 — the parser reads the daemon's real framing. The samples
 * are committed and pinned to framing.task_text by
 * tests/unit/test_source_task_samples.py.
 */

import { describe, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/source_task_samples.json';
import {
  parseSourceTask,
  sourceKindKey,
  sourceTaskCopyText,
  sourceTaskPartText,
  type SourceTask,
} from './source-task-parsing';

interface Sample {
  name: string;
  text: string;
  parsed: null | {
    type: string | null;
    kind: string | null;
    link: string | null;
    merged: number;
    parts: SourceTask['parts'];
  };
}

const samples = (fixture as { samples: Sample[] }).samples;
const byName = (n: string) => {
  const s = samples.find((x) => x.name === n);
  if (!s) throw new Error(`no sample ${n}`);
  return s;
};

describe('parseSourceTask on the committed samples', () => {
  for (const s of samples) {
    it(s.name, () => {
      const got = parseSourceTask(s.text);
      if (s.parsed === null) {
        expect(got).toBeNull();
        return;
      }
      expect(got).not.toBeNull();
      expect({
        type: got!.type,
        kind: got!.kind,
        link: got!.link,
        merged: got!.merged,
        parts: got!.parts,
      }).toEqual(s.parsed);
      expect(got!.title).toBe(s.text.split('\n')[0]);
    });
  }
});

describe('parseSourceTask edges', () => {
  it('a turn note stripped after the quote still parses (trailing space)', () => {
    expect(parseSourceTask(byName('feishu_dm_no_sender').text + '\n\n  ')).not.toBeNull();
  });

  it('extra text after the last part is not the framing', () => {
    expect(parseSourceTask(byName('github_assigned_with_link').text + '\nhello')).toBeNull();
  });

  it('an unknown header line is not the framing', () => {
    const t = byName('github_assigned_with_link').text.replace('Link: ', 'Linky: ');
    expect(parseSourceTask(t)).toBeNull();
  });

  it('every kind maps to a catalogue key', () => {
    for (const k of ['assigned', 'mention', 'label', 'review', 'dm', 'at', 'new_mail', 'new_event', 'before', 'down', 'item']) {
      expect(sourceKindKey(k)).not.toBeNull();
    }
    expect(sourceKindKey(null)).toBeNull();
  });
});

describe('what the bubble and copy show', () => {
  it('a chat message: the body does not repeat the title', () => {
    const task = parseSourceTask(byName('feishu_dm_no_sender').text)!;
    expect(sourceTaskPartText(task, 0)).toBe('重点写导出修复和新的预算提醒。');
    const one = parseSourceTask(byName('telegram_dm_with_sender').text)!;
    expect(sourceTaskPartText(one, 0)).toBe('');
  });

  it('copy is title + body + link, never the wire text', () => {
    const task = parseSourceTask(byName('github_label_long_zh_title').text)!;
    const copy = sourceTaskCopyText(task);
    expect(copy).toBe(
      `${task.title}\n\nIssue in zqiren/xiaoman-budget\n\n导出 11 月账单，CSV 里每一笔都乘了 100。\n\nhttps://github.com/zqiren/xiaoman-budget/issues/3`,
    );
    expect(copy).not.toContain('source_item');
    expect(copy).not.toContain('[Task source');
  });

  it('merged updates read in order, each once', () => {
    const task = parseSourceTask(byName('three_merged_parts').text)!;
    expect(task.parts.map((_, i) => sourceTaskPartText(task, i))).toEqual([
      '',
      '截图在这\n控制台显示 500',
      '只有 Safari 会这样，Chrome 正常',
    ]);
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { translate } from '../../i18n/useT';
import type { HistoryVersion, TurnFile } from '../../types';
import {
  authorOf,
  buildRows,
  changeLabel,
  dayLabel,
  defaultChecked,
  editorColor,
  isMemoryFile,
  splitTurnFiles,
  lastCapsuleOfTurns,
  legendAgents,
  memberKey,
  rowKeyForVersion,
  selectionFor,
  stripFileEditCounts,
  undoneCount,
} from './historyModel';

const zh = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('zh', k, v);
const en = (k: Parameters<typeof translate>[1], v?: Record<string, string | number>) => translate('en', k, v);

function ver(v: number, over: Partial<HistoryVersion> = {}): HistoryVersion {
  return {
    v,
    ts: '2026-11-04T14:30:00',
    kind: 'modify',
    sha: `s${v}`,
    prev_sha: `s${v - 1}`,
    size: 100,
    added: 2,
    removed: 1,
    lines: 3,
    attrib: { kind: 'tool', sessions: [{ session_id: 'S1', agents: ['codex'], name: '发布 v0.20.0' }] },
    ambiguous: false,
    name: null,
    turn: { session_id: 'S1', from_ts: 1, to_ts: null },
    group: `g${v}`,
    current: false,
    ...over,
  };
}

describe('buildRows — one row per agent turn, one version per write', () => {
  it('folds consecutive versions of one group into one row, newest first', () => {
    const versions = [
      ver(5, { group: 'g3', current: true }),
      ver(4, { group: 'g3' }),
      ver(3, { group: 'g3' }),
      ver(2, { group: 'g2' }),
      ver(1, { kind: 'baseline', group: 'g1', turn: null }),
    ];
    const rows = buildRows(versions);
    expect(rows.map((r) => r.members.map((m) => m.v))).toEqual([[5, 4, 3], [2], [1]]);
    expect(rows[0].current).toBe(true);
    // The view of a folded turn diffs against the version before the turn.
    expect(rows[0].before?.v).toBe(2);
    expect(rows[2].before).toBeNull();
  });

  it('selection: null = the current row; a folded row views from the version before it', () => {
    const rows = buildRows([ver(3, { group: 'g2', current: true }), ver(2, { group: 'g2' }), ver(1, { group: 'g1' })]);
    expect(selectionFor(rows, null)).toMatchObject({ isCurrent: true, fromV: 1 });
    expect(selectionFor(rows, rows[1].key)).toMatchObject({ isCurrent: false, fromV: null });
    expect(selectionFor(rows, memberKey(rows[0].members[1]))?.target.v).toBe(2);
    expect(rowKeyForVersion(rows, 2)).toBe(rows[0].key);
  });
});

describe('labels', () => {
  it('改了 N 行 sums a folded turn; a restore says where it came from', () => {
    const rows = buildRows([ver(3, { group: 'g', lines: 5 }), ver(2, { group: 'g', lines: 4 })]);
    expect(changeLabel(rows[0], zh, null)).toBe('改了 9 行');
    expect(changeLabel(rows[0], en, null)).toBe('Changed 9 lines');
    const restore = buildRows([
      ver(9, { kind: 'restore', attrib: { kind: 'restore', by: 'user', from_v: 2, from_ts: '2026-11-04T14:32:00' }, turn: null }),
    ]);
    expect(changeLabel(restore[0], zh, null)).toBe('恢复自 14:32 的版本');
    expect(authorOf(restore[0].head, zh)).toMatchObject({ you: true, label: '你' });
  });

  it('office files show the size change instead of lines', () => {
    const rows = buildRows([ver(2, { lines: null, added: null, removed: null, size: 48132 }), ver(1, { size: 4632 })]);
    expect(changeLabel(rows[0], en, rows[1].head)).toBe('4.5 KB → 47.0 KB');
  });

  it('two sessions at once: both logos and the "one of these" tooltip', () => {
    const v = ver(2, {
      ambiguous: true,
      attrib: {
        kind: 'running',
        ambiguous: true,
        sessions: [
          { session_id: 'A', agents: ['claude-code'], name: 'A' },
          { session_id: 'B', agents: ['codex'], name: 'B' },
        ],
      },
    });
    const a = authorOf(v, zh);
    expect(a.agents).toEqual(['claude-code', 'codex']);
    expect(a.either).toBe('A 或 B（同时在运行）');
  });

  it('day labels: 今天 / 昨天 / 11月2日', () => {
    const now = new Date(2026, 10, 4, 16, 0);
    expect(dayLabel('2026-11-04T09:00:00', 'zh', zh, now)).toBe('今天');
    expect(dayLabel('2026-11-03T09:00:00', 'zh', zh, now)).toBe('昨天');
    expect(dayLabel('2026-11-02T09:00:00', 'zh', zh, now)).toBe('11月2日');
    expect(dayLabel('2026-11-02T09:00:00', 'en', en, now)).toBe('Nov 2');
  });
});

describe('palette', () => {
  it('tints by the writing agent; ambiguous and outside versions are grey and stay out of the legend', () => {
    const base = { v: 1, ts: '', kind: 'tool' as const, session_id: 'S', session_name: null };
    expect(editorColor({ ...base, agents: ['codex'], ambiguous: false, neutral: false })).toContain('codex');
    expect(editorColor({ ...base, agents: ['orbital', 'claude-code'], ambiguous: false, neutral: false })).toContain('claude-code');
    expect(editorColor({ ...base, agents: ['codex', 'pi'], ambiguous: true, neutral: true })).toContain('you');
    expect(
      legendAgents([
        { ...base, agents: ['codex'], ambiguous: false, neutral: false },
        { ...base, agents: [], ambiguous: false, neutral: true },
        { ...base, agents: ['codex'], ambiguous: false, neutral: false },
      ]),
    ).toEqual(['codex']);
  });
});

function tf(over: Partial<TurnFile> = {}): TurnFile {
  return {
    path: 'a.md', kind: 'modify', added: 1, removed: 1, lines: 2,
    before_sha: 'b', turn_sha: 't', current_sha: 't', before_size: 1, turn_size: 1,
    clean: true, later_editor: null, ambiguous: false, whole_file_only: false,
    first_v: 2, last_v: 3,
    editor: { v: 3, ts: '', kind: 'tool', agents: ['codex'], ambiguous: false, session_id: 'S', session_name: 'x', neutral: false },
    ...over,
  };
}

describe('undo sheet defaults (§5.9, §7)', () => {
  it('checks untouched files; never a created file, a later-edited file or an ambiguous one', () => {
    expect(defaultChecked(tf())).toBe(true);
    expect(defaultChecked(tf({ clean: false, current_sha: 'x' }))).toBe(false);
    expect(defaultChecked(tf({ kind: 'create', before_sha: null }))).toBe(false);
    expect(defaultChecked(tf({ ambiguous: true }))).toBe(false);
  });

  it('memory files (Layer-1) are never pre-checked and are split out of the turn', () => {
    expect(isMemoryFile('orbital/PROJECT_STATE.md')).toBe(true);
    expect(isMemoryFile('orbital/INDEX.md')).toBe(true);
    expect(isMemoryFile('orbital/DECISIONS.md')).toBe(true);
    expect(isMemoryFile('orbital/LESSONS.md')).toBe(true);
    expect(isMemoryFile('docs/PROJECT_STATE.md')).toBe(false);
    expect(defaultChecked(tf({ path: 'orbital/PROJECT_STATE.md' }))).toBe(false);
    const { work, memory } = splitTurnFiles([tf({ path: 'orbital/INDEX.md' }), tf({ path: 'a.md' })]);
    expect(work.map((f) => f.path)).toEqual(['a.md']);
    expect(memory.map((f) => f.path)).toEqual(['orbital/INDEX.md']);
    expect(undoneCount([tf({ path: 'orbital/INDEX.md', clean: false, current_sha: 'b' })])).toBe(0);
  });

  it('counts files already back at their pre-turn content as undone', () => {
    expect(undoneCount([tf(), tf({ clean: false, current_sha: 'b' })])).toBe(1);
  });
});

describe('capsule', () => {
  it('drops only the file-writing counts from the breakdown', () => {
    expect(stripFileEditCounts({ edit: 2, write: 1, read: 3, shell: 1, Edit: 1, apply_patch: 1 })).toEqual({
      read: 3,
      shell: 1,
    });
  });

  it('maps each turn to its last capsule (not the memory record) with the next user row as the end', () => {
    const items = [
      { type: 'user_message', timestamp: 'T1' },
      { type: 'agent_run', capsule_id: 'c1' },
      { type: 'agent_message', timestamp: 'x' },
      { type: 'agent_run', capsule_id: 'c2' },
      { type: 'agent_run', capsule_id: 'mem', memory_record: true },
      { type: 'user_message', timestamp: 'T2' },
      { type: 'agent_run', capsule_id: 'c3' },
    ];
    const spans = lastCapsuleOfTurns(items);
    expect([...spans.keys()]).toEqual(['c2', 'c3']);
    expect(spans.get('c2')).toEqual({ fromTs: 'T1', toTs: 'T2' });
    expect(spans.get('c3')).toEqual({ fromTs: 'T2', toTs: null });
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { classifySessionName } from './sessionLabel';

describe('classifySessionName', () => {
  // ---------------------------------------------------------------- plain
  it('plain name passes through', () => {
    expect(classifySessionName('write an essay about the ocean')).toEqual({
      kind: 'plain',
      displayName: 'write an essay about the ocean',
    });
  });

  it('null/empty name → plain with null displayName', () => {
    expect(classifySessionName(null)).toEqual({ kind: 'plain', displayName: null });
    expect(classifySessionName('   ')).toEqual({ kind: 'plain', displayName: null });
  });

  // ---------------------------------------------------------------- queue
  it('queue via origin field, machine name → chip only (null displayName)', () => {
    const r = classifySessionName('[QUEUE ITEM | id=item_ab12 | attempt=1] You are…', 'queue');
    expect(r.kind).toBe('queue');
    expect(r.displayName).toBeNull();
  });

  it('queue via name prefix alone (origin absent, e.g. stale list rows)', () => {
    const r = classifySessionName('[QUEUE ITEM | id=item_ab12 | atte…');
    expect(r.kind).toBe('queue');
    expect(r.displayName).toBeNull();
  });

  it('renamed queue session keeps the queue kind but shows the rename', () => {
    const r = classifySessionName('Reddit 线索整理', 'queue');
    expect(r.kind).toBe('queue');
    expect(r.displayName).toBe('Reddit 线索整理');
  });

  // -------------------------------------------------------------- triggers
  it('schedule trigger → kind schedule, trigger name extracted', () => {
    const r = classifySessionName("[Triggered by schedule 'Daily check' (every d…");
    expect(r.kind).toBe('schedule');
    expect(r.displayName).toBe('Daily check');
    expect(r.detail).toBe("schedule 'Daily check'");
  });

  it('file_watch trigger → kind file_watch', () => {
    const r = classifySessionName("[Triggered by file_watch 'specs watcher']\nChang…");
    expect(r.kind).toBe('file_watch');
    expect(r.displayName).toBe('specs watcher');
  });

  it('trigger name cut mid-quote by the 50-char cap still extracts', () => {
    const r = classifySessionName("[Triggered by schedule 'Very long trigger nam…");
    expect(r.kind).toBe('schedule');
    expect(r.displayName).toBe('Very long trigger nam');
  });

  it('trigger truncated before any name chars → null displayName', () => {
    const r = classifySessionName("[Triggered by schedule '");
    expect(r.kind).toBe('schedule');
    expect(r.displayName).toBeNull();
    expect(r.detail).toBeUndefined();
  });

  // ----------------------------------------------------------- attachments
  it('attachment-first name → basename with upload timestamp stripped', () => {
    const r = classifySessionName('<attached_files>\n- /uploads/2026-08-12T053000-report.pdf (app…');
    expect(r.kind).toBe('attachment');
    expect(r.displayName).toBe('report.pdf');
  });

  it('attachment path cut before the mime paren still yields a basename', () => {
    const r = classifySessionName('<attached_files>\n- /uploads/2026-08-12T053000-repor…');
    expect(r.kind).toBe('attachment');
    expect(r.displayName).toBe('repor');
  });

  it('attachment block cut before the path → null displayName', () => {
    const r = classifySessionName('<attached_files>');
    expect(r.kind).toBe('attachment');
    expect(r.displayName).toBeNull();
  });

  it('queue item with staged files → queue kind, attachment basename shown', () => {
    const r = classifySessionName(
      '<attached_files>\n- data/2026-08-12T053000-notes.md (te…',
      'queue',
    );
    expect(r.kind).toBe('queue');
    expect(r.displayName).toBe('notes.md');
  });

  it('non-leading markup is user text, not a machine prefix', () => {
    const r = classifySessionName("please explain [Triggered by schedule 'x'] syntax");
    expect(r.kind).toBe('plain');
    expect(r.displayName).toBe("please explain [Triggered by schedule 'x'] syntax");
  });
});

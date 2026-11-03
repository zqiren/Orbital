// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { parseTriggerHeader } from './trigger-parsing';

describe('parseTriggerHeader', () => {
  it('strips a schedule header and keeps the task', () => {
    const r = parseTriggerHeader("[Triggered by schedule 'Daily' (every day at 9)]\n\nSummarise the inbox");
    expect(r.trigger).toEqual({ kind: 'schedule', name: 'Daily', schedule: 'every day at 9' });
    expect(r.strippedContent).toBe('Summarise the inbox');
  });

  it('strips a file_watch header and keeps the changed files', () => {
    const r = parseTriggerHeader("[Triggered by file_watch 'Inbox']\n\nChanged files: a.md, b.md\n\nFile them");
    expect(r.trigger).toEqual({ kind: 'file_watch', name: 'Inbox', schedule: null });
    expect(r.strippedContent).toBe('Changed files: a.md, b.md\n\nFile them');
  });

  it('leaves an ordinary message alone', () => {
    const r = parseTriggerHeader('hello');
    expect(r.trigger).toBeNull();
    expect(r.strippedContent).toBe('hello');
  });

  it('only matches at the very start', () => {
    const r = parseTriggerHeader("note: [Triggered by schedule 'x']");
    expect(r.trigger).toBeNull();
  });
});

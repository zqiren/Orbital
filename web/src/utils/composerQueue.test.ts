// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import {
  chipsFromWire, fromPendingWire, mergeIntoDraft, queueRows,
  type OwnQueued, type WorkerQueued,
} from './composerQueue';

const own = (sessionId: string, text: string, queuedAt: string): OwnQueued => ({
  sessionId, text, chips: [], editable: true, queuedAt,
});
const worker = (dispatchId: string, sessionId: string, text: string, queuedAt: string): WorkerQueued => ({
  dispatchId, handle: 'codex', sessionId, text, chips: [], editable: true, queuedAt,
});

describe('queueRows', () => {
  it('shows both kinds for the viewed session, oldest first', () => {
    const rows = queueRows(
      new Map([
        ['n2', own('s1', 'second', '2026-11-24T10:00:02Z')],
        ['n9', own('other', 'elsewhere', '2026-11-24T10:00:00Z')],
        ['n1', own('s1', 'first', '2026-11-24T10:00:01Z')],
      ]),
      [worker('d1', 's1', 'to codex', '2026-11-24T10:00:03Z'), worker('d2', 'other', 'x', '2026-11-24T10:00:00Z')],
      's1',
    );
    expect(rows.map((r) => r.key)).toEqual(['n:n1', 'n:n2', 'd:d1']);
    expect(rows[2]).toMatchObject({ kind: 'worker', handle: 'codex', text: 'to codex' });
  });

  it('no session, no rows', () => {
    expect(queueRows(new Map([['n', own('s1', 'a', 't')]]), [], undefined)).toEqual([]);
  });
});

describe('fromPendingWire', () => {
  it('uses the typed words and the exact attachments the daemon kept', () => {
    const out = fromPendingWire({
      content: '<attached_files>\n- uploads/a.png (image/png, 3 B)\n</attached_files>\n\nlook',
      text: 'look',
      attachments: [{ path: 'uploads/a.png', mime: 'image/png', size: 3 }],
    });
    expect(out).toEqual({
      text: 'look',
      chips: [{ filename: 'a.png', mime: 'image/png', size: 3, uploadedPath: 'uploads/a.png' }],
      editable: true,
    });
  });

  it('falls back to the wire words for an entry without them', () => {
    expect(fromPendingWire({ content: 'plain words' })).toEqual({ text: 'plain words', chips: [], editable: true });
  });

  it('an old entry whose chips cannot be rebuilt is not editable, but still shows its words', () => {
    const out = fromPendingWire({
      content: '<attached_files>\n- /tmp/a.png (image/png, 1.2 KB)\n</attached_files>\n\nplease look',
    });
    expect(out.text).toBe('please look');
    expect(out.editable).toBe(false);
  });

  it('an upload shows the name the user picked, not its stored name', () => {
    expect(chipsFromWire([{ path: 'uploads/2026-11-24T210316-shot.png', mime: 'image/png', size: 3 }])[0].filename)
      .toBe('shot.png');
  });

  it('a local file comes back as a local chip', () => {
    expect(chipsFromWire([{ path: '/Users/me/r.pdf', mime: 'application/pdf', size: 9, local: true }])).toEqual([
      { filename: 'r.pdf', mime: 'application/pdf', size: 9, localPath: '/Users/me/r.pdf' },
    ]);
  });
});

describe('mergeIntoDraft', () => {
  it('fills an empty composer', () => {
    expect(mergeIntoDraft('', 'queued')).toBe('queued');
    expect(mergeIntoDraft('  ', 'queued')).toBe('queued');
  });
  it('keeps a draft and puts the message on a new line after it', () => {
    expect(mergeIntoDraft('half typed ', 'queued')).toBe('half typed\nqueued');
  });
});

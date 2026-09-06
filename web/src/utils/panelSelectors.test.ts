// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §6 / §10 — the panel's pure selectors.
 *
 * Fixtures mirror the shape of a real session JSONL (verified against
 * `orbital-data/p5-cred/orbital/sessions/p5-cred_7c20adc6.jsonl`): the tool
 * names are the bare `read` / `write` / `edit` / `browser`, arguments are a
 * JSON *string*, and the browser's screenshot arrives on the tool RESULT row's
 * `_meta` as an absolute path.
 */
import { describe, expect, it } from 'vitest';
import type { ActivityEvent, ChatMessage } from '../types';
import {
  eventAsMessage,
  latestScreenshot,
  pathForEvent,
  touchedFiles,
  viewForEvent,
} from './panelSelectors';

function toolCall(name: string, args: Record<string, unknown>, at = '2026-09-03T10:00:00Z'): ChatMessage {
  return {
    role: 'assistant',
    content: null,
    source: 'management',
    timestamp: at,
    tool_calls: [
      { id: `call_${name}_${at}`, type: 'function', function: { name, arguments: JSON.stringify(args) } },
    ],
  };
}

function toolResult(meta: Record<string, unknown>, at = '2026-09-03T10:00:01Z'): ChatMessage {
  return {
    role: 'tool',
    content: 'ok',
    source: 'management',
    timestamp: at,
    tool_call_id: 'call_1',
    _meta: meta,
  };
}

function activity(overrides: Partial<ActivityEvent>): ActivityEvent {
  return {
    type: 'agent.activity',
    project_id: 'proj-1',
    id: 'evt-1',
    category: 'tool_use',
    description: 'Used tool',
    tool_name: 'read',
    source: 'management',
    timestamp: '2026-09-03T10:00:00Z',
    ...overrides,
  };
}

describe('touchedFiles', () => {
  it('collects read / write / edit calls, newest first', () => {
    const files = touchedFiles([
      toolCall('read', { path: 'a.md' }, '2026-09-03T10:00:00Z'),
      toolCall('write', { path: 'b.md' }, '2026-09-03T10:00:01Z'),
      toolCall('edit', { path: 'c.md' }, '2026-09-03T10:00:02Z'),
    ]);
    expect(files.map((f) => f.path)).toEqual(['c.md', 'b.md', 'a.md']);
    expect(files.map((f) => f.op)).toEqual(['edited', 'written', 'read']);
  });

  it('emits one row per path and lets the strongest op win (written > edited > read)', () => {
    const files = touchedFiles([
      toolCall('write', { path: 'x.ts' }),
      toolCall('edit', { path: 'x.ts' }),
      toolCall('read', { path: 'x.ts' }),
    ]);
    expect(files).toHaveLength(1);
    expect(files[0]).toMatchObject({ path: 'x.ts', op: 'written' });
  });

  it('escalates a read to written when a later write touches the same path', () => {
    const files = touchedFiles([
      toolCall('read', { path: 'x.ts' }),
      toolCall('write', { path: 'x.ts' }),
    ]);
    expect(files[0].op).toBe('written');
  });

  it('re-sorts a path to the front when it is touched again', () => {
    const files = touchedFiles([
      toolCall('read', { path: 'first.md' }, '2026-09-03T10:00:00Z'),
      toolCall('read', { path: 'second.md' }, '2026-09-03T10:00:01Z'),
      toolCall('read', { path: 'first.md' }, '2026-09-03T10:00:02Z'),
    ]);
    expect(files.map((f) => f.path)).toEqual(['first.md', 'second.md']);
    expect(files[0].lastAt).toBe('2026-09-03T10:00:02Z');
  });

  it('accepts the legacy `file_path` argument name', () => {
    expect(touchedFiles([toolCall('read', { file_path: 'legacy.md' })])).toEqual([
      { path: 'legacy.md', op: 'read', lastAt: '2026-09-03T10:00:00Z' },
    ]);
  });

  it('ignores non-file tools, unparseable arguments, and pathless calls', () => {
    const broken: ChatMessage = {
      role: 'assistant',
      content: null,
      source: 'management',
      timestamp: '2026-09-03T10:00:00Z',
      tool_calls: [{ id: 'c', type: 'function', function: { name: 'read', arguments: '{not json' } }],
    };
    expect(
      touchedFiles([
        toolCall('shell', { command: 'ls' }),
        toolCall('browser', { action: 'navigate', url: 'https://x' }),
        toolCall('grep', { pattern: 'x', path: 'src' }),
        toolCall('read', {}),
        broken,
      ]),
    ).toEqual([]);
  });

  it('returns [] for a session with no tool calls', () => {
    expect(touchedFiles([])).toEqual([]);
    expect(
      touchedFiles([{ role: 'user', content: 'hi', source: 'user', timestamp: '' }]),
    ).toEqual([]);
  });
});

describe('latestScreenshot', () => {
  it('reads the newest tool-result _meta.screenshot_path with url and title', () => {
    expect(
      latestScreenshot([
        toolResult({ url: 'https://a', title: 'A', screenshot_path: '/ws/shots/1.png' }),
        toolCall('read', { path: 'x.md' }),
        toolResult({ url: 'https://b', title: 'B', screenshot_path: '/ws/shots/2.png' }),
      ]),
    ).toEqual({ path: '/ws/shots/2.png', url: 'https://b', title: 'B' });
  });

  it('skips result rows that carry _meta but no screenshot (snapshot rows)', () => {
    expect(
      latestScreenshot([
        toolResult({ url: 'https://a', title: 'A', screenshot_path: '/ws/shots/1.png' }),
        toolResult({ url: 'https://a', title: 'A', snapshot_stats: { lines: 16 } }),
      ]),
    ).toMatchObject({ path: '/ws/shots/1.png' });
  });

  it('omits url/title when they are absent or not strings', () => {
    expect(latestScreenshot([toolResult({ screenshot_path: '/ws/1.png', title: 42 })])).toEqual({
      path: '/ws/1.png',
      url: undefined,
      title: undefined,
    });
  });

  it('returns null when the session has no screenshot', () => {
    expect(latestScreenshot([])).toBeNull();
    expect(latestScreenshot([toolCall('read', { path: 'x' })])).toBeNull();
  });
});

describe('viewForEvent', () => {
  const cases: [ActivityEvent['category'], string, 'files' | 'browser' | null][] = [
    ['file_read', 'read', 'files'],
    ['file_write', 'write', 'files'],
    ['file_edit', 'edit', 'files'],
    ['browser_automation', 'browser', 'browser'],
    ['command_exec', 'shell', null],
    ['web_search', 'web_search', null],
    ['file_search', 'glob', null],
    ['content_search', 'grep', null],
    ['tool_result', 'call_abc', null],
    ['agent_output', '', null],
  ];

  it.each(cases)('category %s (tool %s) → %s', (category, tool_name, expected) => {
    expect(viewForEvent(activity({ category, tool_name }))).toBe(expected);
  });

  it('falls back to the tool name when an older daemon sends category tool_use', () => {
    expect(viewForEvent(activity({ category: 'tool_use', tool_name: 'browser' }))).toBe('browser');
    expect(viewForEvent(activity({ category: 'tool_use', tool_name: 'edit' }))).toBe('files');
    expect(viewForEvent(activity({ category: 'tool_use', tool_name: 'mystery' }))).toBeNull();
  });
});

describe('pathForEvent', () => {
  it('returns the path a file event names', () => {
    expect(
      pathForEvent(activity({ category: 'file_edit', tool_name: 'edit', arguments: { path: 'src/a.ts' } })),
    ).toBe('src/a.ts');
  });

  it('is null for browser events and for events without parsed arguments', () => {
    expect(
      pathForEvent(activity({ category: 'browser_automation', tool_name: 'browser', arguments: { url: 'x' } })),
    ).toBeNull();
    expect(pathForEvent(activity({ category: 'file_read', tool_name: 'read' }))).toBeNull();
  });
});

describe('eventAsMessage', () => {
  it('replays a live file event through touchedFiles', () => {
    const event = activity({
      category: 'file_write',
      tool_name: 'write',
      arguments: { path: 'live.md' },
    });
    const message = eventAsMessage(event);
    expect(message).not.toBeNull();
    expect(touchedFiles([message as ChatMessage])).toEqual([
      { path: 'live.md', op: 'written', lastAt: event.timestamp },
    ]);
  });

  it('is null for non-file tools and for events with no arguments', () => {
    expect(eventAsMessage(activity({ tool_name: 'browser', arguments: { action: 'click' } }))).toBeNull();
    expect(eventAsMessage(activity({ tool_name: 'read' }))).toBeNull();
  });
});

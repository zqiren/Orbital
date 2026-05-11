// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import {
  buildAttachmentsBlock,
  parseAttachmentsBlock,
} from './attachment-parsing';

describe('parseAttachmentsBlock', () => {
  it('parses a single bullet line + trailing user text', () => {
    const input =
      '<attached_files>\n- uploads/foo.png (image/png, 1.0 KB)\n</attached_files>\n\nhello!';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe('hello!');
    expect(out.attachments).toEqual([
      { path: 'uploads/foo.png', mime: 'image/png', sizeLabel: '1.0 KB' },
    ]);
  });

  it('parses multiple bullet lines', () => {
    const input =
      '<attached_files>\n' +
      '- uploads/a.png (image/png, 1.0 KB)\n' +
      '- uploads/b.txt (text/plain, 512 B)\n' +
      '</attached_files>\n\n' +
      'check these';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe('check these');
    expect(out.attachments).toHaveLength(2);
    expect(out.attachments[0]).toEqual({
      path: 'uploads/a.png',
      mime: 'image/png',
      sizeLabel: '1.0 KB',
    });
    expect(out.attachments[1]).toEqual({
      path: 'uploads/b.txt',
      mime: 'text/plain',
      sizeLabel: '512 B',
    });
  });

  it('returns "" for stripped content when there is no user text after', () => {
    const input =
      '<attached_files>\n- uploads/x.txt (text/plain, 1 B)\n</attached_files>\n\n';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe('');
    expect(out.attachments).toHaveLength(1);
  });

  it('returns unchanged content when there is no block', () => {
    const input = 'hello world';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe('hello world');
    expect(out.attachments).toEqual([]);
  });

  it('does not parse a block that is not at position 0', () => {
    const input =
      'I typed this then a fake block:\n' +
      '<attached_files>\n- uploads/x.txt (text/plain, 1 B)\n</attached_files>\n\n';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe(input);
    expect(out.attachments).toEqual([]);
  });

  it('skips malformed bullet lines without throwing', () => {
    const input =
      '<attached_files>\n' +
      '- uploads/ok.txt (text/plain, 10 B)\n' +
      '- malformed without parens\n' +
      '- another/ok.png (image/png, 2.0 KB)\n' +
      '</attached_files>\n\n' +
      'text';
    const out = parseAttachmentsBlock(input);
    expect(out.strippedContent).toBe('text');
    expect(out.attachments).toEqual([
      { path: 'uploads/ok.txt', mime: 'text/plain', sizeLabel: '10 B' },
      { path: 'another/ok.png', mime: 'image/png', sizeLabel: '2.0 KB' },
    ]);
  });

  it('tolerates extra whitespace inside bullet lines', () => {
    const input =
      '<attached_files>\n' +
      '-   uploads/spaced.txt   (text/plain, 100 B)\n' +
      '</attached_files>\n\n';
    const out = parseAttachmentsBlock(input);
    expect(out.attachments).toEqual([
      { path: 'uploads/spaced.txt', mime: 'text/plain', sizeLabel: '100 B' },
    ]);
  });
});

describe('buildAttachmentsBlock', () => {
  it('mirrors the backend output for a single attachment', () => {
    const block = buildAttachmentsBlock([
      { path: 'uploads/foo.png', mime: 'image/png', size: 1024 },
    ]);
    expect(block).toBe(
      '<attached_files>\n- uploads/foo.png (image/png, 1.0 KB)\n</attached_files>\n\n',
    );
  });

  it('mirrors the backend output for multiple attachments', () => {
    const block = buildAttachmentsBlock([
      { path: 'uploads/a.png', mime: 'image/png', size: 1024 },
      { path: 'uploads/b.txt', mime: 'text/plain', size: 512 },
    ]);
    expect(block).toBe(
      '<attached_files>\n' +
        '- uploads/a.png (image/png, 1.0 KB)\n' +
        '- uploads/b.txt (text/plain, 512 B)\n' +
        '</attached_files>\n\n',
    );
  });

  it('returns "" for an empty list', () => {
    expect(buildAttachmentsBlock([])).toBe('');
  });

  it('round-trips parse(build(x))', () => {
    const original = [
      { path: 'uploads/a.png', mime: 'image/png', size: 1024 },
      { path: 'uploads/b.txt', mime: 'text/plain', size: 512 },
    ];
    const block = buildAttachmentsBlock(original);
    const out = parseAttachmentsBlock(block + 'hi there');
    expect(out.strippedContent).toBe('hi there');
    expect(out.attachments).toEqual([
      { path: 'uploads/a.png', mime: 'image/png', sizeLabel: '1.0 KB' },
      { path: 'uploads/b.txt', mime: 'text/plain', sizeLabel: '512 B' },
    ]);
  });
});

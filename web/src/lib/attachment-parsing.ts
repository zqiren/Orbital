// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { humanSize } from './attachment-upload';

export interface ParsedAttachment {
  path: string;
  mime: string;
  sizeLabel: string;
}

export interface ParseAttachmentsResult {
  strippedContent: string;
  attachments: ParsedAttachment[];
}

const BLOCK_RE = /^<attached_files>\n([\s\S]*?)\n<\/attached_files>\n\n?/;

/**
 * Parse a leading <attached_files>...</attached_files> block from `content`.
 *
 * Only recognized when the block is the very first thing in the string —
 * elsewhere it would be user-authored text, not the injected prefix.
 */
export function parseAttachmentsBlock(content: string): ParseAttachmentsResult {
  const match = BLOCK_RE.exec(content);
  if (!match) {
    return { strippedContent: content, attachments: [] };
  }

  const body = match[1];
  const attachments: ParsedAttachment[] = [];

  for (const rawLine of body.split('\n')) {
    const line = rawLine.trim();
    if (!line.startsWith('-')) continue;
    const stripped = line.slice(1).trim();
    // Expected form: "<path> (<mime>, <sizeLabel>)"
    const lastOpen = stripped.lastIndexOf('(');
    const lastClose = stripped.lastIndexOf(')');
    if (lastOpen < 0 || lastClose < 0 || lastClose < lastOpen) continue;
    const path = stripped.slice(0, lastOpen).trim();
    const inside = stripped.slice(lastOpen + 1, lastClose).trim();
    const commaIdx = inside.indexOf(',');
    if (!path || commaIdx < 0) continue;
    const mime = inside.slice(0, commaIdx).trim();
    const sizeLabel = inside.slice(commaIdx + 1).trim();
    if (!mime || !sizeLabel) continue;
    attachments.push({ path, mime, sizeLabel });
  }

  return {
    strippedContent: content.slice(match[0].length),
    attachments,
  };
}

/**
 * Build the same <attached_files> block the backend emits, so the optimistic
 * locally-inserted user_message has the identical wire content as the WS echo
 * that will eventually arrive. This avoids any flicker on broadcast.
 */
export function buildAttachmentsBlock(
  attachments: Array<{ path: string; mime: string; size: number }>,
): string {
  if (attachments.length === 0) return '';
  const lines = ['<attached_files>'];
  for (const a of attachments) {
    lines.push(`- ${a.path} (${a.mime}, ${humanSize(a.size)})`);
  }
  lines.push('</attached_files>');
  return lines.join('\n') + '\n\n';
}

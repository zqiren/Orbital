// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { translate } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

type Tr = (key: StringKey, vars?: Record<string, string | number>) => string;
const EN: Tr = (k, v) => translate('en', k, v);

// Orbital's own tool names with a translated counter. Any other name (a
// worker's own tool, an MCP tool) is shown as is, its count beside it.
const NAMED: Record<string, [StringKey, StringKey]> = {
  read: ['chat.capsule.tool.read.one', 'chat.capsule.tool.read.other'],
  write: ['chat.capsule.tool.write.one', 'chat.capsule.tool.write.other'],
  edit: ['chat.capsule.tool.edit.one', 'chat.capsule.tool.edit.other'],
  shell: ['chat.capsule.tool.shell.one', 'chat.capsule.tool.shell.other'],
  glob: ['chat.capsule.tool.glob.one', 'chat.capsule.tool.glob.other'],
  grep: ['chat.capsule.tool.grep.one', 'chat.capsule.tool.grep.other'],
};

/** A capsule's tool counts as one line ("6 reads, 4 writes, glob"), most
 *  used first. ``tr`` defaults to English, byte-identical to the old line. */
export function formatToolBreakdown(counts: Record<string, number>, tr: Tr = EN): string {
  const entries = Object.entries(counts);
  if (entries.length === 0) return '';
  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return entries
    .map(([name, n]) => {
      const keys = Object.prototype.hasOwnProperty.call(NAMED, name) ? NAMED[name] : undefined;
      if (keys) return tr(n === 1 ? keys[0] : keys[1], { n });
      return n === 1 ? name : tr('chat.capsule.tool.other', { n, name });
    })
    .join(tr('chat.capsule.tool.sep'));
}

/** A capsule's duration ("8s", "1m 28s"); zh "8秒", "1分28秒". */
export function formatCapsuleDuration(startedAt: number, endedAt: number | null, tr: Tr = EN): string {
  if (!endedAt || endedAt <= startedAt) return tr('duration.lessThan1s');
  const ms = endedAt - startedAt;
  if (ms < 1000) return tr('duration.lessThan1s');
  const s = Math.round(ms / 1000);
  if (s < 60) return tr('duration.seconds', { s });
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return rs ? tr('duration.minSec', { m, s: rs }) : tr('duration.minutes', { m });
}

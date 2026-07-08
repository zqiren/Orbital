// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { FileContent } from '../types';

export type OpenPathOutcome =
  | { status: 'ok'; content: FileContent; path: string }
  | { status: 'not_found' }
  | { status: 'ambiguous'; matches: string[] };

/**
 * Fetch a workspace path for the preview drawer, recovering from the two
 * abbreviation failure modes agents exhibit in chat replies (sub-directory
 * relative paths and bare filenames): on a miss, ask the resolve endpoint for
 * suffix matches and transparently open a UNIQUE match. Multiple matches are
 * surfaced as `ambiguous` — never guess which file the agent meant.
 */
export async function fetchPathWithFallback(
  getContent: (path: string) => Promise<FileContent | null>,
  resolve: (path: string) => Promise<string[] | null>,
  path: string,
): Promise<OpenPathOutcome> {
  const exact = await getContent(path);
  if (exact) return { status: 'ok', content: exact, path };

  const matches = await resolve(path);
  if (!matches || matches.length === 0) return { status: 'not_found' };
  if (matches.length > 1) return { status: 'ambiguous', matches };

  const fallback = await getContent(matches[0]);
  if (!fallback) return { status: 'not_found' }; // deleted between resolve and fetch
  return { status: 'ok', content: fallback, path: matches[0] };
}

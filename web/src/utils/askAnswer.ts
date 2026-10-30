// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/** The wire form of a message that answers an ask (spec 146). The model reads
 *  the prefix, so it knows what was answered and that it needn't close it; the
 *  chat shows the question above the user's words instead. */
const ANSWER_RE = /\[Answer to ask [0-9a-f]{6}, already closed: "([\s\S]*?)"\]\n/;

export function formatAskAnswer(askId: string, question: string, words: string): string {
  return `[Answer to ask ${askId}, already closed: "${question}"]\n${words}`;
}

export function parseAskAnswer(content: string): { question: string | null; rest: string } {
  const match = ANSWER_RE.exec(content);
  if (!match) return { question: null, rest: content };
  return { question: match[1], rest: content.replace(ANSWER_RE, '') };
}

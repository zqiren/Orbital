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

/** Spec 160 round 4: the one-turn daemon note the inject route folds into an
 *  answer to the import scan's confirm card. The model reads it on that turn;
 *  the chat never shows it. Round 5: the same for the approval a 对，做第 1 步
 *  tap carries (`<orbital_picked_step>`): the bubble shows the tap. */
const TURN_NOTE_RE = /\s*<(orbital_turn_note|orbital_picked_step)>[\s\S]*?<\/\1>/g;

/** What a user row says, for a one-line quote (the left-off cards): the
 *  answer's wire prefix and any daemon turn note stripped, so an ask answered
 *  with a pill reads as the word tapped, not as
 *  `[Answer to ask d0425a, already closed: "…"] red` (batch 1110, dsh pass). */
export function displayAnswerText(content: string): string {
  const rest = parseAskAnswer(stripTurnNote(content)).rest.trim();
  return rest || content;
}

export function stripTurnNote(content: string): string {
  return content.includes('<orbital_turn_note>') || content.includes('<orbital_picked_step>')
    ? content.replace(TURN_NOTE_RE, '')
    : content;
}

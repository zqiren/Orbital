// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 156 §5.2 / §5.8 — editing a sent message.
 *
 * A user row's `content` is the raw wire string: the machine blocks the
 * composer (and the daemon) put around the words the user typed. The bubble
 * strips them with the parsers below, in this order; the editor edits only
 * what is left, and the rebuild puts every block back byte-for-byte. The
 * result is sent as `content` alone (no `attachments` / `skill` fields), so the
 * daemon neither re-validates nor re-inlines them: the blocks already are the
 * text the agent saw.
 *
 *   [turn note]        <orbital_turn_note>…  daemon-folded, one turn only: dropped
 *   trigger header     [Triggered by …]      kept (edit is never offered there)
 *   attachments block  <attached_files>…     kept
 *   skill block        <use_skill …>…        kept
 *   ask-answer prefix  [Answer to ask …]     kept
 *   body                                     replaced
 *   quotes block       annotation quotes     kept, after the body
 */

import { parseAskAnswer, stripTurnNote } from './askAnswer';
import { splitQuotesBlock } from './annotations';
import { parseAttachmentsBlock } from '../lib/attachment-parsing';
import { parseSkillBlock } from '../lib/skill-parsing';
import { parseTriggerHeader } from '../lib/trigger-parsing';
import { translate, type TVars } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

export interface UserWireParts {
  /** Trigger header + attachments block + skill block + ask-answer prefix. */
  prefix: string;
  /** What the bubble shows as text, and what the editor holds. */
  body: string;
  /** The newlines between the body and the quotes block ('' without one). */
  quotesSep: string;
  /** The annotation quotes block, verbatim; null when there is none. */
  quotes: string | null;
}

/** Split a user row's wire content with the same parsers the bubble uses. */
export function splitUserWire(content: string): UserWireParts {
  const s0 = stripTurnNote(content);
  const s1 = parseTriggerHeader(s0).strippedContent;
  const s2 = parseAttachmentsBlock(s1).strippedContent;
  const s3 = parseSkillBlock(s2).strippedContent;
  const { question, rest } = parseAskAnswer(s3);
  // The ask prefix always leads the text (formatAskAnswer); only then is it
  // a prefix that can be cut off and put back.
  const afterPrefix = question !== null && s3.endsWith(rest) ? rest : s3;
  const prefix = s0.slice(0, s0.length - afterPrefix.length);
  const { text, quotes } = splitQuotesBlock(afterPrefix);
  if (quotes === null) return { prefix, body: text, quotesSep: '', quotes: null };
  const start = afterPrefix.length - quotes.length;
  return { prefix, body: text, quotesSep: afterPrefix.slice(text.length, start), quotes };
}

/**
 * The original wire string with only its body replaced. Pure; unchanged text
 * gives back the original (less a daemon turn note).
 */
export function rebuildEditedContent(original: string, newBody: string): string {
  const parts = splitUserWire(original);
  let out = parts.prefix + newBody;
  if (parts.quotes !== null) {
    // As the composer joins them: `${text}\n\n${quotes}`, or the quotes alone.
    out += newBody ? (parts.quotesSep || '\n\n') + parts.quotes : parts.quotes;
  }
  return out;
}

type Translator = (key: StringKey, vars?: TVars) => string;
const EN: Translator = (k, v) => translate('en', k, v);

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** A trailing `· 改自 14:32` / `· edited 14:32`, in either locale. */
const FORK_SUFFIX_RE = (() => {
  const alts = (['zh', 'en'] as const).map((loc) => {
    const tpl = translate(loc, 'chat.edit.forkName', { name: '\u0000', time: '\u0001' });
    const tail = tpl.slice(tpl.indexOf('\u0000') + 1);
    return escapeRe(tail).replace('\u0001', '\\d{1,2}:\\d{2}');
  });
  return new RegExp(`(?:${alts.join('|')})$`);
})();

/** 24-hour HH:MM in local time, as the bubble shows it. */
function clock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/**
 * §5.8: `{原名} · 改自 HH:MM`, HH:MM being the edited message's time. A fork of
 * a fork keeps one suffix: any trailing fork suffix, in either language, is
 * cut first.
 */
export function forkSessionName(parentName: string, editedAt: string, t: Translator = EN): string {
  let name = parentName.trim();
  while (FORK_SUFFIX_RE.test(name)) name = name.replace(FORK_SUFFIX_RE, '').trimEnd();
  return t('chat.edit.forkName', { name, time: clock(editedAt) });
}

// ── the edited message waiting for its fork to open ─────────────────────────
// §5.7: after the fork route answers, the app opens the new session and the
// edited message is sent there once its history has loaded. The opening can
// remount the chat (the Workbench pane keys it by session), so the message
// waits in this module, keyed by project + session, not in component state.
// In memory only: a reload in that instant loses the send, never the text of
// the parent (the parent is untouched, and the fork shows its kept history).

export interface ForkSend {
  /** The rebuilt wire content (`rebuildEditedContent`). */
  content: string;
  /** The fork's pin (the edited row's target); null = Orbital. */
  target: string | null;
  /** The edited body, put back in the composer when the send fails. */
  body: string;
}

const pendingForkSends = new Map<string, ForkSend>();
const forkKey = (projectId: string, sessionId: string) => `${projectId}\u0000${sessionId}`;

export function stashForkSend(projectId: string, sessionId: string, send: ForkSend): void {
  pendingForkSends.set(forkKey(projectId, sessionId), send);
}

/** The waiting send for this session, removed as it is read. */
export function takeForkSend(projectId: string, sessionId: string): ForkSend | null {
  const key = forkKey(projectId, sessionId);
  const send = pendingForkSends.get(key) ?? null;
  pendingForkSends.delete(key);
  return send;
}

export function __resetForkSendsForTests(): void {
  pendingForkSends.clear();
}

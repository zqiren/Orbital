// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo } from 'react';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';
import { classifySessionName, type SessionKind } from '../lib/sessionLabel';
import type { SessionListEntry } from '../types';

const KIND_KEY: Partial<Record<SessionKind, StringKey>> = {
  queue: 'sessionItem.kind.queue',
  schedule: 'sessionItem.kind.schedule',
  file_watch: 'sessionItem.kind.fileWatch',
  memory: 'sessionItem.kind.memory',
  attachment: 'sessionItem.kind.attachment',
};

interface SessionHeadingProps {
  /** The open session's list entry; undefined for a session that does not
   *  exist yet (the composer is open, nothing sent) or is not in the list. */
  session: SessionListEntry | undefined;
}

/**
 * The top of the chat beside the project column (spec 137 follow-up). The
 * column starts with the project name; without this the conversation ran
 * straight under the top edge with nothing on that row, so the right area
 * had no anchor. Same position as PageHeading, so the chat and a page
 * (Tasks, Settings) share one top line and switching does not jump; a step
 * smaller than the project name, since a session sits inside the project.
 *
 * The title is the session's name as the list shows it (the same cleaning
 * as SessionListItem). Above it, small, the things a row carries as a chip:
 * an automation's kind, or "Imported · codex". A session with no name yet
 * reads as "New session". No bar, no line: the messages scroll under a short
 * fade below the heading (`.session-heading` in index.css — a plain sRGB
 * gradient; Tailwind's oklab gradient to `transparent` paints a dark band in
 * WebKit, which is what the packaged app renders in), so the boundary is the
 * heading itself. While workers run, SubAgentStatusBar adds a line of agent
 * tiles under the title and carries the fade instead (`.sub-agent-line`).
 *
 * The phone keeps its project header and never mounts this.
 */
export default function SessionHeading({ session }: SessionHeadingProps) {
  const t = useT();
  const name = session?.name;
  const origin = session?.origin;
  const info = useMemo(() => (session ? classifySessionName(name, origin) : null), [session, name, origin]);

  let title: string;
  let eyebrow: string | null = null;
  if (!session || !info) {
    title = t('sessionHeading.new');
  } else {
    const kindKey = KIND_KEY[info.kind];
    const kindText = kindKey ? t(kindKey) : null;
    const isImported = session.origin === 'imported' || session.imported !== undefined;
    const importAgent = isImported ? session.imported?.agent || session.pinned_target || null : null;
    const importedText = isImported
      ? importAgent
        ? `${t('session.imported')} · ${importAgent}`
        : t('session.imported')
      : null;
    if (info.displayName) {
      title = info.displayName;
      eyebrow = [importedText, kindText].filter(Boolean).join(' · ') || null;
    } else {
      // A machine session with no name of its own: the kind is the title.
      title = kindText ?? t('sessionHeading.new');
      eyebrow = importedText;
    }
  }

  return (
    <div
      className="session-heading relative z-10 shrink-0 px-7 pt-[20px] pb-1 bg-card"
      data-testid="session-heading"
    >
      {eyebrow && (
        <p className="text-xs text-secondary truncate" data-testid="session-heading-eyebrow">
          {eyebrow}
        </p>
      )}
      <h1
        className="text-[15px] font-semibold tracking-[-0.01em] text-primary truncate"
        data-testid="session-heading-title"
      >
        {title}
      </h1>
    </div>
  );
}

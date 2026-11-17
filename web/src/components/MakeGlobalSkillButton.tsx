// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Project settings → Skills: the icon on a skill row that copies the skill
 * into Orbital's global skills (POST /api/v2/skills/from-project), so every
 * project can use it. The project copy stays where it is.
 *
 * Icon only, with the app's icon tooltip (the dark bubble the edge strip and
 * the turn undo use) on hover and keyboard focus. After a copy, or when the
 * global list already has a skill of that name (`alreadyGlobal`, so it
 * survives a reload), it is a quiet check with its own tooltip. A 409 (a global skill with that name already exists)
 * reports the short "already exists" line through `onMessage` and turns the
 * icon into a small Replace button, which repeats the copy with
 * `overwrite: true`. Other failures report through `onMessage` and leave the
 * icon as it was.
 */

import { useState } from 'react';
import { Check, Globe } from 'lucide-react';
import { api, ApiError } from '../config';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface MakeGlobalSkillButtonProps {
  projectId: string;
  dirName: string;
  name: string;
  /** The section's message line: '' clears it. */
  onMessage: (message: string) => void;
  /** The global list already holds a skill of this name. */
  alreadyGlobal?: boolean;
  testId?: string;
}

/** The app's icon tooltip: a small dark bubble under the icon, shown while
 *  the enclosing `group` is hovered or keyboard-focused. */
function IconTip({ text }: { text: string }) {
  return (
    <span
      role="tooltip"
      className="pointer-events-none absolute right-0 top-[calc(100%+6px)] z-30 whitespace-nowrap rounded-[7px] bg-[#18181B] px-[9px] py-[5px] text-xs font-medium text-white opacity-0 shadow-[0_6px_18px_rgb(0_0_0/0.18)] transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-[#F4F6F9] dark:text-[#18181B]"
    >
      {text}
    </span>
  );
}

function errorKey(e: unknown): StringKey {
  if (e instanceof ApiError && e.status === 413) return 'global.skills.error.tooLarge';
  if (e instanceof ApiError && e.status === 400) return 'global.skills.error.invalid';
  return 'settings.skills.makeGlobal.error';
}

export default function MakeGlobalSkillButton({
  projectId,
  dirName,
  name,
  onMessage,
  alreadyGlobal = false,
  testId,
}: MakeGlobalSkillButtonProps) {
  const t = useT();
  const [state, setState] = useState<'idle' | 'busy' | 'done' | 'conflict'>('idle');
  // Derived, not synced by an effect: the global list arrives after mount.
  const shown = alreadyGlobal && state === 'idle' ? 'done' : state;

  async function copy(overwrite: boolean) {
    onMessage('');
    setState('busy');
    try {
      await api('/api/v2/skills/from-project', {
        method: 'POST',
        body: JSON.stringify({ project_id: projectId, dir_name: dirName, overwrite }),
      });
      setState('done');
    } catch (e) {
      if (!overwrite && e instanceof ApiError && e.status === 409) {
        setState('conflict');
        onMessage(t('global.skills.error.exists'));
      } else {
        setState('idle');
        onMessage(t(errorKey(e)));
      }
    }
  }

  if (shown === 'done') {
    return (
      <span
        role="img"
        tabIndex={0}
        data-testid={testId ? `${testId}-done` : undefined}
        aria-label={t('settings.skills.isGlobal')}
        className="group relative flex shrink-0 items-center justify-center rounded-md p-1 text-secondary focus:outline-none focus-visible:ring-1 focus-visible:ring-accent max-md:min-w-[44px] max-md:min-h-[44px]"
      >
        <Check size={16} strokeWidth={2} aria-hidden="true" />
        <IconTip text={t('settings.skills.isGlobal')} />
      </span>
    );
  }

  if (shown === 'conflict') {
    return (
      <button
        type="button"
        data-testid={testId ? `${testId}-replace` : undefined}
        onClick={() => void copy(true)}
        title={t('settings.skills.makeGlobal.replaceTitle', { name })}
        className="shrink-0 text-xs font-medium text-secondary border border-border rounded-lg px-2.5 py-1 hover:text-primary hover:bg-card-hover transition-all duration-150 max-md:min-h-[36px]"
      >
        {t('settings.skills.makeGlobal.replace')}
      </button>
    );
  }

  return (
    <button
      type="button"
      data-testid={testId}
      disabled={shown === 'busy'}
      onClick={() => void copy(false)}
      aria-label={t('settings.skills.makeGlobal')}
      className="group relative shrink-0 rounded-md text-secondary hover:text-primary transition-colors p-1 focus:outline-none focus-visible:ring-1 focus-visible:ring-accent disabled:opacity-50 max-md:min-w-[44px] max-md:min-h-[44px] flex items-center justify-center"
    >
      <Globe size={16} strokeWidth={2} aria-hidden="true" />
      <IconTip text={t('settings.skills.makeGlobal')} />
    </button>
  );
}

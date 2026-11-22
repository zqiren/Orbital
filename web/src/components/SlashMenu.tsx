// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The composer's `/` picker (spec 132/137/166), shared by the chat composer
 * and the Workbench composer so both list the same entries for the same
 * recipient and behave alike.
 *
 * `useSlashPicker` owns the picker state: the catalog for (project, worker)
 * from the per-project module cache (skill-catalog.ts), the built-ins
 * (`/automate`, not offered when the project has no manager), the sticky Esc
 * and the highlighted row. Open/closed is derived from the composer text
 * (`slashQuery`), never stored. `SlashMenu` renders the list.
 *
 * What a pick does is the caller's: a worker command (`invoke`) prefills its
 * invocation, a built-in chips + prefills, a catalog skill chips.
 */

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import {
  builtinSkills,
  cachedSkillCatalog,
  loadSkillCatalog,
  matchSkills,
  slashQuery,
  withBuiltinSkills,
  type SkillCatalogEntry,
} from '../lib/skill-catalog';

export interface SlashPickerOptions {
  /** The project whose catalog is listed; null = no picker. */
  projectId: string | null;
  /** Who receives the message: a worker slug, or null = Orbital. */
  worker: string | null;
  /** The project has no manager to run a built-in (keyless). */
  managerMissing: boolean;
  /** The composer text. */
  text: string;
}

export interface SlashPicker {
  /** The client-side entries (`/automate`), for seeding a chip. */
  builtins: SkillCatalogEntry[];
  /** The rows the menu shows now; empty = closed. */
  filtered: SkillCatalogEntry[];
  open: boolean;
  selectedIndex: number;
  /** Call on every composer change (loads the catalog on `/`). */
  onTextChange: (value: string) => void;
  /** Arrow/Enter/Tab/Esc while open; true when the key was the menu's. */
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>, pick: (entry: SkillCatalogEntry) => void) => boolean;
  /** After a pick: back to the first row. */
  resetIndex: () => void;
  menuRef: React.RefObject<HTMLDivElement | null>;
}

function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch {
    return 'UTC';
  }
}

export function useSlashPicker({ projectId, worker, managerMissing, text }: SlashPickerOptions): SlashPicker {
  const { locale } = useLocale();
  // The catalog lives in the module cache; this counter re-renders when a
  // fetch lands.
  const [, setVersion] = useState(0);
  const installed = projectId ? cachedSkillCatalog(projectId, worker) : null;
  // Bound to `locale` (their description and prefill are translated), not to
  // the unstable `t`. Keyless: `/automate` needs the manager's trigger tools.
  const builtins = useMemo(
    () =>
      managerMissing ? [] : builtinSkills(
        {
          description: translate(locale, 'chat.skill.automate.description'),
          prefill: translate(locale, 'automations.example.schedule'),
        },
        browserTimeZone(),
      ),
    [locale, managerMissing],
  );
  const catalog = withBuiltinSkills(installed, builtins);

  // Spec 166: warm a worker's list as soon as the recipient is known, so the
  // first `/` doesn't wait on its cold probe and a recipient switch
  // re-renders an open menu with the new list.
  useEffect(() => {
    if (!projectId || !worker) return;
    let live = true;
    void loadSkillCatalog(projectId, worker).then(() => {
      if (live) setVersion((v) => v + 1);
    });
    return () => {
      live = false;
    };
  }, [projectId, worker]);

  // The sticky Esc: holds until the composer stops starting with `/`.
  const [dismissed, setDismissed] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const menuRef = useRef<HTMLDivElement>(null);

  // Keep the highlighted row visible inside the menu. Only the menu's own
  // scrollTop moves: scrollIntoView would also scroll the page (WebKit).
  useEffect(() => {
    const menu = menuRef.current;
    const row = menu?.children[selectedIndex] as HTMLElement | undefined;
    if (!menu || !row) return;
    if (row.offsetTop < menu.scrollTop) menu.scrollTop = row.offsetTop;
    else if (row.offsetTop + row.offsetHeight > menu.scrollTop + menu.clientHeight) {
      menu.scrollTop = row.offsetTop + row.offsetHeight - menu.clientHeight;
    }
  }, [selectedIndex]);

  function onTextChange(value: string) {
    if (!value.startsWith('/')) {
      if (dismissed) setDismissed(false);
      return;
    }
    setSelectedIndex(0);
    if (projectId && slashQuery(value) !== null && !dismissed) {
      // Answers from the cache while it is fresh; re-render only on new data.
      const before = catalog;
      void loadSkillCatalog(projectId, worker).then((list) => {
        if (list !== before) setVersion((v) => v + 1);
      });
    }
  }

  // Open only while the composer is in picker position, the catalog has
  // loaded, and Esc has not dismissed it. No matches means fully closed, so
  // Enter sends and Tab behaves normally (`/Users/me/a.txt hi`).
  const query = dismissed ? null : slashQuery(text);
  const filtered = query !== null && catalog ? matchSkills(catalog, query) : [];
  const open = filtered.length > 0;

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>, pick: (entry: SkillCatalogEntry) => void): boolean {
    if (!open) return false;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1));
      return true;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
      return true;
    }
    if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault();
      pick(filtered[Math.min(selectedIndex, filtered.length - 1)]);
      return true;
    }
    if (e.key === 'Escape') {
      setDismissed(true);
      return true;
    }
    return false;
  }

  return {
    builtins,
    filtered,
    open,
    selectedIndex,
    onTextChange,
    onKeyDown,
    resetIndex: () => setSelectedIndex(0),
    menuRef,
  };
}

/** Batch 1110 item 12: Claude Code's own `/schedule` creates a cloud routine
 *  on the user's Claude account, which Orbital can neither see nor stop; the
 *  picker says so beside it (hiding it would only hide the danger). */
function isCloudRoutineCommand(entry: SkillCatalogEntry): boolean {
  return entry.source === 'worker' && entry.key === 'schedule';
}

export interface SlashMenuProps {
  picker: SlashPicker;
  onPick: (entry: SkillCatalogEntry) => void;
  /** Where the list sits relative to its composer (positioning only). */
  className: string;
}

/** The `/` list. Skills and commands in one flat list (no origin badge); at
 *  most 8 rows visible, then it scrolls. */
export default function SlashMenu({ picker, onPick, className }: SlashMenuProps) {
  const t = useT();
  if (!picker.open) return null;
  return (
    <div
      role="listbox"
      aria-label={t('chat.skill.menuAria')}
      data-testid="skill-menu"
      ref={picker.menuRef}
      className={`${className} w-[min(28rem,calc(100vw-2rem))] max-h-80 max-md:max-h-[352px] overflow-y-auto surface-float-pop z-50 py-1`}
    >
      {picker.filtered.map((skill, i) => (
        <button
          key={skill.path || `worker:${skill.key}`}
          type="button"
          role="option"
          aria-selected={i === picker.selectedIndex}
          data-testid="skill-option"
          data-skill-name={skill.name}
          title={skill.description || skill.name}
          className={`w-full flex items-center px-3 h-10 max-md:h-11 text-left text-body min-w-0 hover:bg-card-hover/50 ${
            i === picker.selectedIndex ? 'bg-card-hover' : ''
          }`}
          onMouseDown={(e) => {
            e.preventDefault();
            onPick(skill);
          }}
        >
          {/* Row centred vertically; name and description share a baseline. */}
          <span className="flex items-baseline gap-2 min-w-0 w-full">
            <span className="shrink-0 max-w-[45%] truncate font-medium text-primary">{skill.name}</span>
            {skill.hint && (
              <span data-testid="skill-option-hint" className="shrink-0 max-w-[25%] truncate text-caption text-secondary">{skill.hint}</span>
            )}
            {skill.description && (
              <span className="min-w-0 flex-1 truncate text-caption text-secondary">{skill.description}</span>
            )}
            {isCloudRoutineCommand(skill) && (
              // Batch 1110 item 12: a worker command that creates a routine
              // on the user's own account, outside Orbital.
              <span data-testid="skill-option-warning" className="shrink-0 truncate text-caption text-warning">
                {t('skills.worker.cloudRoutine')}
              </span>
            )}
          </span>
        </button>
      ))}
    </div>
  );
}

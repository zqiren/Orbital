// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The skill row and the add-a-skill drop target, shared by the project
 * Skills section (SettingsView) and the global one (GlobalSkillsSettings).
 *
 * A row offers delete only when the caller passes `onDelete`; the trash glyph
 * then asks once, in place, before anything is removed (the same two-button
 * confirm as the project Danger Zone). An on/off switch shows when the caller
 * passes `onToggle` (global settings); the project section passes neither
 * the switch nor a mark and looks exactly as before.
 */

import { forwardRef, useState, type ReactNode } from 'react';
import { useT } from '../i18n/useT';

interface SkillRowProps {
  name: string;
  description: string;
  /** Hover text for the row (a read-only row's folder path). */
  title?: string;
  /** A small mark before the name (where the skill was found). */
  leading?: ReactNode;
  /** Present = the row can be deleted. May reject; the row then resets. */
  onDelete?: () => Promise<void> | void;
  /** On/off state for the switch; the switch shows only with `onToggle`. */
  enabled?: boolean;
  onToggle?: (on: boolean) => void;
  /** Accessible name of the switch, already translated. */
  toggleLabel?: string;
  /** Greys the name (e.g. a same-name skill wins over this one). */
  muted?: boolean;
  /** Small mark after the name (a hover sign). */
  badge?: ReactNode;
  /** Extra icon actions, placed just before delete. */
  actions?: ReactNode;
  testId?: string;
}

/** A compact on/off switch, the settings toggle at row size. */
export function RowSwitch({
  on,
  onChange,
  label,
  testId,
}: {
  on: boolean;
  onChange: (on: boolean) => void;
  label: string;
  testId?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      data-testid={testId}
      onClick={() => onChange(!on)}
      className={`relative inline-flex h-[18px] w-[30px] shrink-0 items-center rounded-full transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent/50 max-md:my-[13px] ${
        on ? 'bg-accent' : 'bg-border'
      }`}
    >
      <span
        aria-hidden="true"
        className={`inline-block h-[14px] w-[14px] rounded-full bg-white shadow-[0_1px_2px_rgb(0_0_0/0.22)] transition-transform duration-150 ${
          on ? 'translate-x-[14px]' : 'translate-x-[2px]'
        }`}
      />
    </button>
  );
}

export function SkillRow({
  name,
  description,
  title,
  leading,
  onDelete,
  enabled = true,
  onToggle,
  toggleLabel,
  muted = false,
  badge,
  actions,
  testId,
}: SkillRowProps) {
  const t = useT();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function confirmDelete() {
    if (!onDelete) return;
    setBusy(true);
    try {
      await onDelete();
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  return (
    <div
      data-testid={testId}
      title={title}
      className="flex items-center justify-between gap-2 surface-float-static surface-float-nested px-3 py-2 max-md:min-h-[44px]"
    >
      {/* The dimming sits on the content, not the row, so a hover sign's
          bubble inside `badge` stays at full strength. */}
      <div className="flex min-w-0 items-center gap-2.5">
        {leading && <span className={`flex shrink-0 ${muted ? 'opacity-50' : ''}`}>{leading}</span>}
        <div className="min-w-0">
          <span className="flex items-center gap-1.5">
            <span className={`text-sm font-medium text-primary truncate ${muted ? 'opacity-50' : ''}`}>{name}</span>
            {badge}
          </span>
          {description && (
            <span className={`text-xs text-secondary block truncate ${muted ? 'opacity-50' : ''}`}>{description}</span>
          )}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
      {!confirming && actions}
      {onDelete &&
        (confirming ? (
          <div className="flex shrink-0 items-center gap-1.5">
            <button
              type="button"
              data-testid={testId ? `${testId}-confirm` : undefined}
              disabled={busy}
              onClick={() => void confirmDelete()}
              className="text-xs font-medium text-white bg-error rounded-lg px-2.5 py-1 hover:bg-error/90 transition-all duration-150 disabled:opacity-60 max-md:min-h-[36px]"
            >
              {t('settings.skills.deleteConfirm')}
            </button>
            <button
              type="button"
              data-testid={testId ? `${testId}-cancel` : undefined}
              onClick={() => setConfirming(false)}
              className="text-xs font-medium text-secondary border border-border rounded-lg px-2.5 py-1 hover:bg-card-hover transition-all duration-150 max-md:min-h-[36px]"
            >
              {t('settings.skills.deleteCancel')}
            </button>
          </div>
        ) : (
          <button
            type="button"
            data-testid={testId ? `${testId}-delete` : undefined}
            onClick={() => setConfirming(true)}
            className="shrink-0 text-secondary hover:text-error transition-colors p-1 max-md:min-w-[44px] max-md:min-h-[44px] flex items-center justify-center"
            title={t('settings.skills.deleteTitle', { name })}
            aria-label={t('settings.skills.deleteTitle', { name })}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        ))}
      {onToggle && (
        <RowSwitch
          on={enabled}
          onChange={onToggle}
          label={toggleLabel ?? name}
          testId={testId ? `${testId}-switch` : undefined}
        />
      )}
      </div>
    </div>
  );
}

interface SkillUploadProps {
  uploading: boolean;
  onFile: (file: File) => void;
  testId?: string;
}

/** The dashed "Add Skill (.md or .zip)" target; the ref is the file input. */
export const SkillUpload = forwardRef<HTMLInputElement, SkillUploadProps>(function SkillUpload(
  { uploading, onFile, testId },
  ref,
) {
  const t = useT();
  return (
    <label className="flex items-center justify-center border border-dashed border-border rounded-lg px-3 py-3 cursor-pointer hover:border-accent/50 hover:bg-accent/5 transition-all duration-150 max-md:min-h-[44px]">
      <span className="text-sm text-secondary">
        {uploading ? t('settings.skills.uploading') : t('settings.skills.uploadCta')}
      </span>
      <input
        ref={ref}
        type="file"
        accept=".md,.zip"
        className="hidden"
        data-testid={testId}
        disabled={uploading}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
    </label>
  );
});

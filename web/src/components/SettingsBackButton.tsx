// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { ArrowLeft } from 'lucide-react';

interface SettingsBackButtonProps {
  /** Already translated. */
  label: string;
  onClick: () => void;
  testId: string;
}

/**
 * The way out of a settings page, shared by project and global settings.
 *
 * It used to be a 14px grey text link above the title — the same weight as
 * the subtitle under it, so on a page full of grey helper text it was the one
 * control users could not find. It is a real button now: bordered, primary
 * text, a hover fill, and a full-size tap target on mobile.
 */
export default function SettingsBackButton({ label, onClick, testId }: SettingsBackButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={testId}
      className="inline-flex w-fit max-w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-sm font-medium text-primary shadow-sm transition-colors duration-150 hover:bg-card-hover hover:border-secondary/40 max-md:min-h-[44px]"
    >
      <ArrowLeft size={16} className="shrink-0" aria-hidden="true" />
      <span className="truncate">{label}</span>
    </button>
  );
}

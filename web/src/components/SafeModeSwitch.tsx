// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * SafeModeSwitch — the one 安全模式 / Safe mode row (spec 117 D2), shared by
 * the create-project dialog and project settings so the two never drift.
 *
 * Off means the agent's shell runs as the user with the user's network and
 * the read tools see the whole disk; on means today's sandbox model plus
 * "ask me" when the sandbox blocks a command. The explanation lives in one
 * sentence of hover copy: it appears while the pointer is over the row and
 * can be pinned open (for touch, where there is no hover) with the ⓘ.
 */

import { useState } from 'react';
import { Info } from 'lucide-react';
import { useT } from '../i18n/useT';

interface SafeModeSwitchProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
}

export default function SafeModeSwitch({ checked, onChange, disabled = false }: SafeModeSwitchProps) {
  const t = useT();
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const showHint = hovered || pinned;

  return (
    <div
      data-testid="safe-mode-row"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-body font-medium text-primary">{t('safeMode.label')}</span>
          <button
            // type="button" is load-bearing in the create dialog's <form>.
            type="button"
            onClick={() => setPinned((v) => !v)}
            aria-expanded={showHint}
            aria-label={t('safeMode.about')}
            title={t('safeMode.about')}
            className={`shrink-0 transition-colors duration-150 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center ${
              pinned ? 'text-primary' : 'text-secondary hover:text-primary'
            }`}
          >
            <Info size={13} aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={checked}
          aria-label={t('safeMode.label')}
          disabled={disabled}
          onClick={() => onChange(!checked)}
          data-testid="safe-mode-toggle"
          className={`shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-150 disabled:opacity-50 ${
            checked ? 'bg-accent' : 'bg-border'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-150 ${
              checked ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
      {showHint && (
        <p className="mt-1.5 text-caption text-secondary" data-testid="safe-mode-hint">
          {t('safeMode.tooltip')}
        </p>
      )}
    </div>
  );
}

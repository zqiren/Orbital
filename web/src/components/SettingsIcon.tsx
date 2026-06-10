// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Settings } from 'lucide-react';
import { useT } from '../i18n/useT';

interface SettingsIconProps {
  onClick: () => void;
  title?: string;
}

export default function SettingsIcon({ onClick, title }: SettingsIconProps) {
  const t = useT();
  return (
    <button
      onClick={onClick}
      aria-label={t('settingsIcon.aria')}
      title={title ?? t('settingsIcon.aria')}
      className="flex items-center justify-center p-1 rounded text-secondary hover:text-primary transition-colors duration-150"
    >
      <Settings size={14} />
    </button>
  );
}

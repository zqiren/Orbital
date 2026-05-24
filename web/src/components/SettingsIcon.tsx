// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Settings } from 'lucide-react';

interface SettingsIconProps {
  onClick: () => void;
  title?: string;
}

export default function SettingsIcon({ onClick, title = 'Project settings' }: SettingsIconProps) {
  return (
    <button
      onClick={onClick}
      aria-label="Project settings"
      title={title}
      className="flex items-center justify-center p-1 rounded text-secondary hover:text-primary transition-colors duration-150"
    >
      <Settings size={14} />
    </button>
  );
}

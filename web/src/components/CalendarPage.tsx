// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';

interface CalendarPageProps {
  /**
   * When set, the calendar is mounted as a project lens (the per-project
   * `calendar` tab) filtered to events linked to this project. When absent,
   * this is the global calendar surface. Wave 3 replaces this placeholder body
   * with the real week/agenda grid (spec 011 §0.4); the projectId contract is
   * kept stable so that swap is body-only.
   */
  projectId?: string;
}

export default function CalendarPage({ projectId }: CalendarPageProps) {
  const t = useT();
  return (
    <div
      className="flex flex-col items-center justify-center flex-1 min-h-0 gap-2 px-6 text-center"
      data-testid="calendar-page"
      data-project-lens={projectId ? 'true' : undefined}
    >
      <h1 className="font-mono text-[18px] font-semibold text-primary">
        {t('workspace.calendar.title')}
      </h1>
      <p className="text-sm text-secondary max-w-sm">
        {t('workspace.calendar.comingSoon')}
      </p>
    </div>
  );
}

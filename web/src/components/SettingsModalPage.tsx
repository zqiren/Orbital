// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Dispatch, SetStateAction } from 'react';
import { ArrowLeft } from 'lucide-react';
import type { Project, ProjectUpdateRequest } from '../types';
import type { Route } from '../route';
import SettingsView from './SettingsView';
import { useT } from '../i18n/useT';

interface SettingsModalPageProps {
  project: Project;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  onSave: (data: ProjectUpdateRequest) => void;
  onDelete: () => void;
  /** Open the pricing-table editor overlay (P3-I). */
  onEditPricing?: () => void;
}

export default function SettingsModalPage({
  project,
  route,
  setRoute,
  onSave,
  onDelete,
  onEditPricing,
}: SettingsModalPageProps) {
  const t = useT();
  function handleBack() {
    setRoute({
      name: 'project',
      projectId: route.projectId,
      tab: route.tab,
      sessionId: route.sessionId,
      settings: false,
    });
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-background">
      {/* Header band.
          The inner track mirrors SettingsView's layout exactly — a rail-width
          spacer plus the same 720px column, left-anchored as one unit. Without
          it the band starts at the page's left edge while the form it titles
          starts 274px further right, so the page header did not read as
          belonging to the content under it. The spacer hides with the rail
          below `lg`.

          "Mirrors exactly" is load-bearing: this track and SettingsView's must
          carry the SAME justification and the SAME left padding, or the title
          and the fields it titles stop sharing a left edge. Both moved from
          `justify-center` to `justify-start pl-6` together. */}
      <div className="pt-5 pb-4 border-b border-border">
        <div className="flex justify-start pl-6 max-md:pl-0 max-md:block">
          <div className="w-44 shrink-0 max-lg:hidden" aria-hidden="true" />
          {/* Same box AND same inner padding as the settings column, so the
              title lands on the same left edge as the fields under it. */}
          <div className="flex flex-col gap-1 max-w-[720px] w-full min-w-0 px-6 max-md:px-4">
            <button
              onClick={handleBack}
              data-testid="settings-back-button"
              className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-colors w-fit"
            >
              <ArrowLeft size={14} />
              {t('settingsModal.back', { project: project.name })}
            </button>
            <h1 className="text-lg font-semibold text-primary mt-1" data-testid="settings-modal-title">
              {t('settingsModal.title', { project: project.name })}
            </h1>
            <p className="text-sm text-secondary">
              {t('settingsModal.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <SettingsView
          project={project}
          onSave={onSave}
          onDelete={onDelete}
          onEditPricing={onEditPricing}
          scrollToSection={route.settingsAnchor}
        />
      </div>
    </div>
  );
}

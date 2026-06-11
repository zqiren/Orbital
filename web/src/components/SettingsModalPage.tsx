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
  /**
   * Bumped after a pricing-table save so the Budget meter/breakdown recompute
   * against the new rates. Threaded down to BudgetSection's useCost.
   */
  costRefreshKey?: number;
}

export default function SettingsModalPage({
  project,
  route,
  setRoute,
  onSave,
  onDelete,
  onEditPricing,
  costRefreshKey,
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
      {/* Header band */}
      <div className="flex flex-col gap-1 px-6 pt-5 pb-4 border-b border-border max-md:px-4">
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

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <SettingsView
          project={project}
          onSave={onSave}
          onDelete={onDelete}
          onEditPricing={onEditPricing}
          costRefreshKey={costRefreshKey}
          scrollToSection={route.settingsAnchor}
        />
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Dispatch, SetStateAction } from 'react';
import { ArrowLeft } from 'lucide-react';
import type { Project, ProjectUpdateRequest } from '../types';
import type { Route } from '../route';
import SettingsView from './SettingsView';

interface SettingsModalPageProps {
  project: Project;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  onSave: (data: ProjectUpdateRequest) => void;
  onDelete: () => void;
}

export default function SettingsModalPage({
  project,
  route,
  setRoute,
  onSave,
  onDelete,
}: SettingsModalPageProps) {
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
    <div className="flex flex-col flex-1 min-h-0 bg-[#FAFAFA]">
      {/* Header band */}
      <div className="flex flex-col gap-1 px-6 pt-5 pb-4 border-b border-[#E4E4E7] max-md:px-4">
        <button
          onClick={handleBack}
          data-testid="settings-back-button"
          className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-colors w-fit"
        >
          <ArrowLeft size={14} />
          Back to {project.name}
        </button>
        <p className="text-xs font-mono uppercase tracking-widest text-secondary mt-1">
          Project · this project only
        </p>
        <h1 className="text-lg font-semibold text-primary" data-testid="settings-modal-title">
          Project settings — {project.name}
        </h1>
        <p className="text-sm text-secondary">
          Configure agent behaviour, LLM provider, autonomy, and more for this project.
        </p>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <SettingsView project={project} onSave={onSave} onDelete={onDelete} />
      </div>
    </div>
  );
}

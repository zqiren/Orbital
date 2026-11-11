// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Project, SetupGateSettings } from '../types';

type ProjectLike = Pick<Project, 'project_id' | 'is_scratch'>;

/**
 * True while the user has no project of their own — Quick Tasks (the scratch
 * project the daemon auto-creates at boot) is the only one.
 *
 * An EMPTY list is deliberately not first-run: `listProjects()` resolves `[]`
 * on a failed fetch and the list is `[]` before the first fetch lands, while a
 * successfully loaded list always contains Quick Tasks. That makes "has the
 * scratch project and nothing else" the one signal that is both loaded and
 * new — no separate "loaded" flag to keep in sync. (It is also why the old
 * `projects.length === 0` "create your first one" branch was unreachable.)
 */
export function isFirstRun(projects: readonly ProjectLike[]): boolean {
  return projects.length > 0 && projects.every((p) => !!p.is_scratch);
}

/**
 * Where the setup wizard hands off. The wizard's import step can create
 * projects; if it did, land in the most recently added one (the list keeps
 * creation order for never-dragged projects, which is all a new user has).
 * `null` means the user still has no project → open Create Project.
 */
export function wizardLandingProjectId(projects: readonly ProjectLike[]): string | null {
  const own = projects.filter((p) => !p.is_scratch);
  return own.length > 0 ? own[own.length - 1].project_id : null;
}

/**
 * The project the wizard's Finish opens, always on a new session: the import
 * step's landing project when it is listed, else (the landing was lost, e.g. a
 * reload mid-wizard) `wizardLandingProjectId`. `null` → open Create Project.
 */
export function wizardHandoffProjectId(
  projects: readonly ProjectLike[],
  imported: string | null,
): string | null {
  if (imported && projects.some((p) => p.project_id === imported)) return imported;
  return wizardLandingProjectId(projects);
}

/**
 * The first-run gate App.tsx applies to GET /api/v2/settings: the wizard shows
 * until a key exists or the user skipped the key step. `null` (the daemon did
 * not answer) shows it, as before. An older daemon sends no `setup`, which
 * reads as "not skipped". Saving a key later never resets the flag — once a
 * key exists the flag is irrelevant.
 */
export function needsSetupWizard(settings: SetupGateSettings | null): boolean {
  if (!settings) return true;
  return !settings.llm.api_key_set && settings.setup?.api_key_skipped !== true;
}

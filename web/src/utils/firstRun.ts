// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Project } from '../types';

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

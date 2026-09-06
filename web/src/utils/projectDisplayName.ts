// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import type { Project } from '../types';
import type { StringKey } from '../i18n/strings';

/**
 * The name to show for a project. The scratch (Quick Tasks) project is
 * auto-created by the backend with an English name that is *stored*, so it
 * never went through the catalog and showed as "Quick Tasks" in every locale.
 * Render it through the catalog instead; every other project keeps its name.
 */
export function projectDisplayName(
  project: Pick<Project, 'name' | 'is_scratch'>,
  t: (key: StringKey) => string,
): string {
  return project.is_scratch ? t('project.scratchName') : project.name;
}

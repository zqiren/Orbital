// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import type { Project, ProjectCreateRequest, ProjectUpdateRequest } from '../types';

/**
 * Reorder `projects` to match `orderedIds` — the client-side mirror of the
 * `GET /projects` sort, so the optimistic view a drop paints is byte-identical
 * to the canonical list the server sends back.
 *
 * Same two components, same precedence: scratch first (an id listed in
 * `orderedIds` can never overtake the pinned Quick Tasks row), then manual
 * position, then original index — which keeps any project missing from
 * `orderedIds` at the bottom in creation order rather than jumping it to the
 * front. Pure: returns a new array, mutates nothing.
 */
export function applyManualOrder(
  projects: Project[],
  orderedIds: string[],
): Project[] {
  const rank = new Map(orderedIds.map((id, index) => [id, index] as const));
  return projects
    .map((project, index) => ({ project, index }))
    .sort((a, b) => {
      const scratch = Number(!a.project.is_scratch) - Number(!b.project.is_scratch);
      if (scratch !== 0) return scratch;
      const rankA = rank.get(a.project.project_id) ?? Infinity;
      const rankB = rank.get(b.project.project_id) ?? Infinity;
      if (rankA !== rankB) return rankA - rankB;
      return a.index - b.index;
    })
    .map((entry) => entry.project);
}

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Latest committed list, mirrored into a ref. `reorderProjects` needs a
  // rollback snapshot; taking it from inside a setState updater would be the
  // exact React 19 hazard CLAUDE.md warns about (the updater can be deferred
  // to the render phase, so the closure variable is still stale when read).
  // The ref is written in an effect — always current by the time a pointer
  // drag can finish — and read synchronously.
  const projectsRef = useRef<Project[]>(projects);
  useEffect(() => {
    projectsRef.current = projects;
  }, [projects]);

  const listProjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<Project[]>('/api/v2/projects');
      setProjects(data);
      return data;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to list projects';
      setError(msg);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const createProject = useCallback(async (data: ProjectCreateRequest) => {
    setError(null);
    const project = await api<Project>('/api/v2/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    setProjects((prev) => [...prev, project]);
    return project;
  }, []);

  const getProject = useCallback(async (id: string) => {
    return api<Project>(`/api/v2/projects/${encodeURIComponent(id)}`);
  }, []);

  /**
   * Re-fetch a single project and merge it into the list. Used to refresh
   * runtime fields after a turn completes, without
   * re-listing every project. No-op on fetch error (keeps the prior value).
   */
  const refreshProject = useCallback(async (id: string) => {
    try {
      const updated = await api<Project>(`/api/v2/projects/${encodeURIComponent(id)}`);
      setProjects((prev) => prev.map((p) => (p.project_id === id ? updated : p)));
      return updated;
    } catch {
      return null;
    }
  }, []);

  const updateProject = useCallback(async (id: string, data: ProjectUpdateRequest) => {
    const updated = await api<Project>(
      `/api/v2/projects/${encodeURIComponent(id)}`,
      { method: 'PUT', body: JSON.stringify(data) },
    );
    setProjects((prev) =>
      prev.map((p) => (p.project_id === id ? updated : p)),
    );
    return updated;
  }, []);

  /**
   * Persist a manual project order (spec 056). `orderedIds` must be the
   * complete id list for the block being reordered.
   *
   * Applies the new order synchronously so the drop lands without a
   * round-trip, then POSTs. The response is the canonical list, so a
   * `listProjects()` refetch racing this call converges. If the POST fails the
   * pre-drag order is restored and the error is re-thrown for the caller to
   * surface — the list never sits in a state the server disagrees with.
   *
   * The rollback re-imposes the pre-drag ORDER on whatever the list holds at
   * that moment, rather than restoring the pre-drag array wholesale. `App`
   * refetches projects from several event paths, so a `listProjects()` (or a
   * single-record `refreshProject`) can land while the POST is in flight;
   * restoring the stale snapshot would silently throw that fresher data away,
   * and would resurrect a project deleted in the meantime. Undoing only the
   * thing that failed — the ordering — leaves every other update intact.
   */
  const reorderProjects = useCallback(async (orderedIds: string[]) => {
    const previousIds = projectsRef.current.map((p) => p.project_id);
    setProjects(applyManualOrder(projectsRef.current, orderedIds));
    try {
      const canonical = await api<Project[]>('/api/v2/projects/reorder', {
        method: 'POST',
        body: JSON.stringify({ ordered_ids: orderedIds }),
      });
      setProjects(canonical);
      return canonical;
    } catch (e) {
      setProjects((current) => applyManualOrder(current, previousIds));
      throw e;
    }
  }, []);

  const deleteProject = useCallback(async (id: string) => {
    await api(`/api/v2/projects/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    setProjects((prev) => prev.filter((p) => p.project_id !== id));
  }, []);

  return {
    projects,
    loading,
    error,
    listProjects,
    createProject,
    getProject,
    refreshProject,
    updateProject,
    reorderProjects,
    deleteProject,
  };
}

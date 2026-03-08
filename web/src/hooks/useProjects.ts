// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useState } from 'react';
import { api } from '../config';
import type { Project, ProjectCreateRequest, ProjectUpdateRequest } from '../types';

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const deleteProject = useCallback(async (id: string, clearOutput: boolean = false) => {
    const qs = clearOutput ? '?clear_output=true' : '';
    await api(`/api/v2/projects/${encodeURIComponent(id)}${qs}`, {
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
    updateProject,
    deleteProject,
  };
}

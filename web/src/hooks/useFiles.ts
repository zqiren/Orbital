// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useState } from 'react';
import { api } from '../config';
import type { DirectoryListing, FileContent } from '../types';

export function useFiles() {
  const [directory, setDirectory] = useState<DirectoryListing | null>(null);
  const [fileContent, setFileContent] = useState<FileContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const listDirectory = useCallback(
    async (projectId: string, path?: string) => {
      setLoading(true);
      setError(null);
      try {
        const query = path ? `?path=${encodeURIComponent(path)}` : '';
        const data = await api<DirectoryListing>(
          `/api/v2/projects/${encodeURIComponent(projectId)}/files${query}`,
        );
        setDirectory(data);
        return data;
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Failed to list directory';
        setError(msg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const getFileContent = useCallback(
    async (projectId: string, path: string) => {
      setLoading(true);
      setError(null);
      try {
        const data = await api<FileContent>(
          `/api/v2/projects/${encodeURIComponent(projectId)}/files/content?path=${encodeURIComponent(path)}`,
        );
        setFileContent(data);
        return data;
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Failed to read file';
        setError(msg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { directory, fileContent, loading, error, listDirectory, getFileContent };
}

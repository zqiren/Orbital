// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useState } from 'react';
import { api } from '../config';
import type { PlatformStatus, FolderInfo } from '../types';

export function usePlatform() {
  const [status, setStatus] = useState<PlatformStatus | null>(null);
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<PlatformStatus>('/api/v2/platform/status');
      setStatus(data);
      return data;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to get platform status';
      setError(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const triggerSetup = useCallback(async () => {
    setError(null);
    try {
      const result = await api<{ status: string; success: boolean; error: string | null }>(
        '/api/v2/platform/setup',
        { method: 'POST' },
      );
      await getStatus();
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Setup failed';
      setError(msg);
      throw e;
    }
  }, [getStatus]);

  const getFolders = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<{ status: string; folders: FolderInfo[] }>(
        '/api/v2/platform/folders',
      );
      setFolders(data.folders);
      return data.folders;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to get folders';
      setError(msg);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  // TODO: Wire to folder management UI when built
  const grantFolderAccess = useCallback(async (path: string, mode: 'read_only' | 'read_write') => {
    setError(null);
    try {
      const result = await api<{ status: string }>('/api/v2/platform/folders/grant', {
        method: 'POST',
        body: JSON.stringify({ path, mode }),
      });
      await getFolders();
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to grant access';
      setError(msg);
      throw e;
    }
  }, [getFolders]);

  // TODO: Wire to folder management UI when built
  const revokeFolderAccess = useCallback(async (path: string) => {
    setError(null);
    try {
      const result = await api<{ status: string }>('/api/v2/platform/folders/revoke', {
        method: 'POST',
        body: JSON.stringify({ path }),
      });
      await getFolders();
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to revoke access';
      setError(msg);
      throw e;
    }
  }, [getFolders]);

  return {
    status, folders, loading, error,
    getStatus, triggerSetup, getFolders,
    grantFolderAccess, revokeFolderAccess,
  };
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiBytes } from '../config';
import type { DirectoryListing, FileContent, RecentFile } from '../types';

export type FileBytesState =
  | { status: 'idle'; bytes: null; error: null }
  | { status: 'loading'; bytes: null; error: null }
  | { status: 'ready'; bytes: ArrayBuffer; error: null }
  | { status: 'error'; bytes: null; error: Error };

const BYTES_IDLE: FileBytesState = { status: 'idle', bytes: null, error: null };
const BYTES_LOADING: FileBytesState = { status: 'loading', bytes: null, error: null };

/**
 * Spec 090 — raw bytes of `url` (a document's `preview_url`) for a client-side
 * renderer. A new URL or unmount aborts the in-flight request, and a result is
 * only ever reported for the URL it was fetched for, so a slow fetch for the
 * previous file can never land on the next one. `null` stays idle.
 */
export function useFileBytes(url: string | null): FileBytesState {
  const [settled, setSettled] = useState<{ url: string; state: FileBytesState } | null>(null);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    apiBytes(url, { signal: controller.signal }).then(
      (bytes) => {
        if (!controller.signal.aborted) {
          setSettled({ url, state: { status: 'ready', bytes, error: null } });
        }
      },
      (err: unknown) => {
        if (!controller.signal.aborted) {
          const error = err instanceof Error ? err : new Error(String(err));
          setSettled({ url, state: { status: 'error', bytes: null, error } });
        }
      },
    );
    return () => controller.abort();
  }, [url]);

  if (!url) return BYTES_IDLE;
  return settled?.url === url ? settled.state : BYTES_LOADING;
}

/**
 * The file-content route for a client that renders documents (spec 090). The
 * `document_preview=1` opt-in gets PDF / Office / CSV back as a `document`
 * envelope; without it the daemon keeps the pre-090 shapes for older UIs.
 *
 * Spec 103: `ifRevision` is the revision the caller already holds. A daemon
 * that knows the param answers `{ path, revision, unchanged: true }` when it
 * still matches, without reading the file; an older daemon ignores it and
 * sends the full body, which the caller's revision compare already absorbs.
 */
export function fileContentPath(projectId: string, path: string, ifRevision?: string): string {
  const base = `/api/v2/projects/${encodeURIComponent(projectId)}/files/content?path=${encodeURIComponent(path)}&document_preview=1`;
  return ifRevision ? `${base}&if_revision=${encodeURIComponent(ifRevision)}` : base;
}

/**
 * Spec 093 — show a workspace path in Finder / File Explorer on the machine
 * running the daemon ('' = open the project folder itself). Resolves `false`
 * when the daemon refuses (missing path, relayed request, unsupported OS)
 * rather than throwing, so a click handler can just report it.
 */
export async function revealPath(projectId: string, path: string): Promise<boolean> {
  try {
    await api(`/api/v2/projects/${encodeURIComponent(projectId)}/files/reveal`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Spec 187 — the project's 50 most recently modified files, newest first.
 * `null` on any failure (404 from an older daemon, 503 when ripgrep is
 * missing, a network error): the caller has one failure path.
 */
export async function listRecent(projectId: string): Promise<RecentFile[] | null> {
  try {
    const data = await api<{ files: RecentFile[] }>(
      `/api/v2/projects/${encodeURIComponent(projectId)}/files/recent`,
    );
    return Array.isArray(data?.files) ? data.files : null;
  } catch {
    return null;
  }
}

/** One request at a time per project; ticks that land mid-flight collapse
 *  into a single follow-up. `alive` dies with the effect that made it. */
interface RecentGeneration {
  projectId: string;
  alive: boolean;
  inFlight: boolean;
  pending: boolean;
}

/**
 * Spec 187 — the panel's Recent list. Fetches on mount (and project switch)
 * and whenever `tick` moves while `active` (tree state). No polling, no
 * cache, no retry. A failed request keeps the list already held; a first
 * failure leaves it empty, which the caller renders as nothing.
 *
 * A scan can outlast the 150 ms tick coalesce on a big workspace, so at most
 * one request is in flight and a tick during it sets `pending`, answered by
 * exactly one follow-up. A project switch or unmount kills the generation:
 * its late response is dropped and its pending follow-up never fires.
 */
export function useRecentFiles(projectId: string, tick: number, active: boolean): RecentFile[] {
  const [files, setFiles] = useState<RecentFile[]>([]);
  const genRef = useRef<RecentGeneration | null>(null);
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
  });

  const request = useCallback((gen: RecentGeneration) => {
    if (gen.inFlight) {
      gen.pending = true;
      return;
    }
    gen.inFlight = true;
    const run = () => {
      void listRecent(gen.projectId).then((result) => {
        gen.inFlight = false;
        if (!gen.alive) return;
        if (result !== null) setFiles(result);
        if (gen.pending) {
          gen.pending = false;
          if (activeRef.current) {
            gen.inFlight = true;
            run();
          }
        }
      });
    };
    run();
  }, []);

  useEffect(() => {
    const gen: RecentGeneration = { projectId, alive: true, inFlight: false, pending: false };
    genRef.current = gen;
    setFiles([]);
    if (activeRef.current) request(gen);
    return () => {
      gen.alive = false;
      if (genRef.current === gen) genRef.current = null;
    };
  }, [projectId, request]);

  useEffect(() => {
    // Tick 0 is the mount, which the effect above already fetched for.
    if (tick === 0 || !active || genRef.current === null) return;
    request(genRef.current);
    // `active` is read, not watched: leaving the preview bumps the tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, request]);

  return files;
}

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
    async (projectId: string, path: string, ifRevision?: string) => {
      setLoading(true);
      setError(null);
      try {
        const data = await api<FileContent>(fileContentPath(projectId, path, ifRevision));
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

  const resolvePath = useCallback(
    async (projectId: string, path: string): Promise<string[] | null> => {
      try {
        const data = await api<{ matches: string[] }>(
          `/api/v2/projects/${encodeURIComponent(projectId)}/files/resolve?path=${encodeURIComponent(path)}`,
        );
        return data.matches;
      } catch {
        return null;
      }
    },
    [],
  );

  const saveFileContent = useCallback(
    async (projectId: string, path: string, content: string): Promise<boolean> => {
      try {
        await api(
          `/api/v2/projects/${encodeURIComponent(projectId)}/files/content`,
          {
            method: 'PUT',
            body: JSON.stringify({ path, content }),
          },
        );
        return true;
      } catch {
        return false;
      }
    },
    [],
  );

  return {
    directory,
    fileContent,
    loading,
    error,
    listDirectory,
    getFileContent,
    resolvePath,
    saveFileContent,
    revealPath,
  };
}

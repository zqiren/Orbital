// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Attach by local path (desktop app only).
 *
 * In the desktop app the files are already on this machine, so the composer
 * attaches a file by its absolute path instead of uploading a copy (no size
 * cap). The native bridge (`agent_os/desktop/main.py`) offers `pick_files()`
 * for the attach button and, for drag and drop, hands the dropped files' real
 * paths to the page as an `orbital:native-drop` event. Every other client
 * (browser, phone through the relay) has neither and keeps uploading.
 */

export interface LocalFileEntry {
  /** Absolute realpath on this machine. */
  path: string;
  /** The file's own name. */
  name: string;
  size: number;
  mime: string;
}

export const NATIVE_DROP_EVENT = 'orbital:native-drop';

/** True in the desktop app whose bridge can pick files by path. */
export function canPickLocalFiles(): boolean {
  if (typeof window === 'undefined') return false;
  return typeof window.pywebview?.api?.pick_files === 'function';
}

/** The native multi-select dialog; `[]` on cancel or failure. */
export async function pickLocalFiles(): Promise<LocalFileEntry[]> {
  try {
    const picked = await window.pywebview!.api.pick_files!();
    return Array.isArray(picked) ? picked.filter(isEntry) : [];
  } catch {
    return [];
  }
}

/** True when the desktop shell reports dropped files' real paths. */
export function nativeDropAvailable(): boolean {
  return typeof window !== 'undefined' && window.__orbitalNativeDrop === true
    && canPickLocalFiles();
}

function isEntry(e: unknown): e is LocalFileEntry {
  if (!e || typeof e !== 'object') return false;
  const o = e as Record<string, unknown>;
  return typeof o.path === 'string' && o.path.length > 0
    && typeof o.name === 'string' && typeof o.size === 'number'
    && typeof o.mime === 'string';
}

/**
 * The real paths of the files in the drop being handled now. The shell sends
 * them right after the page's own drop event; if nothing arrives in time the
 * result is empty and the caller uploads the dropped files as before.
 */
export function waitForNativeDrop(timeoutMs = 1500): Promise<LocalFileEntry[]> {
  return new Promise((resolve) => {
    let done = false;
    const finish = (entries: LocalFileEntry[]) => {
      if (done) return;
      done = true;
      window.removeEventListener(NATIVE_DROP_EVENT, onEvent);
      clearTimeout(timer);
      resolve(entries);
    };
    const onEvent = (ev: Event) => {
      const detail = (ev as CustomEvent).detail;
      finish(Array.isArray(detail) ? detail.filter(isEntry) : []);
    };
    window.addEventListener(NATIVE_DROP_EVENT, onEvent);
    const timer = setTimeout(() => finish([]), timeoutMs);
  });
}

/**
 * Split a drop into files with a real path and files to upload. Matched by
 * name, each path used once (two dropped files may share a name only when
 * they come from different folders, and then each gets its own entry).
 */
export function splitDrop(
  files: readonly File[],
  entries: readonly LocalFileEntry[],
): { local: LocalFileEntry[]; upload: File[] } {
  const pool = [...entries];
  const local: LocalFileEntry[] = [];
  const upload: File[] = [];
  for (const f of files) {
    const i = pool.findIndex((e) => e.name === f.name);
    if (i >= 0) {
      local.push(pool[i]);
      pool.splice(i, 1);
    } else {
      upload.push(f);
    }
  }
  return { local, upload };
}

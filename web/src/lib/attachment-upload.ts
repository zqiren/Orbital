// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

const MAX_FILENAME_LEN = 200;

export function timestampedName(originalName: string): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const ts = [
    now.getFullYear(),
    '-',
    pad(now.getMonth() + 1),
    '-',
    pad(now.getDate()),
    'T',
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join('');

  // Strip any directory components — backslashes, slashes, dots that would
  // climb. We never want a name like "../evil.txt" to slip into FormData.
  const justBasename = originalName
    .replace(/\\/g, '/')
    .split('/')
    .pop() ?? originalName;

  // Sanitize: keep alphanumerics, ".", "-", "_"; replace everything else.
  // Disallow leading dots so ".." doesn't survive intact. Tidy up
  // dangling separators around the file extension so we don't ship names
  // like "My-Photo-2-.jpg" — split, scrub each part, rejoin.
  const lastDot = justBasename.lastIndexOf('.');
  const rawBase = lastDot > 0 ? justBasename.slice(0, lastDot) : justBasename;
  const rawExt = lastDot > 0 ? justBasename.slice(lastDot + 1) : '';

  const scrub = (s: string) =>
    s
      .replace(/\.{2,}/g, '-')
      .replace(/[^A-Za-z0-9._-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^[-.]+/, '')
      .replace(/[-.]+$/, '');

  const baseClean = scrub(rawBase);
  const extClean = scrub(rawExt);
  let sanitized = extClean ? `${baseClean}.${extClean}` : baseClean;
  if (!sanitized) sanitized = 'file';

  let combined = `${ts}-${sanitized}`;
  if (combined.length > MAX_FILENAME_LEN) {
    // Keep the extension if there is one.
    const dotIdx = sanitized.lastIndexOf('.');
    const ext = dotIdx >= 0 ? sanitized.slice(dotIdx) : '';
    const prefix = `${ts}-`;
    const room = MAX_FILENAME_LEN - prefix.length - ext.length;
    const base = sanitized
      .slice(0, dotIdx >= 0 ? dotIdx : sanitized.length)
      .slice(0, Math.max(1, room));
    combined = `${prefix}${base}${ext}`;
  }

  return combined;
}

export function humanSize(bytesCount: number): string {
  if (bytesCount < 1024) return `${bytesCount} B`;
  if (bytesCount < 1024 * 1024) return `${(bytesCount / 1024).toFixed(1)} KB`;
  if (bytesCount < 1024 * 1024 * 1024) {
    return `${(bytesCount / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytesCount / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export interface UploadParams {
  projectId: string;
  file: File;
  baseUrl: string;
  isRelayMode: boolean;
  onProgress?: (pct: number) => void;
}

export interface UploadResult {
  path: string;
  size: number;
}

/**
 * Upload a single file to <baseUrl>/api/v2/projects/<id>/files/upload?path=/uploads/.
 *
 * Uses XMLHttpRequest so we can fire progress events — fetch's upload-progress
 * support is not portable. The file is renamed to a timestamp-prefixed
 * filename before being added to the multipart form, so the server-side
 * file.filename is the sanitised, timestamped form.
 */
export function uploadFile(params: UploadParams): Promise<UploadResult> {
  const { projectId, file, baseUrl, isRelayMode, onProgress } = params;
  return new Promise<UploadResult>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `${baseUrl}/api/v2/projects/${encodeURIComponent(projectId)}/files/upload?path=/uploads/`;
    xhr.open('POST', url);

    if (isRelayMode) {
      try {
        const token = localStorage.getItem('relay_jwt');
        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }
      } catch {
        // ignore
      }
    }

    if (onProgress) {
      xhr.upload.onprogress = (e: ProgressEvent) => {
        if (e.lengthComputable && e.total > 0) {
          onProgress((e.loaded / e.total) * 100);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const body = JSON.parse(xhr.responseText) as UploadResult;
          resolve(body);
        } catch {
          reject(new Error(`Upload succeeded but response was not JSON (status ${xhr.status})`));
        }
        return;
      }
      let detail = xhr.responseText;
      try {
        const parsed = JSON.parse(xhr.responseText);
        detail = parsed.detail ?? xhr.responseText;
      } catch {
        // keep raw text
      }
      reject(new Error(`Upload failed (${xhr.status}): ${detail || 'unknown error'}`));
    };

    xhr.onerror = () => reject(new Error('Upload failed: network error'));
    xhr.onabort = () => reject(new Error('Upload aborted'));

    const renamed = timestampedName(file.name);
    const fd = new FormData();
    fd.append('file', file, renamed);
    xhr.send(fd);
  });
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useRef, useState } from 'react';
import type { ChangeEvent, ClipboardEvent, DragEvent } from 'react';
import { BASE_URL, isRelayMode } from '../config';
import { uploadFile } from '../lib/attachment-upload';
import {
  canPickLocalFiles, nativeDropAvailable, pickLocalFiles, splitDrop, waitForNativeDrop,
} from '../lib/localFiles';
import type { LocalFileEntry } from '../lib/localFiles';
import { useT } from '../i18n/useT';

export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
export const MAX_ATTACHMENTS = 10;

export interface PendingAttachment {
  id: string;
  /** The bytes to upload; absent on a local-path chip. */
  file?: File;
  /** Desktop app: the file's absolute path. Nothing is uploaded, no size
   *  cap; the chip is ready (`done`) the moment it appears. */
  localPath?: string;
  filename: string;
  mime: string;
  size: number;
  status: 'pending' | 'uploading' | 'done' | 'error';
  uploadedPath?: string;
  /** deferUpload: the project `uploadedPath` lives in (a later send may
   *  route elsewhere, and then the file goes up again). */
  uploadedProject?: string;
  thumbnailUrl?: string;
  errorMessage?: string;
}

function genId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export interface UseAttachmentsOptions {
  /**
   * Where cap-exceeded copy goes. Chat routes it into its inject-error banner;
   * the queue composer keeps its own line. Messages arrive already localized.
   */
  onError?: (message: string) => void;
  /**
   * Stage only: added files stay `pending` (no upload) until `uploadAll`
   * sends them to a project. The Workbench composer knows its project only
   * after routing, so it uploads at send time.
   */
  deferUpload?: boolean;
  /**
   * Desktop app: the attach button and drag and drop attach files by their
   * absolute local path instead of uploading (chat and Workbench). Ignored
   * in any other client, and paste always uploads.
   */
  localPaths?: boolean;
}

export interface UploadedAttachment {
  path: string;
  mime: string;
  size: number;
  /** A local-path attachment (desktop app): `path` is absolute. */
  local?: boolean;
}

/** Over the upload cap? A local-path chip never is: nothing is copied. */
export function overUploadCap(a: Pick<PendingAttachment, 'size' | 'localPath'>): boolean {
  return !a.localPath && a.size > MAX_ATTACHMENT_BYTES;
}

/**
 * Composer attachment state: staged chips, their uploads, retry, and the
 * object-URL lifetime of image thumbnails.
 *
 * Shared by the chat composer and the queue composer so the caps, the error
 * copy and the thumbnail cleanup are defined once. The hook deliberately knows
 * nothing about how the resulting paths are sent — chat inlines an
 * `<attached_files>` block into the message, the queue sends bare `file_refs`
 * and lets the dispatcher build that block at dispatch time. The Workbench
 * composer stages with `deferUpload` and calls `uploadAll` once routing has
 * picked the project.
 */
export function useAttachments(
  projectId: string,
  options?: UseAttachmentsOptions,
) {
  const t = useT();
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);

  // `t` and `onError` get a fresh identity on every render. Holding them in
  // refs keeps the callbacks below stable, so consuming components can pass
  // them straight to event handlers and effect deps.
  const tRef = useRef(t);
  tRef.current = t;
  const onErrorRef = useRef(options?.onError);
  onErrorRef.current = options?.onError;
  const defer = options?.deferUpload === true;
  // The committed list, for `uploadAll` (called from an event handler, after
  // the chips it sends have rendered).
  const attachmentsRef = useRef(attachments);
  attachmentsRef.current = attachments;

  /** Upload one chip into `pid`, marking it uploading → done / error.
   *  Throws on failure (after marking the chip). */
  const uploadInto = useCallback(
    async (att: PendingAttachment, pid: string): Promise<UploadedAttachment> => {
      setAttachments((prev) =>
        prev.map((a) =>
          a.id === att.id ? { ...a, status: 'uploading' as const, errorMessage: undefined } : a,
        ),
      );
      try {
        if (!att.file) throw new Error(tRef.current('chat.uploadError'));
        const baseUrl = isRelayMode ? window.location.origin : BASE_URL;
        const { path, size } = await uploadFile({
          projectId: pid,
          file: att.file,
          baseUrl,
          isRelayMode,
        });
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === att.id
              ? { ...a, status: 'done' as const, uploadedPath: path, uploadedProject: pid, size }
              : a,
          ),
        );
        return { path, mime: att.mime, size };
      } catch (err) {
        const message =
          err instanceof Error ? err.message : tRef.current('chat.uploadError');
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === att.id
              ? { ...a, status: 'error' as const, errorMessage: message }
              : a,
          ),
        );
        throw err instanceof Error ? err : new Error(message);
      }
    },
    [],
  );

  const uploadAttachment = useCallback(
    async (att: PendingAttachment) => {
      try {
        await uploadInto(att, projectId);
      } catch {
        // The chip shows the error and offers retry.
      }
    },
    [projectId, uploadInto],
  );

  /**
   * deferUpload: upload the staged chips (all of them, or the ids given)
   * into `pid`, one after another, and return their payload in chip order.
   * A chip already uploaded into `pid` is not sent again; one over the size
   * cap is skipped. Throws when any upload fails; the chips stay, the failed
   * one in its error state.
   */
  const uploadAll = useCallback(
    async (pid: string, ids?: readonly string[]): Promise<UploadedAttachment[]> => {
      const want = ids ? new Set(ids) : null;
      const list = attachmentsRef.current.filter(
        (a) => (!want || want.has(a.id)) && !overUploadCap(a),
      );
      const out: UploadedAttachment[] = [];
      for (const a of list) {
        if (a.localPath) {
          // Local path: passes straight through, whatever the project.
          out.push({ path: a.localPath, mime: a.mime, size: a.size, local: true });
          continue;
        }
        if (a.status === 'done' && a.uploadedPath && a.uploadedProject === pid) {
          out.push({ path: a.uploadedPath, mime: a.mime, size: a.size });
          continue;
        }
        out.push(await uploadInto(a, pid));
      }
      return out;
    },
    [uploadInto],
  );

  const addAttachments = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      setAttachments((prev) => {
        const room = MAX_ATTACHMENTS - prev.length;
        if (room <= 0) {
          onErrorRef.current?.(
            tRef.current('chat.maxAttachments', { n: MAX_ATTACHMENTS }),
          );
          return prev;
        }
        const usable = files.slice(0, room);
        if (files.length > usable.length) {
          onErrorRef.current?.(
            tRef.current('chat.onlyNAllowed', { n: MAX_ATTACHMENTS }),
          );
        }
        const additions: PendingAttachment[] = usable.map((file) => {
          const id = genId();
          const isImage = file.type.startsWith('image/');
          const thumbnailUrl = isImage ? URL.createObjectURL(file) : undefined;
          const tooBig = file.size > MAX_ATTACHMENT_BYTES;
          const att: PendingAttachment = {
            id,
            file,
            filename: file.name,
            mime: file.type || 'application/octet-stream',
            size: file.size,
            status: tooBig ? 'error' : 'pending',
            thumbnailUrl,
            errorMessage: tooBig
              ? tRef.current('chat.fileTooLarge')
              : undefined,
          };
          return att;
        });

        // Kick off uploads for the valid ones on the next tick so React has
        // committed the new chip state. Deferred: they wait for uploadAll.
        if (!defer) setTimeout(() => {
          for (const a of additions) {
            if (a.status === 'pending') {
              uploadAttachment(a);
            }
          }
        }, 0);

        return [...prev, ...additions];
      });
    },
    [uploadAttachment, defer],
  );

  /** Desktop app: chips for files attached by their local path. Ready at
   *  once, no upload, no size cap. */
  const addLocalFiles = useCallback((entries: readonly LocalFileEntry[]) => {
    if (entries.length === 0) return;
    setAttachments((prev) => {
      const room = MAX_ATTACHMENTS - prev.length;
      if (room <= 0) {
        onErrorRef.current?.(tRef.current('chat.maxAttachments', { n: MAX_ATTACHMENTS }));
        return prev;
      }
      const usable = entries.slice(0, room);
      if (entries.length > usable.length) {
        onErrorRef.current?.(tRef.current('chat.onlyNAllowed', { n: MAX_ATTACHMENTS }));
      }
      return [
        ...prev,
        ...usable.map((e): PendingAttachment => ({
          id: genId(),
          localPath: e.path,
          filename: e.name,
          mime: e.mime || 'application/octet-stream',
          size: e.size,
          status: 'done',
        })),
      ];
    });
  }, []);

  const localMode = options?.localPaths === true;

  /**
   * The attach button. In the desktop app (with `localPaths`) the native
   * dialog picks files by path; anywhere else `browse()` opens the page's
   * file input, as before.
   */
  const openPicker = useCallback(
    async (browse: () => void) => {
      if (!localMode || !canPickLocalFiles()) {
        browse();
        return;
      }
      addLocalFiles(await pickLocalFiles());
    },
    [localMode, addLocalFiles],
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.thumbnailUrl) {
        try {
          URL.revokeObjectURL(target.thumbnailUrl);
        } catch {
          // ignore
        }
      }
      return prev.filter((a) => a.id !== id);
    });
  }, []);

  const retryAttachment = useCallback(
    (id: string) => {
      setAttachments((prev) => {
        const target = prev.find((a) => a.id === id);
        if (!target) return prev;
        const reset: PendingAttachment = {
          ...target,
          status: 'pending',
          errorMessage: undefined,
        };
        // Deferred: back to staged; the next send uploads it.
        if (!defer) setTimeout(() => uploadAttachment(reset), 0);
        return prev.map((a) => (a.id === id ? reset : a));
      });
    },
    [uploadAttachment, defer],
  );

  const clearAttachments = useCallback(() => {
    setAttachments((prev) => {
      for (const a of prev) {
        if (a.thumbnailUrl) {
          try {
            URL.revokeObjectURL(a.thumbnailUrl);
          } catch {
            // ignore
          }
        }
      }
      return [];
    });
  }, []);

  const handleFilePickerChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (files.length > 0) addAttachments(files);
      e.target.value = '';
    },
    [addAttachments],
  );

  const handlePaste = useCallback(
    (e: ClipboardEvent<HTMLElement>) => {
      const items = Array.from(e.clipboardData?.items ?? []);
      const files = items
        .filter((it) => it.kind === 'file')
        .map((it) => it.getAsFile())
        .filter((f): f is File => f !== null);
      if (files.length > 0) {
        e.preventDefault();
        addAttachments(files);
      }
    },
    [addAttachments],
  );

  const handleDrop = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length === 0) return;
      if (!localMode || !nativeDropAvailable()) {
        addAttachments(files);
        return;
      }
      // Desktop app: the shell sends the dropped files' real paths right
      // after this event. Those become path chips; any file it has no path
      // for (or all of them, if nothing arrives) is uploaded as before.
      void waitForNativeDrop().then((entries) => {
        const { local, upload } = splitDrop(files, entries);
        addLocalFiles(local);
        if (upload.length > 0) addAttachments(upload);
      });
    },
    [addAttachments, addLocalFiles, localMode],
  );

  const anyUploading = attachments.some((a) => a.status === 'uploading');
  const anyDone = attachments.some((a) => a.status === 'done');
  const allError =
    attachments.length > 0 && attachments.every((a) => a.status === 'error');

  return {
    attachments,
    anyUploading,
    anyDone,
    allError,
    addAttachments,
    addLocalFiles,
    openPicker,
    removeAttachment,
    retryAttachment,
    clearAttachments,
    uploadAll,
    handleFilePickerChange,
    handlePaste,
    handleDrop,
  };
}

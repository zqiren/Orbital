// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useRef, useState } from 'react';
import type { ChangeEvent, ClipboardEvent, DragEvent } from 'react';
import { BASE_URL, isRelayMode } from '../config';
import { uploadFile } from '../lib/attachment-upload';
import { useT } from '../i18n/useT';

export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
export const MAX_ATTACHMENTS = 10;

export interface PendingAttachment {
  id: string;
  file: File;
  filename: string;
  mime: string;
  size: number;
  status: 'pending' | 'uploading' | 'done' | 'error';
  uploadedPath?: string;
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
}

/**
 * Composer attachment state: staged chips, their uploads, retry, and the
 * object-URL lifetime of image thumbnails.
 *
 * Shared by the chat composer and the queue composer so the caps, the error
 * copy and the thumbnail cleanup are defined once. The hook deliberately knows
 * nothing about how the resulting paths are sent — chat inlines an
 * `<attached_files>` block into the message, the queue sends bare `file_refs`
 * and lets the dispatcher build that block at dispatch time.
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

  const uploadAttachment = useCallback(
    async (att: PendingAttachment) => {
      setAttachments((prev) =>
        prev.map((a) =>
          a.id === att.id ? { ...a, status: 'uploading' as const } : a,
        ),
      );
      try {
        const baseUrl = isRelayMode ? window.location.origin : BASE_URL;
        const { path, size } = await uploadFile({
          projectId,
          file: att.file,
          baseUrl,
          isRelayMode,
        });
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === att.id
              ? { ...a, status: 'done' as const, uploadedPath: path, size }
              : a,
          ),
        );
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
      }
    },
    [projectId],
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
        // committed the new chip state.
        setTimeout(() => {
          for (const a of additions) {
            if (a.status === 'pending') {
              uploadAttachment(a);
            }
          }
        }, 0);

        return [...prev, ...additions];
      });
    },
    [uploadAttachment],
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
        setTimeout(() => uploadAttachment(reset), 0);
        return prev.map((a) => (a.id === id ? reset : a));
      });
    },
    [uploadAttachment],
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
      if (files.length > 0) addAttachments(files);
    },
    [addAttachments],
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
    removeAttachment,
    retryAttachment,
    clearAttachments,
    handleFilePickerChange,
    handlePaste,
    handleDrop,
  };
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Check, FileText, Loader2, RotateCw, X } from 'lucide-react';
import { humanSize } from '../lib/attachment-upload';

export interface AttachmentChipProps {
  filename: string;
  mime: string;
  size: number;
  status?: 'pending' | 'uploading' | 'done' | 'error';
  thumbnailUrl?: string;
  errorMessage?: string;
  onRemove?: () => void;
  onRetry?: () => void;
}

function truncate(name: string, max = 30): string {
  if (name.length <= max) return name;
  return name.slice(0, max - 1) + '…';
}

export default function AttachmentChip({
  filename,
  mime,
  size,
  status,
  thumbnailUrl,
  errorMessage,
  onRemove,
  onRetry,
}: AttachmentChipProps) {
  const isImage = mime.startsWith('image/');
  const isError = status === 'error';
  const isPendingOrUploading = status === 'pending' || status === 'uploading';
  const isDone = status === 'done';

  const baseClass = [
    'inline-flex items-center gap-2 rounded-md border px-2 py-1 text-xs',
    'max-w-[260px]',
    isError ? 'border-red-500 bg-red-500/10' : 'border-border bg-card-hover',
    isPendingOrUploading ? 'opacity-80' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const truncated = truncate(filename);

  return (
    <div
      className={baseClass}
      data-testid="attachment-chip"
      title={isError && errorMessage ? errorMessage : filename}
    >
      {isImage && thumbnailUrl ? (
        <img
          src={thumbnailUrl}
          alt=""
          className="w-8 h-8 rounded object-cover shrink-0"
        />
      ) : (
        <FileText
          className={`w-4 h-4 shrink-0 ${isPendingOrUploading ? 'opacity-50' : ''}`}
          aria-hidden="true"
        />
      )}

      <div className="flex flex-col min-w-0">
        <span
          className="truncate max-w-[140px] font-medium"
          title={filename}
        >
          {truncated}
        </span>
        <span className="text-[10px] text-secondary">{humanSize(size)}</span>
      </div>

      {isPendingOrUploading && (
        <Loader2
          className="w-4 h-4 shrink-0 animate-spin"
          data-testid="chip-spinner"
          aria-label="Uploading"
        />
      )}

      {isDone && (
        <Check
          className="w-4 h-4 shrink-0 text-green-500"
          data-testid="chip-check"
          aria-label="Uploaded"
        />
      )}

      {isError && onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          aria-label="Retry upload"
          className="shrink-0 p-1 hover:text-foreground text-red-500 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
        >
          <RotateCw className="w-4 h-4" />
        </button>
      ) : null}

      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove attachment"
          className="shrink-0 p-1 text-muted-foreground hover:text-foreground rounded max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}

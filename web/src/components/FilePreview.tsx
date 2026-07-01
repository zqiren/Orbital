// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useState, type IframeHTMLAttributes } from 'react';
import { File, Download, Copy, Check } from 'lucide-react';
import type { FileContent } from '../types';
import MarkdownContent from './MarkdownContent';
import { useT } from '../i18n/useT';

export interface FilePreviewProps {
  fileContent: FileContent | null;
  loading: boolean;
  selectedPath: string | null;
}

// Defense-in-depth CSP for the HTML preview iframe (spec 003 §0.1). The
// load-bearing control is the OPAQUE ORIGIN from omitting `allow-same-origin`;
// this string is belt-and-suspenders only. Note the iframe `csp` attribute is
// not broadly enforced (it was dropped from Blink), so treat it as a hint, not
// a guarantee — it can only ever further restrict, never weaken the sandbox.
const HTML_IFRAME_CSP =
  "default-src 'none'; script-src 'unsafe-inline' https:; " +
  "style-src 'unsafe-inline'; img-src data: https:; font-src data:";

/**
 * Read-only file preview pane (spec 002 §3.3). Lifted verbatim out of
 * `FileExplorer.tsx` into its own module so both the Files-tab explorer and the
 * `FilePreviewDrawer` (clickable chat paths) share one renderer. Behavior is
 * unchanged: image / binary / text(+markdown) variants, copy + download.
 */
export default function FilePreview({ fileContent, loading, selectedPath }: FilePreviewProps) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  // HTML preview view mode. Ephemeral + shared across files (spec 003 §3.6):
  // defaults to "rendered", a toggle drops back to the source <pre>.
  const [htmlViewMode, setHtmlViewMode] = useState<'rendered' | 'source'>('rendered');
  // Save raw text/HTML to disk. A `download` anchor writes a file and never
  // renders/executes in the app origin — safe even though blob: URLs are
  // SAME-ORIGIN with the app (which is exactly why we must NOT window.open one:
  // a top-level blob: document would run agent <script> with access to the
  // app's localStorage relay JWT). Mirrors `handleDownload` minus the base64
  // decode, since HTML content arrives as a raw string.
  const handleDownloadRaw = useCallback((content: string, mime: string, filename: string) => {
    const blob = new Blob([content], { type: mime || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);
  const handleDownload = useCallback((content: string, mime: string, filename: string) => {
    const bytes = Uint8Array.from(atob(content), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: mime || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);

  if (!selectedPath) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <p className="text-sm text-secondary">{t('fileExplorer.selectFile')}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="h-5 w-48 bg-sidebar rounded animate-pulse mb-4" />
        <div className="space-y-2">
          <div className="h-4 w-full bg-sidebar rounded animate-pulse" />
          <div className="h-4 w-3/4 bg-sidebar rounded animate-pulse" />
          <div className="h-4 w-5/6 bg-sidebar rounded animate-pulse" />
          <div className="h-4 w-2/3 bg-sidebar rounded animate-pulse" />
          <div className="h-4 w-4/5 bg-sidebar rounded animate-pulse" />
          <div className="h-4 w-1/2 bg-sidebar rounded animate-pulse" />
        </div>
      </div>
    );
  }

  if (!fileContent) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <p className="text-sm text-secondary">{t('fileExplorer.unableToLoad')}</p>
      </div>
    );
  }

  const fileName = fileContent.path.split('/').pop() ?? fileContent.path;
  const fileType = fileContent.type ?? 'text';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(fileContent.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard API may fail in insecure contexts
    }
  };

  // Image preview
  if (fileType === 'image') {
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
          <span className="text-xs text-secondary ml-2 shrink-0">{formatSize(fileContent.size)}</span>
        </div>
        <div className="flex-1 overflow-auto flex items-center justify-center p-4 bg-sidebar">
          <img
            src={`data:${fileContent.mime ?? 'image/png'};base64,${fileContent.content}`}
            alt={fileName}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </div>
      </div>
    );
  }

  // Binary file info card
  if (fileType === 'binary') {
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b border-border">
          <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
        </div>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="bg-sidebar rounded-lg p-6 max-w-sm w-full text-center">
            <File size={48} className="mx-auto text-secondary mb-4" />
            <p className="font-semibold text-sm text-primary mb-1">{fileName}</p>
            <p className="text-xs text-secondary mb-1">{formatSize(fileContent.size)}</p>
            {fileContent.mime && (
              <p className="text-xs text-secondary mb-4">{fileContent.mime}</p>
            )}
            {fileContent.content && (
              <button
                onClick={() => handleDownload(fileContent.content, fileContent.mime || 'application/octet-stream', fileName)}
                className="inline-flex items-center gap-1.5 bg-accent text-white text-sm font-medium rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150"
              >
                <Download size={14} />
                {t('fileExplorer.download')}
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // HTML preview: render visually in a sandboxed iframe (spec 003), with a
  // "Rendered | Source" toggle and an "Open in new tab" escape hatch.
  if (fileType === 'html') {
    const showSource = htmlViewMode === 'source';
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-2">
          <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-xs text-secondary">{formatSize(fileContent.size)}</span>
            <div className="flex items-center rounded-md border border-border overflow-hidden">
              <button
                onClick={() => setHtmlViewMode('rendered')}
                aria-pressed={!showSource}
                className={`text-xs px-2 py-1 transition-colors ${
                  showSource ? 'text-secondary hover:text-primary' : 'bg-accent text-white'
                }`}
              >
                {t('fileExplorer.rendered')}
              </button>
              <button
                onClick={() => setHtmlViewMode('source')}
                aria-pressed={showSource}
                className={`text-xs px-2 py-1 transition-colors ${
                  showSource ? 'bg-accent text-white' : 'text-secondary hover:text-primary'
                }`}
              >
                {t('fileExplorer.source')}
              </button>
            </div>
            <button
              onClick={() => handleDownloadRaw(fileContent.content, 'text/html', fileName)}
              className="flex items-center gap-1 text-xs text-secondary hover:text-primary transition-colors"
            >
              <Download size={14} />
              {t('fileExplorer.download')}
            </button>
          </div>
        </div>
        {fileContent.truncated && (
          <div className="px-4 py-2 bg-sidebar border-b border-border">
            <p className="text-xs text-secondary">{t('fileExplorer.truncated')}</p>
          </div>
        )}
        {showSource ? (
          <div className="flex-1 overflow-auto">
            <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
              {fileContent.content}
            </pre>
          </div>
        ) : (
          <div className="flex-1 flex flex-col min-h-0">
            <p className="px-4 py-1.5 text-xs text-secondary bg-sidebar border-b border-border shrink-0">
              {t('fileExplorer.sandboxedHint')}
            </p>
            <iframe
              key={fileContent.path}
              title={fileName}
              srcDoc={fileContent.content}
              sandbox="allow-scripts"
              referrerPolicy="no-referrer"
              className="flex-1 w-full border-0 bg-white"
              // `csp` is a non-standard, inconsistently-enforced iframe
              // attribute that can only further restrict (never weaken) — the
              // opaque-origin sandbox is the real control. Not in React's TS
              // types, so bridge it as an extra prop.
              {...({ csp: HTML_IFRAME_CSP } as unknown as IframeHTMLAttributes<HTMLIFrameElement>)}
            />
          </div>
        )}
      </div>
    );
  }

  // Text preview (default, backward compatible)
  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-secondary hover:text-primary transition-colors ml-2 shrink-0"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? t('fileExplorer.copied') : t('fileExplorer.copy')}
        </button>
      </div>
      {fileContent.truncated && (
        <div className="px-4 py-2 bg-sidebar border-b border-border">
          <p className="text-xs text-secondary">
            {t('fileExplorer.truncated')}
          </p>
        </div>
      )}
      <div className="flex-1 overflow-auto">
        {fileName.toLowerCase().endsWith('.md') ? (
          <div className="bg-sidebar p-4 min-h-full">
            <MarkdownContent content={fileContent.content} />
          </div>
        ) : (
          <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
            {fileContent.content}
          </pre>
        )}
      </div>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

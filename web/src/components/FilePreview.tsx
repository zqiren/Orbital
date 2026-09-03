// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type IframeHTMLAttributes,
  type ReactNode,
} from 'react';
import { File, Download, Copy, Check, Pencil } from 'lucide-react';
import type { FileContent } from '../types';
import MarkdownContent from './MarkdownContent';
import AnnotateOverlay from './panel/AnnotateOverlay';
import type { AnnotationBox } from '../utils/annotations';
import { useT } from '../i18n/useT';

export interface FilePreviewProps {
  fileContent: FileContent | null;
  loading: boolean;
  selectedPath: string | null;
  /**
   * Persist edited content for a `.md` file (last-write-wins). Resolves `true`
   * on success, `false` on failure. Its presence is what makes markdown
   * EDITABLE — omit it and the pane stays read-only (chat/Files both pass it,
   * threading the project id). Never wired for a truncated file (see `editable`).
   */
  onSave?: (path: string, content: string) => Promise<boolean>;
  /**
   * Spec 078 §5.4 (workstream E): when true, selecting text shows a "Quote"
   * affordance and `onQuote` receives {path, text, lines?}. Optional; the
   * panel's FilesView passes them, the Files tab does not.
   */
  quoting?: boolean;
  /**
   * text present → a text-span quote; box present → an image-region quote
   * (imageDataUrl = the rendered image for the annotated PNG); neither → a
   * whole-file quote (binary previews). FilesView maps this to AnnotationDraft.
   */
  onQuote?: (quote: {
    path: string;
    text?: string;
    lines?: [number, number];
    box?: { x: number; y: number; w: number; h: number };
    imageDataUrl?: string;
  }) => void;
}

// Defense-in-depth CSP for the HTML preview iframe (spec 003 §0.1). The
// load-bearing control is the OPAQUE ORIGIN from omitting `allow-same-origin`;
// this string is belt-and-suspenders only. Note the iframe `csp` attribute is
// not broadly enforced (it was dropped from Blink), so treat it as a hint, not
// a guarantee — it can only ever further restrict, never weaken the sandbox.
const HTML_IFRAME_CSP =
  "default-src 'none'; script-src 'unsafe-inline' https:; " +
  "style-src 'unsafe-inline'; img-src data: https:; font-src data:";

type QuoteFn = NonNullable<FilePreviewProps['onQuote']>;

/** 1-based line number of a character offset in `source`. */
function lineAt(source: string, offset: number): number {
  let n = 1;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source.charCodeAt(i) === 10) n++;
  }
  return n;
}

/**
 * Inclusive 1-based line range for a half-open character range. The end line
 * is taken from the LAST selected character, so a selection that stops right
 * after a newline does not claim the following line.
 */
function linesFromOffsets(
  source: string,
  start: number,
  end: number,
): [number, number] | undefined {
  if (start < 0 || end <= start || end > source.length) return undefined;
  return [lineAt(source, start), lineAt(source, Math.max(start, end - 1))];
}

/**
 * Line range for a rendered view, where DOM offsets do not map to the source:
 * resolved only when the selected text occurs EXACTLY ONCE (spec §13 Q4 —
 * "text always; lines when unique"). Ambiguous text keeps the verbatim quote
 * and drops the line numbers rather than guessing.
 */
function linesFromUniqueMatch(source: string, text: string): [number, number] | undefined {
  const first = source.indexOf(text);
  if (first < 0) return undefined;
  if (source.indexOf(text, first + 1) !== -1) return undefined;
  return linesFromOffsets(source, first, first + text.length);
}

/** Character offset of (node, offset) within `host`'s concatenated text. */
function textOffsetWithin(host: Node, node: Node, offset: number): number | null {
  if (node.nodeType !== Node.TEXT_NODE) return null;
  const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
  let total = 0;
  let cur: Node | null;
  while ((cur = walker.nextNode()) !== null) {
    if (cur === node) return total + offset;
    total += (cur.textContent ?? '').length;
  }
  return null;
}

/**
 * Spec 078 §5.4 — a text-like preview you can quote from. Wraps the existing
 * body untouched and adds a "Quote" pill on mouse-up over a non-empty
 * selection. Only mounted when `quoting` is on, so the non-panel Files tab
 * renders byte-identically to before.
 *
 * `exact` marks the source views (`<pre>`), where the DOM text IS the file
 * text and selection offsets translate straight to line numbers. The rendered
 * markdown view is not exact and falls back to the unique-match rule.
 */
function QuoteRegion({
  source,
  exact,
  path,
  onQuote,
  className,
  children,
}: {
  source: string;
  exact: boolean;
  path: string;
  onQuote: QuoteFn;
  className?: string;
  children: ReactNode;
}) {
  const t = useT();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [pill, setPill] = useState<
    { text: string; lines?: [number, number]; left: number; top: number } | null
  >(null);

  const handleMouseUp = () => {
    const host = hostRef.current;
    const sel = typeof window !== 'undefined' ? window.getSelection() : null;
    if (!host || !sel || sel.rangeCount === 0) {
      setPill(null);
      return;
    }
    const text = sel.toString();
    if (!text.trim()) {
      setPill(null);
      return;
    }
    const range = sel.getRangeAt(0);
    if (!host.contains(range.commonAncestorContainer)) {
      setPill(null);
      return;
    }

    let lines: [number, number] | undefined;
    if (exact) {
      const start = textOffsetWithin(host, range.startContainer, range.startOffset);
      const end = textOffsetWithin(host, range.endContainer, range.endOffset);
      lines =
        start !== null && end !== null
          ? linesFromOffsets(source, start, end)
          : linesFromUniqueMatch(source, text);
    } else {
      lines = linesFromUniqueMatch(source, text);
    }

    // Anchor the pill under the selection. jsdom has no Range rect, and a
    // missing rect just parks the pill at the region's top-left.
    const rect =
      typeof range.getBoundingClientRect === 'function'
        ? range.getBoundingClientRect()
        : null;
    const hostRect = host.getBoundingClientRect();
    setPill({
      text,
      lines,
      left: rect ? Math.max(0, rect.left - hostRect.left) : 0,
      top: rect ? Math.max(0, rect.bottom - hostRect.top + 4) : 0,
    });
  };

  return (
    <div
      ref={hostRef}
      className={`relative ${className ?? ''}`}
      onMouseUp={handleMouseUp}
      data-testid="quote-region"
    >
      {children}
      {pill && (
        <button
          type="button"
          // mousedown would collapse the selection before the click lands.
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            onQuote({ path, text: pill.text, ...(pill.lines ? { lines: pill.lines } : {}) });
            setPill(null);
            try {
              window.getSelection()?.removeAllRanges();
            } catch {
              // ignore
            }
          }}
          style={{ position: 'absolute', left: pill.left, top: pill.top, zIndex: 10 }}
          className="text-[11px] font-medium px-2 py-1 rounded-md bg-accent text-white shadow-sm"
        >
          {t('panel.files.quote')}
        </button>
      )}
    </div>
  );
}

/**
 * Read-only file preview pane (spec 002 §3.3). Lifted verbatim out of
 * `FileExplorer.tsx` into its own module so both the Files-tab explorer and the
 * `FilePreviewDrawer` (clickable chat paths) share one renderer. Behavior is
 * unchanged: image / binary / text(+markdown) variants, copy + download.
 *
 * Spec 078 §5.4 adds an opt-in quoting layer (`quoting` + `onQuote`): text
 * selection in the text-like views, a drag box over images, and a whole-file
 * action for previews with no renderer. With `quoting` off nothing about the
 * pane changes.
 */
export default function FilePreview({
  fileContent,
  loading,
  selectedPath,
  onSave,
  quoting,
  onQuote,
}: FilePreviewProps) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  // HTML preview view mode. Ephemeral + shared across files (spec 003 §3.6):
  // defaults to "rendered", a toggle drops back to the source <pre>.
  const [htmlViewMode, setHtmlViewMode] = useState<'rendered' | 'source'>('rendered');
  // Markdown editing state (all hooks live ABOVE the early returns — rules of
  // hooks). `draft` is the textarea buffer; `override` holds the just-saved
  // draft so the view pane reflects the save even though `fileContent` is a
  // prop the parent hasn't re-fetched. `editView` toggles the textarea vs. a
  // live MarkdownContent preview of the draft.
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [editView, setEditView] = useState<'write' | 'preview'>('write');
  const [override, setOverride] = useState<string | null>(null);
  // Spec 078: the rendered <img>, so a drag box measured in on-screen pixels
  // can be reported to the agent in IMAGE pixels (§5.4 "box in image pixels").
  const imgRef = useRef<HTMLImageElement | null>(null);
  const toImagePixels = useCallback((box: AnnotationBox): AnnotationBox => {
    const el = imgRef.current;
    const cw = el?.clientWidth ?? 0;
    const ch = el?.clientHeight ?? 0;
    const nw = el?.naturalWidth ?? 0;
    const nh = el?.naturalHeight ?? 0;
    if (cw <= 0 || ch <= 0 || nw <= 0 || nh <= 0) return box;
    const sx = nw / cw;
    const sy = nh / ch;
    return {
      x: Math.round(box.x * sx),
      y: Math.round(box.y * sy),
      w: Math.round(box.w * sx),
      h: Math.round(box.h * sy),
    };
  }, []);
  // Selecting a different file discards any in-flight edit + saved override so
  // one file's draft never bleeds into the next.
  useEffect(() => {
    setEditing(false);
    setDraft('');
    setSaving(false);
    setSaveError(false);
    setEditView('write');
    setOverride(null);
  }, [selectedPath]);
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
  // What a quote names. `selectedPath` is the path the caller asked for (and
  // is non-null past the guard above); `fileContent.path` is the fallback.
  const quotePath = selectedPath || fileContent.path;

  const handleCopy = async () => {
    try {
      // Copy the currently-shown content — the saved override when present, so
      // Copy after an edit yields the new text, not the stale prop.
      await navigator.clipboard.writeText(override ?? fileContent.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard API may fail in insecure contexts
    }
  };

  // Image preview
  if (fileType === 'image') {
    const imageDataUrl = `data:${fileContent.mime ?? 'image/png'};base64,${fileContent.content}`;
    const img = (
      <img
        ref={imgRef}
        src={imageDataUrl}
        alt={fileName}
        style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
      />
    );
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
          <span className="text-xs text-secondary ml-2 shrink-0">{formatSize(fileContent.size)}</span>
        </div>
        <div className="flex-1 overflow-auto flex items-center justify-center p-4 bg-sidebar">
          {quoting && onQuote ? (
            // The wrapper shrink-wraps the <img>, so the overlay's coordinate
            // space IS the displayed image (no letterboxing to correct for).
            <div className="relative inline-block max-w-full max-h-full leading-none">
              {img}
              <AnnotateOverlay
                active
                boxes={[]}
                onAdd={(box) =>
                  onQuote({
                    path: quotePath,
                    box: toImagePixels(box),
                    imageDataUrl,
                  })
                }
              />
            </div>
          ) : (
            img
          )}
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
            {/* Spec 078 §5.4: a preview with no renderer still has one thing
                worth quoting — the file itself. */}
            {quoting && onQuote && (
              <button
                onClick={() => onQuote({ path: quotePath })}
                className="mt-3 inline-flex items-center gap-1.5 border border-border text-primary text-sm font-medium rounded-lg px-4 py-2 hover:bg-card transition-all duration-150"
              >
                {t('panel.files.quoteFile')}
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
            {quoting && onQuote ? (
              <QuoteRegion
                source={fileContent.content}
                exact
                path={quotePath}
                onQuote={onQuote}
                className="min-h-full"
              >
                <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
                  {fileContent.content}
                </pre>
              </QuoteRegion>
            ) : (
              <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
                {fileContent.content}
              </pre>
            )}
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

  // Text preview (default, backward compatible). Markdown (`.md`) is
  // additionally EDITABLE when an `onSave` handler is supplied and the file was
  // not truncated (editing a partial draft would clobber the tail on save).
  const isMarkdown = fileName.toLowerCase().endsWith('.md');
  const editable = isMarkdown && !!onSave && !fileContent.truncated;
  // View-mode content: the saved override wins over the fetched prop.
  const viewContent = override ?? fileContent.content;

  const handleStartEdit = () => {
    setDraft(viewContent);
    setSaveError(false);
    setEditView('write');
    setEditing(true);
  };
  const handleCancelEdit = () => {
    setEditing(false);
    setDraft('');
    setSaveError(false);
  };
  const handleSave = async () => {
    if (!onSave) return;
    setSaving(true);
    setSaveError(false);
    const ok = await onSave(fileContent.path, draft);
    if (ok) {
      // Reflect the save locally (parent doesn't re-fetch) and exit edit mode.
      setOverride(draft);
      setEditing(false);
    } else {
      // Keep the draft so the user doesn't lose edits; surface the failure.
      setSaveError(true);
    }
    setSaving(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-2">
        <h3 className="font-semibold text-sm text-primary truncate">{fileName}</h3>
        <div className="flex items-center gap-2 shrink-0">
          {editing ? (
            <>
              {/* Write | Preview segmented toggle (mirrors the html Rendered|Source control). */}
              <div className="flex items-center rounded-md border border-border overflow-hidden">
                <button
                  onClick={() => setEditView('write')}
                  aria-pressed={editView === 'write'}
                  className={`text-xs px-2 py-1 transition-colors ${
                    editView === 'write' ? 'bg-accent text-white' : 'text-secondary hover:text-primary'
                  }`}
                >
                  {t('fileExplorer.editWrite')}
                </button>
                <button
                  onClick={() => setEditView('preview')}
                  aria-pressed={editView === 'preview'}
                  className={`text-xs px-2 py-1 transition-colors ${
                    editView === 'preview' ? 'bg-accent text-white' : 'text-secondary hover:text-primary'
                  }`}
                >
                  {t('fileExplorer.editPreview')}
                </button>
              </div>
              <button
                onClick={handleCancelEdit}
                disabled={saving}
                className="text-xs text-secondary hover:text-primary transition-colors disabled:opacity-50"
              >
                {t('fileExplorer.cancel')}
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="bg-accent text-white text-xs font-medium rounded-md px-2.5 py-1 hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {saving ? t('fileExplorer.saving') : t('fileExplorer.save')}
              </button>
            </>
          ) : (
            <>
              {editable && (
                <button
                  onClick={handleStartEdit}
                  className="flex items-center gap-1 text-xs text-secondary hover:text-primary transition-colors"
                >
                  <Pencil size={14} />
                  {t('fileExplorer.edit')}
                </button>
              )}
              {isMarkdown && fileContent.truncated && (
                <span className="text-xs text-secondary opacity-70" aria-disabled="true">
                  {t('fileExplorer.editTooLarge')}
                </span>
              )}
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-xs text-secondary hover:text-primary transition-colors"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
                {copied ? t('fileExplorer.copied') : t('fileExplorer.copy')}
              </button>
            </>
          )}
        </div>
      </div>
      {fileContent.truncated && (
        <div className="px-4 py-2 bg-sidebar border-b border-border">
          <p className="text-xs text-secondary">
            {t('fileExplorer.truncated')}
          </p>
        </div>
      )}
      {saveError && (
        <div className="px-4 py-2 bg-error/10 border-b border-error/30">
          <p className="text-xs text-error">{t('fileExplorer.saveFailed')}</p>
        </div>
      )}
      {editing ? (
        editView === 'preview' ? (
          <div className="flex-1 overflow-auto bg-sidebar p-4 min-h-0">
            <MarkdownContent content={draft} />
          </div>
        ) : (
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            spellCheck={false}
            aria-label={t('fileExplorer.edit')}
            className="flex-1 w-full min-h-0 resize-none font-mono text-sm text-primary bg-sidebar p-4 outline-none border-0"
          />
        )
      ) : (
        <div className="flex-1 overflow-auto">
          {(() => {
            const body = isMarkdown ? (
              <div className="bg-sidebar p-4 min-h-full">
                <MarkdownContent content={viewContent} />
              </div>
            ) : (
              <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
                {viewContent}
              </pre>
            );
            if (!quoting || !onQuote) return body;
            // The <pre> renders the file text verbatim, so selection offsets
            // ARE source offsets. The rendered markdown view is not exact and
            // falls back to the unique-match rule for line numbers.
            return (
              <QuoteRegion
                source={viewContent}
                exact={!isMarkdown}
                path={quotePath}
                onQuote={onQuote}
                className="min-h-full"
              >
                {body}
              </QuoteRegion>
            );
          })()}
        </div>
      )}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.3 (D10) — Files view: tree state (workspace tree, touched files
 * badged + "Touched this session" group) ⇄ preview state (FilePreview in
 * place; since spec 088 its one header row carries Back and a More menu with
 * "Open in Files"). Never both at once.
 * CONTRACT FILE: props are final.
 *
 * The tree is a flat `Map<dirPath, entries>` + an expanded set rather than the
 * nested mutable tree the Files *tab* keeps (`FileExplorer.tsx`). The panel has
 * to auto-expand every ancestor of every touched file, which is one line
 * against a map and a recursive rewrite against a tree — and it sidesteps the
 * `setState`-updater closure trap `FileExplorer.toggleDirectory` needs
 * `flushSync` for (CLAUDE.md, React anti-patterns): nothing here reads a
 * variable an updater wrote.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, File, Folder } from 'lucide-react';
import type { FileContent, FileEntry, WebSocketEvent } from '../../types';
import type { AnnotationDraft } from '../../utils/annotations';
import type { TouchedFile, TouchedOp } from '../../utils/panelSelectors';
import { fetchPathWithFallback } from '../../utils/openPathWithFallback';
import { canRevealInFileManager } from '../../utils/clientPlatform';
import { useFiles } from '../../hooks/useFiles';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useT } from '../../i18n/useT';
import type { StringKey } from '../../i18n/strings';
import FilePreview from '../FilePreview';

export interface FilesViewProps {
  projectId: string;
  touched: TouchedFile[];
  /** Current file in preview state, or null for the tree state. */
  file: string | null;
  onSelectFile: (path: string | null) => void;
  onOpenInFiles: (path: string) => void;
  onAddAnnotation: (draft: AnnotationDraft) => void;
  onSave?: (path: string, content: string) => Promise<boolean>;
}

const OP_LABEL: Record<TouchedOp, StringKey> = {
  read: 'panel.files.read',
  edited: 'panel.files.edited',
  written: 'panel.files.written',
};

/** Agents write `./x`, `x`, and sometimes a trailing slash — compare one shape. */
function normalizePath(path: string): string {
  return path.replace(/^\.\//, '').replace(/\/+$/, '');
}

/** Every directory that must be open for `path` to be visible ('' = root). */
function ancestorsOf(path: string): string[] {
  const parts = normalizePath(path).split('/');
  parts.pop();
  const out: string[] = [];
  let acc = '';
  for (const part of parts) {
    acc = acc ? `${acc}/${part}` : part;
    out.push(acc);
  }
  return out;
}

/** Bursts of events (a turn's parallel tool results) become one refresh. */
const REFRESH_COALESCE_MS = 150;

/**
 * Whether a re-read brought anything new. `revision` (mtime + size) is the
 * daemon's answer; older daemons don't send one, so fall back to the payload.
 */
function sameContent(a: FileContent | null, b: FileContent | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.path !== b.path) return false;
  if (a.revision !== undefined && b.revision !== undefined) return a.revision === b.revision;
  return a.size === b.size && a.content === b.content && a.type === b.type;
}

function sameEntries(a: FileEntry[] | undefined, b: FileEntry[]): boolean {
  if (a === undefined || a.length !== b.length) return false;
  return a.every(
    (entry, i) =>
      entry.name === b[i].name &&
      entry.type === b[i].type &&
      entry.size === b[i].size &&
      entry.modified_at === b[i].modified_at,
  );
}

/** Same ordering as the Files tab: agent_output pinned, directories, then name. */
function sortEntries(entries: FileEntry[]): FileEntry[] {
  return [...entries].sort((a, b) => {
    if (a.name === 'agent_output' && a.type === 'directory') return -1;
    if (b.name === 'agent_output' && b.type === 'directory') return 1;
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

export default function FilesView({
  projectId,
  touched,
  file,
  onSelectFile,
  onOpenInFiles,
  onAddAnnotation,
  onSave,
}: FilesViewProps) {
  const t = useT();
  const { listDirectory, getFileContent, resolvePath, revealPath } = useFiles();

  // ── tree state ──────────────────────────────────────────────────────────
  const [dirs, setDirs] = useState<Map<string, FileEntry[]>>(() => new Map());
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [pendingDirs, setPendingDirs] = useState<Set<string>>(() => new Set());
  const [touchedOpen, setTouchedOpen] = useState(true);
  // Directories already requested — a ref, so two callers in one tick cannot
  // both decide to fetch (state would still be the pre-update value for the
  // second one).
  const requestedRef = useRef<Set<string>>(new Set());

  const ensureDir = useCallback(
    async (path: string) => {
      if (requestedRef.current.has(path)) return;
      requestedRef.current.add(path);
      setPendingDirs((prev) => new Set(prev).add(path));
      const listing = await listDirectory(projectId, path || undefined);
      setDirs((prev) => new Map(prev).set(path, sortEntries(listing?.entries ?? [])));
      setPendingDirs((prev) => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
    },
    [listDirectory, projectId],
  );

  // ── live refresh ────────────────────────────────────────────────────────
  // The panel is how a user watches the agent work, and nothing here used to
  // look at the disk twice: a directory was listed once, a file read once per
  // selection — and ChatTab opens a file on the tool CALL event, before the
  // write has run, so the one read was routinely the pre-edit snapshot.
  //
  // The daemon has no file watcher, but it does say when something may have
  // changed: a tool result landed (write / edit / shell all end in one), a
  // sub-agent worker produced output, or the run changed state. Each bumps
  // `refreshTick`; the tree and the open file re-read themselves on it.
  // Events only — no polling.
  const { on, off } = useWebSocket();
  const [refreshTick, setRefreshTick] = useState(0);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const bump = () => {
      if (timer !== null) return;
      timer = setTimeout(() => {
        timer = null;
        setRefreshTick((n) => n + 1);
      }, REFRESH_COALESCE_MS);
    };
    const mine = (event: WebSocketEvent) =>
      (event as { project_id?: string }).project_id === projectId;
    const onActivity = (event: WebSocketEvent) => {
      // Results, not calls: on the call event the tool has not run yet.
      if (mine(event) && (event as { category?: string }).category === 'tool_result') bump();
    };
    const onProjectEvent = (event: WebSocketEvent) => {
      if (mine(event)) bump();
    };
    // Coming back to the window: whatever happened meanwhile, show it.
    const onVisible = () => {
      if (document.visibilityState === 'visible') bump();
    };
    on('agent.activity', onActivity);
    on('chat.sub_agent_message', onProjectEvent);
    on('agent.status', onProjectEvent);
    window.addEventListener('focus', bump);
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      if (timer !== null) clearTimeout(timer);
      off('agent.activity', onActivity);
      off('chat.sub_agent_message', onProjectEvent);
      off('agent.status', onProjectEvent);
      window.removeEventListener('focus', bump);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [projectId, on, off]);

  // Re-list what is on screen: the root and the folders currently expanded
  // (spec 103 R3 — not every folder ever listed, and nothing at all in preview
  // state, where the tree is not mounted). Read through refs so the effect
  // depends on the tick alone — not on the state it is about to update.
  const dirsRef = useRef(dirs);
  const expandedRef = useRef(expanded);
  const fileRef = useRef(file);
  useEffect(() => {
    dirsRef.current = dirs;
    expandedRef.current = expanded;
    fileRef.current = file;
  });
  useEffect(() => {
    if (refreshTick === 0 || fileRef.current !== null) return;
    let cancelled = false;
    const onScreen = [...dirsRef.current.keys()].filter(
      (path) => path === '' || expandedRef.current.has(path),
    );
    for (const path of onScreen) {
      void listDirectory(projectId, path || undefined).then((listing) => {
        // A failed listing keeps what is shown rather than blanking the tree.
        if (cancelled || !listing) return;
        const next = sortEntries(listing.entries);
        setDirs((prev) => (sameEntries(prev.get(path), next) ? prev : new Map(prev).set(path, next)));
      });
    }
    return () => {
      cancelled = true;
    };
  }, [refreshTick, listDirectory, projectId]);

  // Leaving the preview: the tree sat out every tick meanwhile, so give it one.
  const prevFileRef = useRef(file);
  useEffect(() => {
    const was = prevFileRef.current;
    prevFileRef.current = file;
    if (was !== null && file === null) setRefreshTick((n) => n + 1);
  }, [file]);

  // Switching project throws the whole tree away.
  useEffect(() => {
    requestedRef.current = new Set();
    setDirs(new Map());
    setExpanded(new Set());
    setPendingDirs(new Set());
    void ensureDir('');
  }, [projectId, ensureDir]);

  // D10: the folders holding this session's touched files are open on arrival,
  // so the badges are visible without hunting.
  const touchedKey = touched.map((entry) => entry.path).join(' ');
  useEffect(() => {
    const needed = new Set<string>();
    for (const entry of touched) for (const dir of ancestorsOf(entry.path)) needed.add(dir);
    if (needed.size === 0) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      needed.forEach((dir) => next.add(dir));
      return next;
    });
    // Shallowest first so a parent's listing is in flight before its child's.
    [...needed]
      .sort((a, b) => a.split('/').length - b.split('/').length)
      .forEach((dir) => void ensureDir(dir));
    // `touchedKey` is the value identity of `touched` (a fresh array each render).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [touchedKey, ensureDir]);

  const toggleDir = useCallback(
    (path: string) => {
      setExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(path)) next.delete(path);
        else next.add(path);
        return next;
      });
      void ensureDir(path);
    },
    [ensureDir],
  );

  const touchedByPath = useMemo(() => {
    const map = new Map<string, TouchedOp>();
    for (const entry of touched) map.set(normalizePath(entry.path), entry.op);
    return map;
  }, [touched]);

  // ── preview state ───────────────────────────────────────────────────────
  const [content, setContent] = useState<FileContent | null>(null);
  const [contentLoading, setContentLoading] = useState(false);
  const latestRequestRef = useRef<string | null>(null);

  useEffect(() => {
    if (file === null) {
      latestRequestRef.current = null;
      setContent(null);
      setContentLoading(false);
      return;
    }
    latestRequestRef.current = file;
    setContentLoading(true);
    void fetchPathWithFallback(
      (p) => getFileContent(projectId, p),
      (p) => resolvePath(projectId, p),
      file,
    ).then((outcome) => {
      // Drop a slow fetch that a newer selection superseded.
      if (latestRequestRef.current !== file) return;
      setContentLoading(false);
      setContent(outcome.status === 'ok' ? outcome.content : null);
    });
  }, [file, projectId, getFileContent, resolvePath]);

  // Silent re-read of the open file: no loading state (the view must not
  // flash, and FilePreview keeps its scroll position), and the content object
  // is only replaced when the file actually changed, so an unchanged file
  // costs one small request and zero renders. An edit in progress is safe —
  // FilePreview's draft is its own state and survives a content swap.
  //
  // Spec 103 (P2): the re-read is conditional. It carries the revision of the
  // content on screen; a daemon that knows `if_revision` answers ~100 bytes
  // when it still matches, so an unchanged image or binary never travels
  // again. Only the first read of a selection (the effect above) is sent
  // bare. An older daemon ignores the param and the revision compare below
  // absorbs the full body as before.
  const contentRef = useRef(content);
  useEffect(() => {
    contentRef.current = content;
  });
  useEffect(() => {
    if (refreshTick === 0 || file === null) return;
    const held = contentRef.current;
    void fetchPathWithFallback(
      (p) => getFileContent(projectId, p, held !== null && held.path === p ? held.revision : undefined),
      (p) => resolvePath(projectId, p),
      file,
    ).then((outcome) => {
      if (latestRequestRef.current !== file) return;
      // The first read is still in flight — it will land fresh on its own.
      const next = outcome.status === 'ok' ? outcome.content : null;
      // A transient miss (file mid-rename, mid-write) keeps the last good view.
      if (next === null) return;
      // Nothing moved on disk: keep what is shown, touch nothing.
      if (next.unchanged) return;
      setContentLoading(false);
      setContent((prev) => (sameContent(prev, next) ? prev : next));
    });
    // `file` is read, not watched: a selection change has its own effect above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTick]);

  // Spec 078 §5.4: FilePreview reports what the user selected; the shape of
  // the quote decides which annotation form it becomes.
  const handleQuote = useCallback(
    (quote: {
      path: string;
      text?: string;
      lines?: [number, number];
      box?: { x: number; y: number; w: number; h: number };
      imageDataUrl?: string;
    }) => {
      if (quote.text !== undefined) {
        onAddAnnotation({
          kind: 'text',
          path: quote.path,
          text: quote.text,
          lines: quote.lines,
          note: '',
        });
        return;
      }
      if (quote.box !== undefined) {
        onAddAnnotation({
          kind: 'image',
          path: quote.path,
          box: quote.box,
          note: '',
          imageDataUrl: quote.imageDataUrl,
        });
        return;
      }
      onAddAnnotation({ kind: 'file', path: quote.path, note: '' });
    },
    [onAddAnnotation],
  );

  // ── preview state render (never alongside the tree — D10) ───────────────
  // Spec 088: no bar of its own. Back and "Open in Files" ride in the
  // preview's single header row, next to the filename and the file actions.
  // Spec 093: "Reveal in Finder" joins them in More where the reveal can work.
  if (file !== null) {
    return (
      <div className="flex flex-col h-full min-h-0" data-testid="files-view-preview">
        <div className="flex-1 overflow-y-auto min-h-0">
          <FilePreview
            fileContent={content}
            loading={contentLoading}
            selectedPath={content?.path ?? file}
            onSave={onSave}
            quoting
            onQuote={handleQuote}
            panelHeader={{
              onBack: () => onSelectFile(null),
              onOpenInFiles: () => onOpenInFiles(content?.path ?? file),
            }}
            onReveal={
              canRevealInFileManager()
                ? (path) => void revealPath(projectId, path)
                : undefined
            }
          />
        </div>
      </div>
    );
  }

  // ── tree state render ───────────────────────────────────────────────────
  const renderDir = (dirPath: string, depth: number) => {
    const entries = dirs.get(dirPath);
    if (entries === undefined) {
      return pendingDirs.has(dirPath) ? (
        <p
          key={`${dirPath}:loading`}
          className="text-2xs text-secondary py-1"
          style={{ paddingLeft: `${depth * 14 + 8}px` }}
        >
          {t('app.loading')}
        </p>
      ) : null;
    }
    if (entries.length === 0 && dirPath !== '') {
      return (
        <p
          key={`${dirPath}:empty`}
          className="text-2xs text-secondary py-1"
          style={{ paddingLeft: `${depth * 14 + 8}px` }}
        >
          {t('fileExplorer.emptyDir')}
        </p>
      );
    }
    return entries.map((entry) => {
      const path = dirPath ? `${dirPath}/${entry.name}` : entry.name;
      if (entry.type === 'directory') {
        const isOpen = expanded.has(path);
        return (
          <div key={path}>
            <button
              type="button"
              onClick={() => toggleDir(path)}
              aria-expanded={isOpen}
              className="flex items-center gap-1.5 w-full text-left py-1 px-2 rounded-md text-xs text-primary hover:bg-card-hover transition-colors"
              style={{ paddingLeft: `${depth * 14 + 8}px` }}
            >
              {isOpen ? (
                <ChevronDown size={13} className="shrink-0 text-secondary" aria-hidden />
              ) : (
                <ChevronRight size={13} className="shrink-0 text-secondary" aria-hidden />
              )}
              <Folder size={13} className="shrink-0 text-secondary" aria-hidden />
              <span className="truncate">{entry.name}</span>
            </button>
            {isOpen && renderDir(path, depth + 1)}
          </div>
        );
      }
      const op = touchedByPath.get(normalizePath(path));
      return (
        <button
          key={path}
          type="button"
          onClick={() => onSelectFile(path)}
          className="flex items-center gap-1.5 w-full text-left py-1 px-2 rounded-md text-xs text-primary hover:bg-card-hover transition-colors"
          style={{ paddingLeft: `${depth * 14 + 8}px` }}
        >
          <span className="w-[13px] shrink-0" aria-hidden />
          <File size={13} className="shrink-0 text-secondary" aria-hidden />
          <span className="truncate">{entry.name}</span>
          {op && (
            <span className="ml-auto shrink-0 text-2xs text-secondary font-mono">
              {t(OP_LABEL[op])}
            </span>
          )}
        </button>
      );
    });
  };

  return (
    <div className="flex flex-col h-full min-h-0 overflow-y-auto" data-testid="files-view-tree">
      {touched.length > 0 && (
        <div className="shrink-0 border-b border-border p-2">
          <button
            type="button"
            onClick={() => setTouchedOpen((prev) => !prev)}
            aria-expanded={touchedOpen}
            data-testid="panel-touched-group"
            className="flex items-center gap-1.5 w-full text-left py-1 px-1 rounded-md text-xs font-medium text-secondary hover:text-primary hover:bg-card-hover transition-colors"
          >
            {touchedOpen ? (
              <ChevronDown size={13} className="shrink-0" aria-hidden />
            ) : (
              <ChevronRight size={13} className="shrink-0" aria-hidden />
            )}
            {touched.length === 1
              ? t('panel.files.touched.one')
              : t('panel.files.touched.other', { n: touched.length })}
          </button>
          {touchedOpen &&
            touched.map((entry) => (
              <button
                key={entry.path}
                type="button"
                onClick={() => onSelectFile(entry.path)}
                className="flex items-center gap-1.5 w-full text-left py-1 px-2 rounded-md text-xs text-primary hover:bg-card-hover transition-colors"
              >
                <File size={13} className="shrink-0 text-secondary" aria-hidden />
                <span className="truncate">{entry.path}</span>
                <span className="ml-auto shrink-0 text-2xs text-secondary font-mono">
                  {t(OP_LABEL[entry.op])}
                </span>
              </button>
            ))}
        </div>
      )}
      <div className="flex-1 min-h-0 p-2">{renderDir('', 0)}</div>
    </div>
  );
}

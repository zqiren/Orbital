// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useCallback, useRef } from 'react';
import { ChevronRight, Folder, FolderPlus, Home, Monitor, FileText, Download, Clock, Loader2 } from 'lucide-react';
import { api, ApiError } from '../config';
import { useT, translate } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';

interface BrowseEntry {
  name: string;
  path: string;
  has_children: boolean;
}

interface BrowseResponse {
  path: string;
  parent: string | null;
  display_name: string;
  entries: BrowseEntry[];
}

interface FolderInfo {
  path: string;
  display_name: string;
  accessible: boolean;
  access_note: string | null;
}

interface FoldersResponse {
  status: string;
  folders: FolderInfo[];
}

interface MkdirResponse {
  path: string;
}

export interface FolderBrowserPanelProps {
  /**
   * Called on a definitive folder choice: clicking a leaf folder (no
   * children), clicking "Use this folder", or creating a new folder (which
   * navigates into it and selects it in one step).
   */
  onSelect: (path: string) => void;
  /** Shorter panel for embedding inline in a modal (default: full height). */
  compact?: boolean;
}

const SHORTCUT_ICONS: Record<string, typeof Home> = {
  Home: Home,
  Desktop: Monitor,
  Documents: FileText,
  Downloads: Download,
};

/**
 * The workspace browsing UI (backlog #25): shortcuts + breadcrumb + entries +
 * manual path input + "New folder", with no modal chrome of its own so it can
 * be embedded inline — today by CreateProject's workspace picker, its only
 * consumer. The standalone-dialog wrapper it was extracted from was deleted in
 * backlog #26 once that inline embed became the sole call site.
 */
export default function FolderBrowserPanel({ onSelect, compact = false }: FolderBrowserPanelProps) {
  const t = useT();
  const { locale } = useLocale();
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [shortcuts, setShortcuts] = useState<FolderInfo[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualInput, setManualInput] = useState('');
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [newFolderError, setNewFolderError] = useState<string | null>(null);
  const [creatingFolder, setCreatingFolder] = useState(false);
  const newFolderInputRef = useRef<HTMLInputElement>(null);

  const browse = useCallback(async (path?: string) => {
    setLoading(true);
    setError(null);
    try {
      const query = path ? `?path=${encodeURIComponent(path)}` : '';
      const data = await api<BrowseResponse>(`/api/v2/platform/browse${query}`);
      setCurrentPath(data.path);
      setEntries(data.entries);
      setManualInput(data.path);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail);
      } else {
        // Use `translate(locale, ...)` rather than the `t` from useT(): useT()
        // returns a fresh closure every render, so depending on it here would
        // give `browse` a new identity every render. The mount effect below
        // depends on `browse`, so an unstable `browse` re-fires the effect on
        // every render → an infinite browse/re-render loop (folder list
        // "blinks" and entries are never clickable). `locale` is a stable
        // string, so `browse` only changes when the language actually changes.
        setError(translate(locale, 'folderPicker.browseError'));
      }
    } finally {
      setLoading(false);
    }
  }, [locale]);

  useEffect(() => {
    browse();
    // Load shortcuts from platform folders
    api<FoldersResponse>('/api/v2/platform/folders')
      .then((data) => setShortcuts(data.folders))
      .catch(() => setShortcuts([]));
    // Load recent paths from localStorage
    try {
      const stored = localStorage.getItem('folderPicker_recent');
      if (stored) setRecentPaths(JSON.parse(stored));
    } catch { /* ignore */ }
  }, [browse]);

  function saveRecent(path: string) {
    const updated = [path, ...recentPaths.filter((p) => p !== path)].slice(0, 5);
    setRecentPaths(updated);
    try {
      localStorage.setItem('folderPicker_recent', JSON.stringify(updated));
    } catch { /* ignore */ }
  }

  function select(path: string) {
    saveRecent(path);
    onSelect(path);
  }

  function handleFolderClick(entry: BrowseEntry) {
    if (entry.has_children) {
      browse(entry.path);
    } else {
      // Select directly if no children to drill into
      select(entry.path);
    }
  }

  function handleManualNavigate() {
    if (manualInput.trim()) {
      browse(manualInput.trim());
    }
  }

  function openNewFolder() {
    setNewFolderOpen(true);
    setNewFolderName('');
    setNewFolderError(null);
    requestAnimationFrame(() => newFolderInputRef.current?.focus());
  }

  function cancelNewFolder() {
    setNewFolderOpen(false);
    setNewFolderName('');
    setNewFolderError(null);
  }

  async function handleCreateFolder() {
    const name = newFolderName.trim();
    if (!name || creatingFolder) return;
    setCreatingFolder(true);
    setNewFolderError(null);
    try {
      const result = await api<MkdirResponse>('/api/v2/platform/mkdir', {
        method: 'POST',
        body: JSON.stringify({ parent: currentPath, name }),
      });
      setNewFolderOpen(false);
      setNewFolderName('');
      await browse(result.path);
      select(result.path);
    } catch (err) {
      setNewFolderError(err instanceof ApiError ? err.detail : t('folderPicker.newFolder.error'));
    } finally {
      setCreatingFolder(false);
    }
  }

  // Build breadcrumb segments from currentPath relative to home
  function getBreadcrumbs(): { label: string; path: string }[] {
    if (!currentPath) return [];
    const parts = currentPath.replace(/\\/g, '/').split('/').filter(Boolean);
    const isWindows = /^[A-Za-z]:/.test(currentPath);
    const crumbs: { label: string; path: string }[] = [];

    // Add root entry for navigating to filesystem root / drive listing
    crumbs.push({ label: isWindows ? t('folderPicker.thisPc') : '/', path: '/' });

    for (let i = 0; i < parts.length; i++) {
      const seg = parts.slice(0, i + 1);
      let fullPath: string;
      if (isWindows) {
        fullPath = seg.join('\\');
        if (seg.length === 1) fullPath += '\\';
      } else {
        fullPath = '/' + seg.join('/');
      }
      crumbs.push({ label: parts[i], path: fullPath });
    }
    return crumbs;
  }

  const breadcrumbs = getBreadcrumbs();

  return (
    <div className={`flex flex-col ${compact ? 'h-64' : 'h-[360px]'}`}>
      {/* Shortcuts + entries, two-panel on desktop, stacked on mobile */}
      <div className="flex flex-1 min-h-0 max-md:flex-col">
        <div className="w-[160px] border-r border-border p-2.5 overflow-y-auto shrink-0 max-md:w-full max-md:border-r-0 max-md:border-b max-md:p-2 max-md:overflow-x-auto max-md:overflow-y-hidden max-md:flex max-md:gap-1.5 max-md:shrink-0">
          {shortcuts.map((folder) => {
            const Icon = SHORTCUT_ICONS[folder.display_name] || Folder;
            return (
              <button
                key={folder.path}
                type="button"
                onClick={() => browse(folder.path)}
                className={`w-full text-left flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-all duration-150 max-md:w-auto max-md:whitespace-nowrap max-md:min-h-[36px] ${
                  currentPath === folder.path
                    ? 'bg-accent/10 text-accent'
                    : 'text-primary hover:bg-card-hover'
                }`}
              >
                <Icon size={14} className="shrink-0" />
                <span className="truncate">{folder.display_name}</span>
              </button>
            );
          })}
          {/* Root / drive navigation */}
          <button
            type="button"
            onClick={() => browse('/')}
            className={`w-full text-left flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-all duration-150 max-md:w-auto max-md:whitespace-nowrap max-md:min-h-[36px] ${
              currentPath === '/'
                ? 'bg-accent/10 text-accent'
                : 'text-primary hover:bg-card-hover'
            }`}
          >
            <Monitor size={14} className="shrink-0" />
            <span className="truncate">{/^[A-Za-z]:/.test(currentPath) ? t('folderPicker.thisPc') : t('folderPicker.root')}</span>
          </button>
          {recentPaths.length > 0 && (
            <>
              <div className="border-t border-border my-2 max-md:hidden" />
              {recentPaths.map((rp) => {
                const label = rp.split(/[\\/]/).filter(Boolean).pop() || rp;
                return (
                  <button
                    key={rp}
                    type="button"
                    onClick={() => browse(rp)}
                    className="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-primary hover:bg-card-hover transition-all duration-150 max-md:hidden"
                  >
                    <Clock size={14} className="shrink-0 text-secondary" />
                    <span className="truncate">{label}</span>
                  </button>
                );
              })}
            </>
          )}
        </div>

        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          {/* Breadcrumb + New folder */}
          <div className="px-3 py-2 border-b border-border flex items-center justify-between gap-2 shrink-0">
            <div className="flex items-center gap-1 flex-wrap text-xs text-secondary min-w-0">
              {breadcrumbs.map((crumb, i) => (
                <span key={crumb.path} className="flex items-center gap-1">
                  {i > 0 && <ChevronRight size={10} className="text-secondary/50" />}
                  <button
                    type="button"
                    onClick={() => browse(crumb.path)}
                    className={`hover:text-accent transition-all duration-150 ${
                      i === breadcrumbs.length - 1 ? 'text-primary font-medium' : ''
                    }`}
                  >
                    {crumb.label}
                  </button>
                </span>
              ))}
            </div>
            <button
              type="button"
              onClick={openNewFolder}
              disabled={!currentPath || loading}
              className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:text-accent/80 transition-all duration-150 shrink-0 disabled:opacity-40"
            >
              <FolderPlus size={13} />
              {t('folderPicker.newFolder.button')}
            </button>
          </div>

          {newFolderOpen && (
            // A plain div, not a <form>: this panel is embedded inline inside
            // CreateProject's own <form> (backlog #25 review) — a nested
            // <form> is invalid HTML and its implicit-submit-on-Enter
            // behavior is browser-dependent (this app ships on
            // pywebview→WKWebView). Enter is handled explicitly below via
            // onKeyDown + preventDefault, which also stops it from bubbling
            // up to the outer form when embedded.
            <div className="px-3 py-2 border-b border-border flex items-center gap-2 shrink-0">
              <input
                ref={newFolderInputRef}
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') cancelNewFolder();
                  else if (e.key === 'Enter') { e.preventDefault(); handleCreateFolder(); }
                }}
                placeholder={t('folderPicker.newFolder.placeholder')}
                className="flex-1 text-xs bg-sidebar border border-border rounded-md px-2 py-1 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
              <button
                type="button"
                onClick={handleCreateFolder}
                disabled={!newFolderName.trim() || creatingFolder}
                className="inline-flex items-center gap-1 text-xs font-medium text-white bg-accent rounded-md px-2.5 py-1 hover:bg-accent/90 transition-all duration-150 disabled:opacity-40"
              >
                {creatingFolder && <Loader2 size={12} className="animate-spin" />}
                {t('folderPicker.newFolder.create')}
              </button>
              <button
                type="button"
                onClick={cancelNewFolder}
                className="text-xs text-secondary hover:text-primary transition-all duration-150"
              >
                {t('folderPicker.cancel')}
              </button>
            </div>
          )}
          {newFolderOpen && newFolderError && (
            <p className="text-xs text-error px-3 pb-2 -mt-1 shrink-0">{newFolderError}</p>
          )}

          {/* Entries list */}
          <div className="flex-1 overflow-y-auto px-2 py-1">
            {loading && (
              <div className="flex items-center justify-center py-8 text-secondary">
                <Loader2 size={16} className="animate-spin" />
              </div>
            )}
            {!loading && error && (
              <p className="text-xs text-error px-2 py-4">{error}</p>
            )}
            {!loading && !error && entries.length === 0 && (
              <p className="text-xs text-secondary px-2 py-4">{t('folderPicker.empty')}</p>
            )}
            {!loading && !error && entries.map((entry) => (
              <button
                key={entry.path}
                type="button"
                onClick={() => handleFolderClick(entry)}
                className="w-full text-left flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-primary hover:bg-card-hover transition-all duration-150 max-md:min-h-[44px]"
              >
                <Folder size={15} className="shrink-0 text-accent" />
                <span className="flex-1 truncate">{entry.name}</span>
                {entry.has_children && (
                  <ChevronRight size={14} className="shrink-0 text-secondary/50" />
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Manual path input + explicit "use this folder" (covers folders with
          children too). A plain div, not a <form> — see the new-folder note
          above; the same nested-form hazard applies here. */}
      <div className="border-t border-border px-3 py-2 shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={manualInput}
            onChange={(e) => setManualInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleManualNavigate(); } }}
            placeholder={t('folderPicker.manualPath.placeholder')}
            className="flex-1 text-xs font-mono bg-sidebar border border-border rounded-lg px-2.5 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
          />
          <button
            type="button"
            onClick={() => select(currentPath)}
            disabled={!currentPath}
            className="text-xs font-medium text-white bg-accent rounded-lg px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-40 shrink-0"
          >
            {t('folderPicker.useThisFolder')}
          </button>
        </div>
      </div>
    </div>
  );
}

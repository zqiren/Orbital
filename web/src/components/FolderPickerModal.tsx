// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useCallback } from 'react';
import { X, ChevronRight, Folder, Home, Monitor, FileText, Download, Clock, Loader2 } from 'lucide-react';
import { api, ApiError } from '../config';

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

export interface FolderPickerModalProps {
  open: boolean;
  onSelect: (path: string) => void;
  onClose: () => void;
}

const SHORTCUT_ICONS: Record<string, typeof Home> = {
  Home: Home,
  Desktop: Monitor,
  Documents: FileText,
  Downloads: Download,
};

export default function FolderPickerModal({ open, onSelect, onClose }: FolderPickerModalProps) {
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [shortcuts, setShortcuts] = useState<FolderInfo[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualInput, setManualInput] = useState('');

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
        setError('Failed to browse directory');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
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
  }, [open, browse]);

  function saveRecent(path: string) {
    const updated = [path, ...recentPaths.filter((p) => p !== path)].slice(0, 5);
    setRecentPaths(updated);
    try {
      localStorage.setItem('folderPicker_recent', JSON.stringify(updated));
    } catch { /* ignore */ }
  }

  function handleSelect() {
    if (currentPath) {
      saveRecent(currentPath);
      onSelect(currentPath);
    }
  }

  function handleFolderClick(entry: BrowseEntry) {
    if (entry.has_children) {
      browse(entry.path);
    } else {
      // Select directly if no children to drill into
      saveRecent(entry.path);
      onSelect(entry.path);
    }
  }

  function handleManualNavigate(e: React.FormEvent) {
    e.preventDefault();
    if (manualInput.trim()) {
      browse(manualInput.trim());
    }
  }

  // Build breadcrumb segments from currentPath relative to home
  function getBreadcrumbs(): { label: string; path: string }[] {
    if (!currentPath) return [];
    const parts = currentPath.replace(/\\/g, '/').split('/').filter(Boolean);
    const isWindows = /^[A-Za-z]:/.test(currentPath);
    const crumbs: { label: string; path: string }[] = [];

    // Add root entry for navigating to filesystem root / drive listing
    crumbs.push({ label: isWindows ? 'This PC' : '/', path: '/' });

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

  if (!open) return null;

  const breadcrumbs = getBreadcrumbs();

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-background rounded-xl shadow-xl border border-border w-full max-w-[680px] max-h-[80vh] flex flex-col mx-4 animate-slide-up max-md:max-h-[95vh] max-md:max-w-full max-md:mx-2">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-primary">Choose Workspace Folder</h2>
          <button
            onClick={onClose}
            className="text-secondary hover:text-primary transition-all duration-150 p-1 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body: two-panel on desktop, stacked on mobile */}
        <div className="flex flex-1 min-h-0 max-md:flex-col">
          {/* Shortcuts panel */}
          <div className="w-[180px] border-r border-border p-3 overflow-y-auto shrink-0 max-md:w-full max-md:border-r-0 max-md:border-b max-md:p-2 max-md:overflow-x-auto max-md:overflow-y-hidden max-md:flex max-md:gap-1.5 max-md:shrink-0">
            <p className="text-[11px] font-medium text-secondary uppercase tracking-wider mb-2 max-md:hidden">
              Shortcuts
            </p>
            {shortcuts.map((folder) => {
              const Icon = SHORTCUT_ICONS[folder.display_name] || Folder;
              return (
                <button
                  key={folder.path}
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
              onClick={() => browse('/')}
              className={`w-full text-left flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-all duration-150 max-md:w-auto max-md:whitespace-nowrap max-md:min-h-[36px] ${
                currentPath === '/'
                  ? 'bg-accent/10 text-accent'
                  : 'text-primary hover:bg-card-hover'
              }`}
            >
              <Monitor size={14} className="shrink-0" />
              <span className="truncate">{/^[A-Za-z]:/.test(currentPath) ? 'This PC' : 'Root /'}</span>
            </button>
            {recentPaths.length > 0 && (
              <>
                <div className="border-t border-border my-2 max-md:hidden" />
                <p className="text-[11px] font-medium text-secondary uppercase tracking-wider mb-2 max-md:hidden">
                  Recent
                </p>
                {recentPaths.map((rp) => {
                  const label = rp.split(/[\\/]/).filter(Boolean).pop() || rp;
                  return (
                    <button
                      key={rp}
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

          {/* Main panel */}
          <div className="flex-1 flex flex-col min-h-0 min-w-0">
            {/* Breadcrumb */}
            <div className="px-4 py-2 border-b border-border text-xs text-secondary flex items-center gap-1 flex-wrap shrink-0">
              {breadcrumbs.map((crumb, i) => (
                <span key={crumb.path} className="flex items-center gap-1">
                  {i > 0 && <ChevronRight size={10} className="text-secondary/50" />}
                  <button
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
                <p className="text-xs text-secondary px-2 py-4">No folders here.</p>
              )}
              {!loading && !error && entries.map((entry) => (
                <button
                  key={entry.path}
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

        {/* Footer */}
        <div className="border-t border-border px-5 py-3 shrink-0 space-y-2">
          {/* Manual path input */}
          <form onSubmit={handleManualNavigate} className="flex gap-2">
            <input
              type="text"
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="Type a path and press Enter..."
              className="flex-1 text-xs font-mono bg-sidebar border border-border rounded-lg px-3 py-1.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
          </form>

          {/* Action buttons */}
          <div className="flex items-center justify-end gap-2 max-md:flex-col-reverse">
            <button
              onClick={onClose}
              className="text-sm text-secondary hover:text-primary transition-all duration-150 px-4 py-2 max-md:w-full max-md:min-h-[44px]"
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!currentPath}
              className="bg-accent text-white text-sm font-medium rounded-lg px-5 py-2 hover:bg-accent/90 transition-all duration-150 disabled:opacity-40 max-md:w-full max-md:min-h-[44px]"
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

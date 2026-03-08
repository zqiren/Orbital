// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import {
  Folder,
  File,
  ChevronRight,
  ChevronDown,
  ArrowLeft,
  Plus,
  Download,
  Copy,
  Check,
} from 'lucide-react';
import { api, BASE_URL, isRelayMode } from '../config';
import type { FileEntry, DirectoryListing, FileContent } from '../types';

const MAX_UPLOAD_SIZE = 10 * 1024 * 1024; // 10MB

interface FileExplorerProps {
  projectId: string;
}

interface TreeNode {
  entry: FileEntry;
  path: string;
  children: TreeNode[] | null; // null = not loaded yet (directories), also null for files
  expanded: boolean;
  loading: boolean;
}

export default function FileExplorer({ projectId }: FileExplorerProps) {
  const [rootNodes, setRootNodes] = useState<TreeNode[]>([]);
  const [rootLoading, setRootLoading] = useState(true);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<FileContent | null>(null);
  const [contentLoading, setContentLoading] = useState(false);
  const [mobileShowPreview, setMobileShowPreview] = useState(false);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDirectory = useCallback(
    async (path: string): Promise<FileEntry[]> => {
      const query = path ? `?path=${encodeURIComponent(path)}` : '';
      const data = await api<DirectoryListing>(
        `/api/v2/projects/${encodeURIComponent(projectId)}/files${query}`,
      );
      return sortEntries(data.entries);
    },
    [projectId],
  );

  const fetchFileContent = useCallback(
    async (path: string) => {
      setContentLoading(true);
      try {
        const data = await api<FileContent>(
          `/api/v2/projects/${encodeURIComponent(projectId)}/files/content?path=${encodeURIComponent(path)}`,
        );
        setFileContent(data);
      } catch {
        setFileContent(null);
      } finally {
        setContentLoading(false);
      }
    },
    [projectId],
  );

  const refreshRoot = useCallback(() => {
    setRootLoading(true);
    fetchDirectory('').then((entries) => {
      setRootNodes(entries.map((entry) => toTreeNode(entry, '')));
      setRootLoading(false);
    }).catch(() => {
      setRootLoading(false);
    });
  }, [fetchDirectory]);

  useEffect(() => {
    let cancelled = false;
    setRootLoading(true);
    setRootNodes([]);
    setSelectedPath(null);
    setFileContent(null);
    setMobileShowPreview(false);
    setUploadError(null);

    fetchDirectory('').then((entries) => {
      if (cancelled) return;
      setRootNodes(entries.map((entry) => toTreeNode(entry, '')));
      setRootLoading(false);
    }).catch(() => {
      if (!cancelled) setRootLoading(false);
    });

    return () => { cancelled = true; };
  }, [projectId, fetchDirectory]);

  const toggleDirectory = useCallback(
    async (path: string) => {
      let needsFetch = false;

      // flushSync required: the updater sets needsFetch inside setState,
      // which React 19 may defer. Without flushSync, needsFetch can remain
      // false and the fetch is skipped, causing an infinite loading spinner.
      flushSync(() => {
        setRootNodes((prev) => {
          const target = findNode(prev, path);
          if (!target) return prev;

          if (target.expanded) {
            return updateNode(prev, path, { expanded: false });
          }
          if (target.children !== null) {
            return updateNode(prev, path, { expanded: true });
          }
          // Guard against duplicate fetches (e.g. rapid double-click):
          // if already loading, don't start another fetch.
          if (target.loading) {
            return prev;
          }
          needsFetch = true;
          return updateNode(prev, path, { loading: true, expanded: true });
        });
      });

      if (needsFetch) {
        try {
          const entries = await fetchDirectory(path);
          const children = entries.map((entry) => toTreeNode(entry, path));
          setRootNodes((prev) => setNodeChildren(prev, path, children));
        } catch {
          setRootNodes((prev) => setNodeChildren(prev, path, []));
        }
      }
    },
    [fetchDirectory],
  );

  const handleFileClick = useCallback(
    (path: string) => {
      setSelectedPath(path);
      setMobileShowPreview(true);
      fetchFileContent(path);
    },
    [fetchFileContent],
  );

  const handleBack = useCallback(() => {
    setMobileShowPreview(false);
  }, []);

  const handleUploadClick = useCallback(() => {
    setUploadError(null);
    fileInputRef.current?.click();
  }, []);

  const handleFileSelected = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      // Reset input so the same file can be selected again
      e.target.value = '';

      if (file.size > MAX_UPLOAD_SIZE) {
        setUploadError(`File too large (${formatSize(file.size)}). Maximum is 10 MB.`);
        return;
      }

      setUploading(true);
      setUploadError(null);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const baseUrl = isRelayMode ? window.location.origin : BASE_URL;
        const url = `${baseUrl}/api/v2/projects/${encodeURIComponent(projectId)}/files/upload`;
        const headers: Record<string, string> = {};

        if (isRelayMode) {
          try {
            const token = localStorage.getItem('relay_jwt');
            if (token) {
              headers['Authorization'] = `Bearer ${token}`;
            }
          } catch {
            // ignore
          }
        }

        const response = await fetch(url, { method: 'POST', body: formData, headers });
        if (!response.ok) {
          const body = await response.text();
          let detail: string;
          try {
            const parsed = JSON.parse(body);
            detail = parsed.detail ?? body;
          } catch {
            detail = body;
          }
          throw new Error(detail || 'Upload failed');
        }

        refreshRoot();
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [projectId, refreshRoot],
  );

  return (
    <div className="flex h-full overflow-hidden">
      {/* File Tree - left panel on desktop, full screen on mobile when no preview */}
      <div
        className={`
          w-full md:w-[260px] md:min-w-[260px] md:block
          border-r border-border flex flex-col
          ${mobileShowPreview ? 'hidden md:flex' : 'flex'}
        `}
      >
        <div className="flex-1 overflow-y-auto p-3">
          {rootLoading ? (
            <TreeSkeleton />
          ) : rootNodes.length === 0 ? (
            <p className="text-sm text-secondary px-2 py-1">No files found</p>
          ) : (
            rootNodes.map((node) => (
              <TreeItem
                key={node.path}
                node={node}
                depth={0}
                selectedPath={selectedPath}
                onToggle={toggleDirectory}
                onFileClick={handleFileClick}
              />
            ))
          )}
        </div>

        {/* Upload error message */}
        {uploadError && (
          <div className="mx-3 mb-3 px-3 py-2 bg-error/10 border border-error/30 rounded-md">
            <p className="text-xs text-error">{uploadError}</p>
          </div>
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileSelected}
        />

        {/* Upload button - fixed footer, always visible at bottom */}
        <div className="shrink-0 px-3 pb-3 max-md:pb-[max(12px,env(safe-area-inset-bottom,12px))] flex justify-end md:block">
          <button
            onClick={handleUploadClick}
            disabled={uploading}
            className="
              flex items-center justify-center gap-1.5
              bg-accent text-white rounded-full md:rounded-lg
              w-10 h-10 md:w-auto md:h-auto md:px-3 md:py-2 md:w-full
              text-sm font-medium
              hover:bg-accent/90 transition-all duration-150
              disabled:opacity-50 disabled:cursor-not-allowed
              shadow-lg md:shadow-none
            "
          >
            {uploading ? (
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Plus size={18} />
            )}
            <span className="hidden md:inline">{uploading ? 'Uploading...' : 'Upload File'}</span>
          </button>
        </div>
      </div>

      {/* File Preview - right panel on desktop, full screen on mobile when viewing */}
      <div
        className={`
          flex-1 overflow-y-auto min-w-0
          ${mobileShowPreview ? 'block' : 'hidden md:block'}
        `}
      >
        {mobileShowPreview && (
          <button
            onClick={handleBack}
            className="md:hidden flex items-center gap-1.5 px-4 py-3 text-sm text-secondary hover:text-primary transition-colors border-b border-border w-full"
          >
            <ArrowLeft size={16} />
            Back to files
          </button>
        )}
        <FilePreview
          fileContent={fileContent}
          loading={contentLoading}
          selectedPath={selectedPath}
        />
      </div>
    </div>
  );
}

interface TreeItemProps {
  node: TreeNode;
  depth: number;
  selectedPath: string | null;
  onToggle: (path: string) => void;
  onFileClick: (path: string) => void;
}

function TreeItem({ node, depth, selectedPath, onToggle, onFileClick }: TreeItemProps) {
  const isDirectory = node.entry.type === 'directory';
  const isSelected = node.path === selectedPath;

  const handleClick = () => {
    if (isDirectory) {
      onToggle(node.path);
    } else {
      onFileClick(node.path);
    }
  };

  return (
    <>
      <button
        onClick={handleClick}
        className={`
          flex items-center gap-1.5 w-full text-left py-1 px-2 rounded-md
          text-sm transition-colors duration-150
          ${isSelected ? 'bg-card-hover text-primary' : 'text-primary hover:bg-card-hover'}
        `}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
      >
        {isDirectory ? (
          <>
            {node.loading ? (
              <span className="w-4 h-4 shrink-0 flex items-center justify-center">
                <span className="w-3 h-3 border border-secondary border-t-transparent rounded-full animate-spin" />
              </span>
            ) : node.expanded ? (
              <ChevronDown size={16} className="shrink-0 text-secondary" />
            ) : (
              <ChevronRight size={16} className="shrink-0 text-secondary" />
            )}
            <Folder size={16} className="shrink-0 text-secondary" />
          </>
        ) : (
          <>
            <span className="w-4 shrink-0" />
            <File size={16} className="shrink-0 text-secondary" />
          </>
        )}
        <span className="truncate">{node.entry.name}</span>
      </button>

      {isDirectory && node.expanded && node.children && (
        <div className="transition-all duration-150">
          {node.children.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onToggle={onToggle}
              onFileClick={onFileClick}
            />
          ))}
          {node.children.length === 0 && !node.loading && (
            <p
              className="text-xs text-secondary py-1"
              style={{ paddingLeft: `${(depth + 1) * 16 + 8}px` }}
            >
              Empty directory
            </p>
          )}
        </div>
      )}
    </>
  );
}

interface FilePreviewProps {
  fileContent: FileContent | null;
  loading: boolean;
  selectedPath: string | null;
}

function FilePreview({ fileContent, loading, selectedPath }: FilePreviewProps) {
  const [copied, setCopied] = useState(false);
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
        <p className="text-sm text-secondary">Select a file to preview</p>
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
        <p className="text-sm text-secondary">Unable to load file</p>
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
                Download
              </button>
            )}
          </div>
        </div>
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
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {fileContent.truncated && (
        <div className="px-4 py-2 bg-sidebar border-b border-border">
          <p className="text-xs text-secondary">
            File truncated -- showing first 500KB
          </p>
        </div>
      )}
      <div className="flex-1 overflow-auto">
        <pre className="font-mono text-sm text-primary bg-sidebar p-4 whitespace-pre-wrap break-words min-h-full">
          {fileContent.content}
        </pre>
      </div>
    </div>
  );
}

const SKELETON_WIDTHS = [100, 72, 120, 88, 64, 108, 80, 96];

function TreeSkeleton() {
  return (
    <div className="space-y-1 px-2">
      {SKELETON_WIDTHS.map((w, i) => (
        <div key={i} className="flex items-center gap-2 py-1">
          <div className="w-4 h-4 bg-sidebar rounded animate-pulse" />
          <div
            className="h-4 bg-sidebar rounded animate-pulse"
            style={{ width: `${w}px` }}
          />
        </div>
      ))}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function toTreeNode(entry: FileEntry, parentPath: string): TreeNode {
  const path = parentPath ? `${parentPath}/${entry.name}` : entry.name;
  return {
    entry,
    path,
    children: null,
    expanded: false,
    loading: false,
  };
}

function sortEntries(entries: FileEntry[]): FileEntry[] {
  return [...entries].sort((a, b) => {
    // Pin agent_output directory to top
    if (a.name === 'agent_output' && a.type === 'directory') return -1;
    if (b.name === 'agent_output' && b.type === 'directory') return 1;
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

function findNode(nodes: TreeNode[], path: string): TreeNode | null {
  for (const node of nodes) {
    if (node.path === path) return node;
    if (node.children) {
      const found = findNode(node.children, path);
      if (found) return found;
    }
  }
  return null;
}

function updateNode(
  nodes: TreeNode[],
  path: string,
  updates: Partial<TreeNode>,
): TreeNode[] {
  return nodes.map((node) => {
    if (node.path === path) {
      return { ...node, ...updates };
    }
    if (node.children) {
      return { ...node, children: updateNode(node.children, path, updates) };
    }
    return node;
  });
}

function setNodeChildren(
  nodes: TreeNode[],
  path: string,
  children: TreeNode[],
): TreeNode[] {
  return nodes.map((node) => {
    if (node.path === path) {
      return { ...node, children, loading: false };
    }
    if (node.children) {
      return { ...node, children: setNodeChildren(node.children, path, children) };
    }
    return node;
  });
}

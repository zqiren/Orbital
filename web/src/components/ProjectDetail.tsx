// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { AgentRunStatus, FileContent, Project, Trigger } from '../types';
import type { Route } from '../route';
import StatusBadge from './StatusBadge';
import TriggerStrip from './TriggerStrip';
import SettingsIcon from './SettingsIcon';
import BudgetCorner from './BudgetCorner';
import FilePreviewDrawer, { OpenPathContext } from './FilePreviewDrawer';
import { useQueue } from '../hooks/useQueue';
import { useFiles } from '../hooks/useFiles';
import { fetchPathWithFallback } from '../utils/openPathWithFallback';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface ProjectDetailProps {
  project: Project;
  agentStatus: AgentRunStatus;
  statusSummary?: string;
  route: Extract<Route, { name: 'project' }>;
  setRoute: Dispatch<SetStateAction<Route>>;
  triggers?: Trigger[];
  onTriggerToggle?: (triggerId: string, enabled: boolean) => void;
  onTriggerDelete?: (triggerId: string) => void;
  /**
   * The global/default LLM model (from app settings). Used as the header's
   * model label only when the project has no per-project model pinned. NOT a
   * substitute for `agent_name` — a model id, not an identity.
   */
  globalDefaultModel?: string;
  children?: React.ReactNode;
}

// Workbench and Calendar are GLOBAL workspace surfaces — they aggregate
// across projects and deliberately have no per-project tab (user decision,
// 2026-07-24).
type TabKey = 'queue' | 'chat' | 'files';

const BASE_TABS: { key: TabKey; labelKey: StringKey }[] = [
  { key: 'chat', labelKey: 'projectDetail.tab.chat' },
  { key: 'queue', labelKey: 'projectDetail.tab.queue' },
  { key: 'files', labelKey: 'projectDetail.tab.files' },
];

export default function ProjectDetail({
  project,
  agentStatus,
  statusSummary,
  route,
  setRoute,
  triggers = [],
  onTriggerToggle,
  onTriggerDelete,
  globalDefaultModel,
  children,
}: ProjectDetailProps) {
  const t = useT();

  // The active tab indicator: when settings overlay is showing, no tab is highlighted
  const activeTab = route.settings ? null : route.tab;

  const tabs = BASE_TABS;

  // Tab count badges (queue only — Chat's session count lives in the sidebar).
  const { snapshot } = useQueue(project.project_id);
  const queueCount = snapshot?.items.filter(
    (item) => item.state === 'queued' || item.state === 'running',
  ).length ?? 0;

  // File preview drawer (spec 002). The open/close state lives on the route
  // (`previewPath`); the fetched content + lazy 404 probe live here. Opening is
  // PROBE-FIRST: fetch the content, then open the drawer only once it's in hand
  // (404/error just toasts, never opening) — so the panel never flashes blank
  // and a missing file never opens-then-closes.
  const { getFileContent, resolvePath, saveFileContent } = useFiles();
  const [previewContent, setPreviewContent] = useState<FileContent | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewToast, setPreviewToast] = useState<string | null>(null);
  // Latest requested path — guards against a slow fetch for an earlier click
  // resolving after a newer one (replace-on-click).
  const latestPreviewReqRef = useRef<string | null>(null);
  const previewPath = route.previewPath ?? null;

  const handleOpenPath = useCallback(
    async (path: string) => {
      latestPreviewReqRef.current = path;
      setPreviewLoading(true);
      // Probe-first: fetch BEFORE opening so the drawer never slides in blank.
      // On replace-on-click the already-open drawer keeps showing the prior
      // content until the new content is ready — no flash there either.
      // Agents abbreviate paths in chat ('drafts/x.md', bare 'DECISIONS.md'),
      // so a miss falls back to the resolve endpoint and transparently opens
      // a unique suffix match.
      const outcome = await fetchPathWithFallback(
        (p) => getFileContent(project.project_id, p),
        (p) => resolvePath(project.project_id, p),
        path,
      );
      // Drop the result if a newer open superseded this one.
      if (latestPreviewReqRef.current !== path) return;
      setPreviewLoading(false);
      if (outcome.status !== 'ok') {
        setPreviewToast(
          outcome.status === 'ambiguous'
            ? t('chat.path.ambiguous', { name: path })
            : t('chat.path.notFound'),
        );
        return;
      }
      const data = outcome.content;
      setPreviewContent(data);
      // Two-frame open (WKWebView "blink" fix). Render the content into the
      // drawer's OFF-SCREEN compositing layer first, then trigger the slide-in
      // one paint later — so WebKit can't composite the layer's stale/empty
      // backing store for the first animation frame (the blink you see in
      // pywebview but not Chromium). The double rAF guarantees a paint landed
      // between "content in" and "start sliding".
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          if (latestPreviewReqRef.current !== path) return;
          // Route to the RESOLVED path (not the clicked one) so save/deep-link
          // hit the real file when the fallback rewrote an abbreviated path.
          setRoute((prev) =>
            prev.name === 'project' ? { ...prev, previewPath: outcome.path } : prev,
          );
        }),
      );
    },
    [getFileContent, resolvePath, project.project_id, setRoute, t],
  );

  const handleClosePreview = useCallback(() => {
    latestPreviewReqRef.current = null;
    setRoute((prev) =>
      prev.name === 'project' ? { ...prev, previewPath: undefined } : prev,
    );
  }, [setRoute]);

  // Auto-dismiss the "file not found" toast.
  useEffect(() => {
    if (!previewToast) return;
    const id = setTimeout(() => setPreviewToast(null), 4000);
    return () => clearTimeout(id);
  }, [previewToast]);

  // Navigating to another tab/surface closes the preview drawer — it is a
  // chat overlay, not a persistent panel (clearing `previewPath` so it can't
  // linger over the Files tab or strand open after a settings round-trip).
  function handleTabChange(tab: TabKey) {
    setRoute({ ...route, tab, settings: false, previewPath: undefined });
  }

  function handleSettingsClick() {
    setRoute({ ...route, settings: true, previewPath: undefined });
  }

  // Budget corner → open settings scrolled to the Budget section.
  function handleOpenBudgetSettings() {
    setRoute({ ...route, settings: true, settingsAnchor: 'budget', previewPath: undefined });
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 px-6 pt-5 pb-4 max-md:px-4">
        {/* Left cluster keeps a readable minimum so a long budget pill on the
            right can never crush the title to 0 width (P3-J header collision). */}
        <div className="flex items-center gap-3 min-w-[40%] flex-1">
          <h1 className="font-mono text-[18px] font-semibold text-primary truncate min-w-0">{project.name}</h1>
          <StatusBadge status={agentStatus} />
        </div>
        <div className="flex items-center gap-3 min-w-0 shrink">
          {/* Budget corner (P3-G): converted window spend / warn / exhausted
              pill, next to the model name. Reads the same GET /cost as the
              settings meter; event-driven (no polling). Click → Budget settings.
              Replaces the dead P3-F $0.00 segment properly. */}
          <BudgetCorner
            project={project}
            pauseReason={snapshot?.pause_reason}
            onOpenBudgetSettings={handleOpenBudgetSettings}
          />
          {(() => {
            // Model label: the project's pinned model, else the global default.
            // NEVER agent_name (an identity, not a model).
            const modelName = project.model || globalDefaultModel || '';
            if (!modelName) return null;
            return (
              <span className="font-mono text-[11px] text-secondary">{modelName}</span>
            );
          })()}
          <SettingsIcon onClick={handleSettingsClick} />
        </div>
      </div>

      {/* Status summary line */}
      {statusSummary && (
        <div className="px-6 pb-2 max-md:px-4">
          <p className="text-xs text-secondary truncate">{statusSummary}</p>
        </div>
      )}

      {/* Trigger strip — between header and tab bar */}
      {triggers.length > 0 && onTriggerToggle && (
        <TriggerStrip triggers={triggers} onToggle={onTriggerToggle} onDelete={onTriggerDelete} />
      )}

      {/* Tab bar */}
      <div className="flex gap-1 px-6 border-b border-border max-md:px-4">
        {tabs.map((tab) => {
          // Chat shows no count badge — the session count lives in the
          // sidebar header. Only the queue badge (pending/running items) shows.
          const count = tab.key === 'queue' ? queueCount : 0;
          return (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={`text-[12.5px] font-medium px-3 py-2 -mb-px transition-all duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center gap-1.5 ${
                activeTab === tab.key
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-secondary hover:text-primary'
              }`}
            >
              {t(tab.labelKey)}
              {count > 0 && (
                <span className="text-[10.5px] font-mono text-secondary">
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab content. `relative` so the file-preview drawer (absolute) overlays
          only this content area, not the header/tab bar. The OpenPathContext
          hands the drawer-open handler down to the chat's MarkdownContent. */}
      <div className="flex-1 overflow-hidden min-h-0 relative">
        <OpenPathContext.Provider value={handleOpenPath}>
          {children}
        </OpenPathContext.Provider>
        <FilePreviewDrawer
          open={previewPath !== null}
          // Content-driven (NOT previewPath): the drawer renders its content
          // while still off-screen so the slide-in reveals an already-painted
          // layer. Also keeps content during the close slide-out (no empty
          // "select a file" flash on close either).
          selectedPath={previewContent?.path ?? null}
          fileContent={previewContent}
          loading={previewLoading}
          onClose={handleClosePreview}
          onSave={(path, content) => saveFileContent(project.project_id, path, content)}
        />
        {previewToast && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg bg-primary text-background text-[13px] shadow-lg max-w-[90%] text-center pointer-events-none">
            {previewToast}
          </div>
        )}
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Workbench reading pane (2026-11-04, glance option A): the selected
 * card's session, embedded. A right docked panel on desktop, a bottom sheet
 * on a phone. The header carries the session's name, 打开会话 (the full
 * session) and a close ×; the body is the real ChatView in its narrow
 * layout, composer included, live like the session's own page.
 *
 * Two differences from the session's page:
 *  - being on screen does not check the reply (`checksOnView={false}`): the
 *    pane is a glance, the card stays until ✓, and a turn that ends while the
 *    pane is open brings the card back to 等你;
 *  - a document the chat opens comes up OVER the pane with a back chevron
 *    (the phone's pattern), never as a second panel. Its full-width control
 *    covers the whole window except the 20px edge strip, as in a project.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, Maximize2, Minimize2, X } from 'lucide-react';
import type { AgentRunStatus, FileContent, Project } from '../../types';
import { useT } from '../../i18n/useT';
import { useFiles } from '../../hooks/useFiles';
import { useCoverWindow } from '../../hooks/useWindowCover';
import { fetchPathWithFallback } from '../../utils/openPathWithFallback';
import ChatView from '../ChatView';
import FilePreview from '../FilePreview';
import { OpenPathContext } from '../filePreviewContext';
import type { CardDot } from './feedModel';

export interface WorkbenchPaneProps {
  project: Project;
  sessionId: string;
  title: string;
  dot: CardDot;
  agentStatus: AgentRunStatus;
  statusTick?: number;
  agents: Array<{ slug: string; name: string }>;
  phone: boolean;
  onClose: () => void;
  /** 打开会话: the full session (also a session link inside the chat). */
  onOpenSession: (sessionId: string) => void;
  /** The user sent from the pane's composer; `result` is the inject response. */
  onUserSend: (result?: unknown) => void;
}

export default function WorkbenchPane({
  project, sessionId, title, dot, agentStatus, statusTick, agents, phone,
  onClose, onOpenSession, onUserSend,
}: WorkbenchPaneProps) {
  const t = useT();
  const projectId = project.project_id;
  const { getFileContent, resolvePath, saveFileContent } = useFiles();

  // The document over the pane, if any. Keyed by session, so selecting
  // another card never shows the previous one's document.
  const [doc, setDoc] = useState<{ session: string; path: string; content: FileContent } | null>(null);
  const shownDoc = doc && doc.session === sessionId ? doc : null;
  const [maximized, setMaximized] = useState(false);
  // A pinned project list collapses to the strip while the document covers
  // the window (App, useWindowCover).
  useCoverWindow(maximized && !!shownDoc);
  const [toast, setToast] = useState<string | null>(null);
  const latestRef = useRef<string | null>(null);

  const openPath = useCallback(
    async (path: string) => {
      latestRef.current = path;
      const outcome = await fetchPathWithFallback(
        (p) => getFileContent(projectId, p),
        (p) => resolvePath(projectId, p),
        path,
      );
      if (latestRef.current !== path) return;
      if (outcome.status !== 'ok') {
        setToast(outcome.status === 'ambiguous'
          ? t('chat.path.ambiguous', { name: path })
          : t('chat.path.notFound'));
        return;
      }
      setDoc({ session: sessionId, path: outcome.path, content: outcome.content });
    },
    [getFileContent, resolvePath, projectId, sessionId, t],
  );

  const closeDoc = useCallback(() => {
    latestRef.current = null;
    setMaximized(false);
    setDoc(null);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(id);
  }, [toast]);

  // Esc restores a document covering the window.
  useEffect(() => {
    if (!maximized) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented) setMaximized(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [maximized]);

  const maximizeButton = phone ? null : (
    <button
      type="button"
      data-testid="wb-pane-doc-maximize"
      onClick={() => setMaximized((m) => !m)}
      aria-label={maximized ? t('panel.restore') : t('panel.maximize')}
      title={maximized ? t('panel.restore') : t('panel.maximize')}
      className="flex items-center justify-center w-7 h-7 shrink-0 rounded-md text-secondary hover:text-primary hover:bg-card-hover transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
    >
      {maximized ? <Minimize2 size={14} aria-hidden /> : <Maximize2 size={14} aria-hidden />}
    </button>
  );

  const pane = (
    <aside
      className={phone ? 'wb-sheet' : 'wb-pane'}
      data-testid="wb-pane"
      data-variant={phone ? 'sheet' : 'dock'}
      aria-label={title}
    >
      {phone && <div className="wb-grab" aria-hidden="true" />}
      <div className="wb-ph">
        <span className={`wb-dot is-${dot}`} aria-hidden="true" />
        <span className="wb-pttl" data-testid="wb-pane-title" title={title}>{title}</span>
        <button
          type="button"
          className="wb-olink"
          data-testid="wb-pane-open"
          onClick={() => onOpenSession(sessionId)}
        >
          {t('workbench.card.open')}
          <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="wb-ab"
          data-testid="wb-pane-close"
          aria-label={t('workbench.pane.close')}
          title={t('workbench.pane.close')}
          onClick={onClose}
        >
          <X size={15} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>
      <div className="wb-pbody">
        <OpenPathContext.Provider value={openPath}>
          <ChatView
            key={`${projectId}:${sessionId}`}
            projectId={projectId}
            project={project}
            agentStatus={agentStatus}
            statusTick={statusTick}
            agents={agents}
            sessionId={sessionId}
            onOpenSession={onOpenSession}
            checksOnView={false}
            onUserSend={onUserSend}
          />
        </OpenPathContext.Provider>
      </div>
      {shownDoc && (
        <div
          className={`wb-pdoc${maximized ? ' panel-max-window' : ''}`}
          data-testid="wb-pane-doc"
          data-maximized={maximized ? 'true' : 'false'}
        >
          {/* A relative link inside the document opens through the same
              handler, replacing it. */}
          <OpenPathContext.Provider value={openPath}>
            <FilePreview
              fileContent={shownDoc.content}
              loading={false}
              selectedPath={shownDoc.path}
              onSave={(path, content) => saveFileContent(projectId, path, content)}
              panelHeader={{ onBack: closeDoc, backLabel: t('workbench.pane.back') }}
              headerExtras={maximizeButton}
              history={{ projectId }}
            />
          </OpenPathContext.Provider>
        </div>
      )}
      {toast && <div className="wb-ptoast" role="status">{toast}</div>}
    </aside>
  );

  if (!phone) return pane;
  return (
    <>
      <div className="wb-scrim" data-testid="wb-pane-scrim" onClick={onClose} aria-hidden="true" />
      {pane}
    </>
  );
}

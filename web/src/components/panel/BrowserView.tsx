// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.3/§5.6 — the Browser view: the agent's page, live and
 * interactive (canvas painted from screencast frames, input forwarded), with
 * the session's last screenshot as fallback when no browser is open.
 * CONTRACT FILE: props are final; workstream D implements.
 */
import { useT } from '../../i18n/useT';
import type { Annotation, AnnotationDraft } from '../../utils/annotations';

export interface BrowserViewProps {
  projectId: string;
  /** Stream only while true (view visible + panel expanded). */
  active: boolean;
  /** Fallback when no live page: the session's last screenshot (workspace-relative path) + title. */
  fallback: { path: string; title?: string } | null;
  annotating: boolean;
  annotations: Annotation[];
  onAddAnnotation: (draft: AnnotationDraft) => void;
}

export default function BrowserView(_props: BrowserViewProps) {
  const t = useT();
  return <div className="text-xs text-secondary p-3">{t('panel.browser.empty')}</div>; // TODO(spec 078 workstream D)
}

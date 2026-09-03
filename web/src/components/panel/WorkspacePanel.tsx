// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5 — the panel body: a Files | Browser switch, an Annotate button,
 * and the selected view. No tabs beyond the two views, no other chrome.
 * CONTRACT FILE: props are final; workstream B implements.
 */
import type { ReactNode } from 'react';
import type { PanelView } from '../../utils/panelSelectors';

export interface WorkspacePanelProps {
  view: PanelView;
  onViewChange: (v: PanelView) => void;
  annotating: boolean;
  onToggleAnnotate: () => void;
  browser: ReactNode;
  files: ReactNode;
}

export default function WorkspacePanel(_props: WorkspacePanelProps) {
  return null; // TODO(spec 078 workstream B)
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.1/§5.2 — per-session panel state + lifecycle (expand at runtime,
 * collapse at rest). CONTRACT FILE: shape is final; workstream B implements.
 */
import type { PanelView } from '../utils/panelSelectors';

export interface PanelState {
  expanded: boolean;
  userCollapsedThisRun: boolean;
  view: PanelView;
  file: string | null;
}
export interface PanelStateApi extends PanelState {
  expand: () => void;
  collapse: () => void; // user intent: stays collapsed for this run
  setView: (v: PanelView) => void;
  setFile: (path: string | null) => void;
}

export function usePanelState(_projectId: string, _sessionId: string | undefined): PanelStateApi {
  // TODO(spec 078 workstream B)
  return {
    expanded: false, userCollapsedThisRun: false, view: 'files', file: null,
    expand() {}, collapse() {}, setView() {}, setFile() {},
  };
}

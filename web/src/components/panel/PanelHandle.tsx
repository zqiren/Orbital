// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.1 — the collapsed state: a 20px edge handle at the right of the
 * chat column with a chevron and a small dot while the agent is working.
 * CONTRACT FILE: props are final; workstream B implements.
 */
export interface PanelHandleProps {
  working: boolean;
  onExpand: () => void;
}

export default function PanelHandle(_props: PanelHandleProps) {
  return null; // TODO(spec 078 workstream B)
}

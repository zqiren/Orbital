// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.4 — box-drawing overlay used over the live browser canvas and
 * over image previews. Absolutely positioned to fill its parent (the parent
 * must be position:relative). When `active`, a drag draws a box and a note
 * field appears; Enter calls onAdd. Existing boxes render numbered pins.
 * CONTRACT FILE: props are final; workstream E implements.
 */
import type { AnnotationBox } from '../../utils/annotations';

export interface AnnotateOverlayProps {
  active: boolean;
  boxes: { n: number; box: AnnotationBox }[];
  onAdd: (box: AnnotationBox, note: string) => void;
  onRemove?: (n: number) => void;
}

export default function AnnotateOverlay(_props: AnnotateOverlayProps) {
  return null; // TODO(spec 078 workstream E)
}

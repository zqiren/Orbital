// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.3 (D10) — Files view: tree state (workspace tree, touched files
 * badged + "Touched this session" group) ⇄ preview state (FilePreview in
 * place with ‹ Files and "Open in Files"). Never both at once.
 * CONTRACT FILE: props are final; workstream B implements.
 */
import type { AnnotationDraft } from '../../utils/annotations';
import type { TouchedFile } from '../../utils/panelSelectors';

export interface FilesViewProps {
  projectId: string;
  touched: TouchedFile[];
  /** Current file in preview state, or null for the tree state. */
  file: string | null;
  onSelectFile: (path: string | null) => void;
  onOpenInFiles: (path: string) => void;
  annotating: boolean;
  onAddAnnotation: (draft: AnnotationDraft) => void;
  onSave?: (path: string, content: string) => Promise<boolean>;
}

export default function FilesView(_props: FilesViewProps) {
  return null; // TODO(spec 078 workstream B)
}

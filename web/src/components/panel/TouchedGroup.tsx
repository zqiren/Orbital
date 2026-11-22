// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 D10 — the "Touched this session" group: every path the manager's
 * own read / write / edit tools touched in this session.
 *
 * NOT RENDERED since spec 187: the 最近修改 / Recent group replaced it at the
 * top of the Files view (owner decision §10.1). Kept whole so restoring it
 * is one line in FilesView. `touched` itself still drives the tree's
 * auto-expanded folders and op badges.
 */
import { useState } from 'react';
import { ChevronDown, ChevronRight, File } from 'lucide-react';
import type { TouchedFile, TouchedOp } from '../../utils/panelSelectors';
import { useT } from '../../i18n/useT';
import type { StringKey } from '../../i18n/strings';

// The same map FilesView uses for its tree badges (kept local: a component
// file exports components only, for fast refresh).
const OP_LABEL: Record<TouchedOp, StringKey> = {
  read: 'panel.files.read',
  edited: 'panel.files.edited',
  written: 'panel.files.written',
};

export default function TouchedGroup({
  touched,
  onSelectFile,
}: {
  touched: TouchedFile[];
  onSelectFile: (path: string) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(true);
  if (touched.length === 0) return null;
  return (
    <div className="shrink-0 border-b border-border p-2">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        data-testid="panel-touched-group"
        className="flex items-center gap-1.5 w-full text-left py-1 px-1 rounded-md text-caption font-medium text-secondary hover:text-primary hover:bg-card-hover transition-colors"
      >
        {open ? (
          <ChevronDown size={13} className="shrink-0" aria-hidden />
        ) : (
          <ChevronRight size={13} className="shrink-0" aria-hidden />
        )}
        {touched.length === 1
          ? t('panel.files.touched.one')
          : t('panel.files.touched.other', { n: touched.length })}
      </button>
      {open &&
        touched.map((entry) => (
          <button
            key={entry.path}
            type="button"
            onClick={() => onSelectFile(entry.path)}
            className="flex items-center gap-1.5 w-full text-left py-1 px-2 rounded-md text-caption text-primary hover:bg-card-hover transition-colors"
          >
            <File size={13} className="shrink-0 text-secondary" aria-hidden />
            <span className="truncate">{entry.path}</span>
            <span className="ml-auto shrink-0 text-micro text-secondary font-mono">
              {t(OP_LABEL[entry.op])}
            </span>
          </button>
        ))}
    </div>
  );
}

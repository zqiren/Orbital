// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { X } from 'lucide-react';
import { useT } from '../i18n/useT';
import FolderBrowserPanel from './FolderBrowserPanel';

export interface FolderPickerModalProps {
  open: boolean;
  onSelect: (path: string) => void;
  onClose: () => void;
}

export default function FolderPickerModal({ open, onSelect, onClose }: FolderPickerModalProps) {
  const t = useT();

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-background rounded-xl shadow-xl border border-border w-full max-w-[680px] max-h-[80vh] flex flex-col mx-4 animate-slide-up max-md:max-h-[95vh] max-md:max-w-full max-md:mx-2">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-primary">{t('folderPicker.title')}</h2>
          <button
            onClick={onClose}
            className="text-secondary hover:text-primary transition-all duration-150 p-1 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
          >
            <X size={16} />
          </button>
        </div>

        {/* Browser panel (shortcuts + breadcrumb + entries + manual path + new folder) */}
        <div className="flex-1 min-h-0">
          <FolderBrowserPanel onSelect={onSelect} />
        </div>

        {/* Footer */}
        <div className="border-t border-border px-5 py-3 shrink-0 flex items-center justify-end">
          <button
            onClick={onClose}
            className="text-sm text-secondary hover:text-primary transition-all duration-150 px-4 py-2 max-md:w-full max-md:min-h-[44px]"
          >
            {t('folderPicker.cancel')}
          </button>
        </div>
      </div>
    </div>
  );
}

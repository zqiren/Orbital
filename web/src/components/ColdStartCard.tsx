// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

interface ColdStartCardProps {
  folderName: string;
  onScan: () => void;
  onSkip: () => void;
  busy?: boolean;
}

/**
 * First-session consent card for imported (non-empty) workspaces. Renders in
 * the empty chat view; spends no tokens until the user clicks Scan. Scan starts
 * the cold-start scan session; Skip dismisses the card for this view.
 */
export function ColdStartCard({ folderName, onScan, onSkip, busy }: ColdStartCardProps) {
  return (
    <div className="mb-3 rounded-lg border border-border bg-white overflow-hidden">
      <div className="bg-accent/10 px-4 py-2 border-b border-border">
        <span className="text-accent font-semibold text-sm">Scan this workspace?</span>
      </div>
      <div className="px-4 py-3.5 space-y-3.5">
        <p className="text-sm text-secondary leading-relaxed">
          <span className="font-mono text-[13px] text-primary">{folderName}</span> already has files in it.
          I can scan it to understand the project and propose a summary for you to review —
          nothing is saved until you approve.
        </p>
        <div className="flex gap-2">
          <button
            onClick={onScan}
            disabled={busy}
            className="bg-accent text-white text-sm font-medium rounded-lg px-4 py-1.5 hover:opacity-90 transition-opacity duration-150 max-md:min-h-[44px] disabled:opacity-50"
          >
            {busy ? 'Scanning…' : 'Scan'}
          </button>
          <button
            onClick={onSkip}
            disabled={busy}
            className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-1.5 hover:bg-card-hover transition-colors duration-150 max-md:min-h-[44px] disabled:opacity-50"
          >
            Skip
          </button>
        </div>
      </div>
    </div>
  );
}

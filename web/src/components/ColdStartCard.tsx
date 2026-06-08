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
    <div className="border border-accent/30 border-l-[3px] border-l-accent rounded-lg bg-background">
      <div className="px-4 py-1.5 border-b border-accent/20 text-accent/80 text-xs uppercase tracking-[0.6px] font-semibold">
        Scan this workspace?
      </div>
      <div className="px-4 py-3 space-y-3 text-sm">
        <p className="text-secondary">
          <span className="font-medium text-primary">{folderName}</span> has existing files.
          I can scan it to understand the project and propose a summary for you to confirm —
          nothing is written until you approve.
        </p>
        <div className="flex gap-2">
          <button
            onClick={onScan}
            disabled={busy}
            className="bg-primary text-white px-4 py-1.5 rounded max-md:min-h-[44px] disabled:opacity-50"
          >
            {busy ? 'Scanning…' : 'Scan'}
          </button>
          <button
            onClick={onSkip}
            disabled={busy}
            className="bg-background text-secondary border border-border px-4 py-1.5 rounded max-md:min-h-[44px]"
          >
            Skip
          </button>
        </div>
      </div>
    </div>
  );
}

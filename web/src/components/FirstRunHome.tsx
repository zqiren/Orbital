// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * FirstRunHome — the content pane while the user has no project of their own
 * (see `isFirstRun`). Replaces the bare "Select a project from the sidebar."
 * sentence, which gave a brand-new user nothing to click: the root cause of
 * the first-journey confusion was that nobody told them a project is the
 * thing to create.
 *
 * One idea, one action. Quick Tasks is deliberately not mentioned here.
 *
 * The illustration is the message in picture form — files gathering in a
 * folder, set on the actual logo (`OrbitalMark`, the locked geometry — not a
 * lookalike) — so it carries no copy of its own and needs no translation. Pure CSS/SVG; the only motion is a one-shot entrance
 * (`animate-rise-in`), which index.css disables under reduced motion.
 */

import { BarChart3, FileText, Folder, Image as ImageIcon, Plus } from 'lucide-react';
import { useT } from '../i18n/useT';
import OrbitalMark from './OrbitalMark';

interface FirstRunHomeProps {
  onNewProject: () => void;
}

function FileCard({
  className,
  children,
}: {
  className: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`absolute flex h-[92px] w-[72px] flex-col gap-2 rounded-xl border border-border bg-card p-3 shadow-[0_4px_14px_-6px_rgb(0_0_0/0.22)] ${className}`}
    >
      <span className="text-secondary">{children}</span>
      <span className="h-1 w-full rounded-full bg-border" />
      <span className="h-1 w-4/5 rounded-full bg-border" />
      <span className="h-1 w-3/5 rounded-full bg-border" />
    </div>
  );
}

/** The real logo as the backdrop, with the folder group centred on its disc
 * and drawn ON TOP of every part of the mark (disc, stem, leaf, bud) — the
 * folder is the subject, the logo is the stage. The mark is square with the
 * disc at its exact centre (that is how the logo is constructed), so centring
 * the group on the box centres it on the disc. */
function Illustration() {
  return (
    <div aria-hidden="true" className="relative h-[400px] w-[400px] max-md:scale-[0.8] max-md:-my-10">
      <OrbitalMark className="absolute inset-0 h-full w-full" />

      {/* Work gathering in the folder — centred on the disc (200, 200). */}
      <FileCard className="left-[104px] top-[96px] -rotate-[14deg]">
        <FileText size={16} />
      </FileCard>
      <FileCard className="left-[164px] top-[76px] rotate-[2deg]">
        <BarChart3 size={16} />
      </FileCard>
      <FileCard className="left-[224px] top-[96px] rotate-[13deg]">
        <ImageIcon size={16} />
      </FileCard>

      <div className="absolute left-1/2 top-[140px] flex h-[128px] w-[164px] -translate-x-1/2 items-center justify-center rounded-[20px] border border-border bg-card shadow-[0_16px_34px_-12px_rgb(0_0_0/0.42)]">
        <Folder size={58} strokeWidth={1.5} className="text-accent" fill="currentColor" fillOpacity={0.12} />
      </div>
    </div>
  );
}

export default function FirstRunHome({ onNewProject }: FirstRunHomeProps) {
  const t = useT();
  return (
    <div
      data-testid="first-run-home"
      className="relative flex flex-1 min-h-0 items-center justify-center overflow-hidden px-6"
    >
      {/* A warm wash behind the hero so the pane is not one flat grey. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(52% 44% at 50% 38%, rgb(244 163 140 / 0.10), transparent 72%)',
        }}
      />
      <div className="relative flex max-w-md flex-col items-center text-center animate-rise-in">
        <Illustration />
        <h1 className="-mt-4 text-2xl font-semibold text-primary">
          {t('firstRun.title')}
        </h1>
        <p className="mt-3 max-w-sm text-sm leading-relaxed text-secondary [text-wrap:balance]">
          {t('firstRun.body')}
        </p>
        <button
          type="button"
          onClick={onNewProject}
          className="mt-7 inline-flex items-center gap-2 rounded-xl bg-accent px-6 py-3 text-sm font-medium text-white shadow-[0_6px_16px_-6px_rgb(83_154_248/0.7)] transition-all duration-150 hover:bg-accent/90 hover:shadow-[0_8px_20px_-6px_rgb(83_154_248/0.8)] max-md:min-h-[44px]"
        >
          <Plus size={16} aria-hidden="true" />
          {t('firstRun.cta')}
        </button>
      </div>
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';
import { useTheme, type ThemePreference } from '../hooks/useTheme';

/**
 * Appearance picker for Global Settings: System / Light / Dark.
 *
 * Content only — the section heading is the enclosing SettingsSection's
 * `title` (`appearance.title`), the same split TelemetrySettings uses.
 *
 * Each option is a real radio input inside its label, so arrow keys move the
 * choice and a screen reader announces a radio group; the card is just the
 * label's skin. The choice applies immediately and is stored per device
 * (`useTheme`), so there is nothing to Save.
 *
 * The thumbnails are deliberately drawn in FIXED colours rather than theme
 * tokens: each one depicts a theme, so it must look the same whichever theme
 * is currently on. They are the two token ladders from index.css in miniature.
 */

interface Palette {
  nav: string;
  sidebar: string;
  card: string;
  line: string;
  ink: string;
  composer: string;
}

const LIGHT: Palette = {
  nav: '#E8EBF0',
  sidebar: '#F0F3F7',
  card: '#FFFFFF',
  line: '#E1E5EA',
  ink: '#C9CED6',
  composer: '#FFFFFF',
};

const DARK: Palette = {
  nav: '#111317',
  sidebar: '#16191D',
  card: '#1E2126',
  line: '#2D3138',
  ink: '#454A53',
  composer: '#262A30',
};

const ACCENT = '#539AF8';

function Thumb({ p }: { p: Palette }) {
  return (
    <div className="flex h-full w-full" style={{ background: p.card }}>
      <div className="h-full w-[13%]" style={{ background: p.nav }} />
      <div className="flex h-full w-[26%] flex-col gap-[3px] p-[4px]" style={{ background: p.sidebar }}>
        <div className="h-[5px] w-full rounded-[2px]" style={{ background: p.line }} />
        <div className="h-[5px] w-3/4 rounded-[2px]" style={{ background: p.line }} />
        <div className="h-[5px] w-5/6 rounded-[2px]" style={{ background: p.line }} />
      </div>
      <div className="flex h-full flex-1 flex-col gap-[4px] p-[6px]">
        <div className="h-[4px] w-4/5 rounded-full" style={{ background: p.ink }} />
        <div className="h-[4px] w-3/5 rounded-full" style={{ background: p.ink }} />
        <div className="ml-auto h-[9px] w-2/5 rounded-[3px]" style={{ background: ACCENT, opacity: 0.85 }} />
        <div
          className="mt-auto h-[11px] w-full rounded-[3px] border"
          style={{ background: p.composer, borderColor: p.line }}
        />
      </div>
    </div>
  );
}

function SystemThumb() {
  return (
    <div className="relative h-full w-full">
      <Thumb p={LIGHT} />
      <div className="absolute inset-0" style={{ clipPath: 'inset(0 0 0 50%)' }}>
        <Thumb p={DARK} />
      </div>
    </div>
  );
}

const OPTIONS: { value: ThemePreference; labelKey: StringKey }[] = [
  { value: 'system', labelKey: 'appearance.option.system' },
  { value: 'light', labelKey: 'appearance.option.light' },
  { value: 'dark', labelKey: 'appearance.option.dark' },
];

export default function AppearanceSection() {
  const t = useT();
  const { preference, setPreference } = useTheme();

  return (
    <fieldset data-testid="appearance-section">
      <legend className="sr-only">{t('appearance.title')}</legend>
      <div className="grid max-w-[480px] grid-cols-3 gap-3 max-md:gap-2">
        {OPTIONS.map(({ value, labelKey }) => {
          const selected = preference === value;
          return (
            <label
              key={value}
              data-testid={`appearance-option-${value}`}
              className={`flex cursor-pointer flex-col gap-2 rounded-xl border-2 p-1.5 transition-colors duration-150 has-[input:focus-visible]:ring-2 has-[input:focus-visible]:ring-accent/50 ${
                selected ? 'border-primary' : 'border-border hover:border-secondary/40'
              }`}
            >
              <input
                type="radio"
                name="orbital-theme"
                value={value}
                checked={selected}
                onChange={() => setPreference(value)}
                className="sr-only"
              />
              <div
                aria-hidden="true"
                className="h-[64px] w-full overflow-hidden rounded-lg border border-border"
              >
                {value === 'system' ? <SystemThumb /> : <Thumb p={value === 'dark' ? DARK : LIGHT} />}
              </div>
              <span
                className={`px-0.5 pb-0.5 text-body ${
                  selected ? 'font-medium text-primary' : 'text-secondary'
                }`}
              >
                {t(labelKey)}
              </span>
            </label>
          );
        })}
      </div>
      <p className="mt-2.5 text-caption text-secondary">{t('appearance.hint')}</p>
    </fieldset>
  );
}

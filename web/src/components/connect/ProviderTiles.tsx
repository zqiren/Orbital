// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The logo tiles for the popular providers, then "More", which opens a
 * search over every provider in the catalog. A narrow panel (4 columns)
 * shows the first seven tiles and lets "More" stand in for a selected 8th or
 * 9th, so the selection is always visible.
 */
import { useEffect, useRef } from 'react';
import { Check, Search } from 'lucide-react';
import type { CatalogProvider } from '../../types';
import { useT } from '../../i18n/useT';
import ProviderLogo from './ProviderLogo';
import { CUSTOM, matchesProvider, NARROW_TILE_COUNT, shortName, TILE_ORDER } from './providerMeta';

interface ProviderTilesProps {
  catalog: CatalogProvider[];
  selected: string;
  moreOpen: boolean;
  query: string;
  onPick: (id: string) => void;
  onToggleMore: () => void;
  onQuery: (q: string) => void;
}

const tileBase =
  'surface-float surface-float-nested !rounded-[12px] flex h-[72px] @md:h-[76px] flex-col items-center justify-center gap-2 px-0.5 @md:px-1 min-w-0 focus-visible:outline-2 focus-visible:outline-accent';

export default function ProviderTiles({
  catalog,
  selected,
  moreOpen,
  query,
  onPick,
  onToggleMore,
  onQuery,
}: ProviderTilesProps) {
  const t = useT();
  const byId = new Map(catalog.map((p) => [p.id, p]));
  const tiles = TILE_ORDER.filter((id) => byId.has(id));
  const index = tiles.indexOf(selected as (typeof TILE_ORDER)[number]);
  const inTiles = index >= 0;
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (moreOpen) searchRef.current?.focus();
  }, [moreOpen]);

  const customLabel = t('llm.provider.custom');
  const nameOf = (p: CatalogProvider) => (p.id === CUSTOM ? customLabel : p.display_name);

  function moreTile(showSelected: boolean, variant: 'wide' | 'narrow') {
    const sel = showSelected ? byId.get(selected) : undefined;
    const on = !!sel || moreOpen;
    return (
      <button
        type="button"
        key={`more-${variant}`}
        data-testid={variant === 'wide' ? 'connect-more' : 'connect-more-narrow'}
        aria-expanded={moreOpen}
        onClick={onToggleMore}
        className={`${tileBase} ${variant === 'wide' ? 'hidden @md:flex' : 'flex @md:hidden'} ${
          on ? 'ring-[1.5px] ring-accent' : ''
        }`}
      >
        {sel ? (
          <ProviderLogo id={sel.id} name={sel.display_name} size={24} />
        ) : (
          <Search aria-hidden="true" className="h-[22px] w-[22px] text-muted" strokeWidth={1.6} />
        )}
        <span
          className={`max-w-full truncate text-micro @md:text-caption ${
            on ? 'font-medium text-primary' : 'text-secondary'
          }`}
        >
          {sel ? (sel.id === CUSTOM ? t('connect.customShort') : shortName(sel, sel.id)) : t('connect.more')}
        </span>
      </button>
    );
  }

  const hits = catalog.filter((p) => matchesProvider(p.id, nameOf(p), query));
  // Tiles first (their order), then the rest by name, Custom always last.
  const ranked = [
    ...tiles.map((id) => hits.find((p) => p.id === id)).filter((p): p is CatalogProvider => !!p),
    ...hits
      .filter((p) => !tiles.includes(p.id as never) && p.id !== CUSTOM)
      .sort((a, b) => a.display_name.localeCompare(b.display_name)),
  ];
  const custom = catalog.find((p) => p.id === CUSTOM);

  return (
    <div>
      <div className="grid grid-cols-4 gap-2 @md:grid-cols-5" role="group" aria-label={t('llm.field.provider')}>
        {tiles.map((id, i) => {
          const p = byId.get(id)!;
          const on = selected === id;
          return (
            <button
              type="button"
              key={id}
              data-testid={`connect-tile-${id}`}
              aria-pressed={on}
              onClick={() => onPick(id)}
              className={`${tileBase} ${i >= NARROW_TILE_COUNT ? 'hidden @md:flex' : ''} ${
                on ? 'ring-[1.5px] ring-accent' : ''
              }`}
            >
              <ProviderLogo id={id} name={p.display_name} size={24} />
              <span
                className={`max-w-full truncate text-micro @md:text-caption ${
                  on ? 'font-medium text-primary' : 'text-secondary'
                }`}
              >
                {shortName(p, id)}
              </span>
            </button>
          );
        })}
        {moreTile(!!selected && !inTiles, 'wide')}
        {moreTile(!!selected && (!inTiles || index >= NARROW_TILE_COUNT), 'narrow')}
      </div>

      {moreOpen && (
        <div className="mt-2 surface-float-static surface-float-nested !rounded-[12px]" data-testid="connect-more-panel">
          <div className="m-1 mb-0.5 flex h-9 items-center gap-2 rounded-[9px] bg-sidebar px-2.5 text-muted">
            <Search aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.6} />
            <input
              ref={searchRef}
              type="text"
              value={query}
              data-testid="connect-more-search"
              aria-label={t('llm.provider.search')}
              placeholder={t('llm.provider.search')}
              onChange={(e) => onQuery(e.target.value)}
              className="min-w-0 flex-1 bg-transparent text-body text-primary outline-none placeholder:text-muted"
            />
          </div>
          <div className="max-h-[280px] overflow-y-auto p-1">
            {ranked.length === 0 && (
              <p className="px-2 py-4 text-center text-body text-muted">{t('connect.noProvider')}</p>
            )}
            {[...ranked, ...(custom ? [custom] : [])].map((p, i) => (
              <div key={p.id}>
                {p.id === CUSTOM && i > 0 && <div className="mx-2 my-1 h-px bg-border" />}
                <button
                  type="button"
                  data-testid={`connect-pick-${p.id}`}
                  onClick={() => onPick(p.id)}
                  className="flex h-[34px] w-full items-center gap-2.5 rounded-[8px] px-2 text-left text-body text-primary hover:bg-card-hover"
                >
                  <ProviderLogo id={p.id} name={p.display_name} size={18} />
                  <span className={`min-w-0 flex-1 truncate ${selected === p.id ? 'font-medium' : ''}`}>
                    {nameOf(p)}
                  </span>
                  {selected === p.id && <Check aria-hidden="true" className="h-3.5 w-3.5 text-accent" />}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The single 模型 row under the key/account row, in the same frame. It opens
 * the list INSIDE the card (the card grows; nothing floats out of a wizard
 * or dialog). A list of up to MODEL_SEARCH_THRESHOLD rows shows them all with
 * no search; a longer one gets a search and a fixed-height scroll area. A
 * provider with no list at all gets a plain text field.
 */
import { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, ChevronUp, Search } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { needsSearch } from './providerMeta';

interface ModelRowProps {
  models: string[];
  /** Display names by id, where the provider gives them (account catalogs). */
  labels?: Record<string, string>;
  value: string;
  open: boolean;
  onToggle: () => void;
  onChange: (model: string) => void;
  /** Ids in monospace (key providers); account model names in the UI font. */
  mono?: boolean;
}

/** The open list. Its own component so the search resets each time it opens. */
function ModelList({ models, labels, value, onChange, font }: {
  models: string[];
  labels: Record<string, string>;
  value: string;
  onChange: (model: string) => void;
  font: string;
}) {
  const t = useT();
  const [query, setQuery] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);
  const big = needsSearch(models.length);
  useEffect(() => {
    if (big) searchRef.current?.focus({ preventScroll: true });
  }, [big]);

  const label = (m: string) => labels[m] || m;
  const q = query.trim().toLowerCase();
  const list = big && q ? models.filter((m) => `${m} ${labels[m] ?? ''}`.toLowerCase().includes(q)) : models;
  const typed = big && query.trim() && !models.includes(query.trim()) ? query.trim() : '';

  return (
    <div data-testid="connect-model-list">
      <div className="mx-3 h-px bg-border" />
      {big && (
        <div className="mx-2 mb-0.5 mt-2 flex h-9 items-center gap-2 rounded-[9px] bg-card px-2.5 text-muted">
          <Search aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.6} />
          <input
            ref={searchRef}
            type="text"
            value={query}
            data-testid="connect-model-search"
            aria-label={t('connect.model.search')}
            placeholder={t('connect.model.search')}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.nativeEvent.isComposing && typed) {
                e.preventDefault();
                onChange(typed);
              }
            }}
            spellCheck={false}
            className="min-w-0 flex-1 bg-transparent text-body text-primary outline-none placeholder:text-muted"
          />
          <span className="shrink-0 text-caption text-muted">{list.length}</span>
        </div>
      )}
      <div
        role="listbox"
        aria-label={t('llm.field.model')}
        className={`overflow-y-auto px-1 pb-1.5 pt-1 ${big ? 'h-[248px]' : ''}`}
      >
        {list.map((m) => (
          <button
            type="button"
            role="option"
            aria-selected={m === value}
            key={m}
            data-testid={`connect-model-option-${m}`}
            onClick={() => onChange(m)}
            className="flex h-[34px] w-full items-center gap-2.5 rounded-[8px] px-2 text-left hover:bg-card-hover"
          >
            <span className={`min-w-0 flex-1 truncate text-primary ${font} ${m === value ? 'font-medium' : ''}`}>
              {label(m)}
            </span>
            {m === value && <Check aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-accent" />}
          </button>
        ))}
        {typed && (
          <button
            type="button"
            data-testid="connect-model-use-typed"
            onClick={() => onChange(typed)}
            className="flex h-[34px] w-full items-center gap-2.5 rounded-[8px] px-2 text-left text-secondary hover:bg-card-hover"
          >
            <span className="min-w-0 flex-1 truncate text-body">{t('llm.model.useCustom', { value: typed })}</span>
          </button>
        )}
        {list.length === 0 && !typed && (
          <p className="px-2 py-4 text-center text-body text-muted">{t('connect.model.none')}</p>
        )}
      </div>
    </div>
  );
}

export default function ModelRow({ models, labels = {}, value, open, onToggle, onChange, mono = true }: ModelRowProps) {
  const t = useT();
  const font = mono ? 'font-mono text-caption' : 'text-body';

  if (models.length === 0) {
    // No list from this provider: the model is typed.
    return (
      <label className="flex h-[42px] items-center gap-2 px-3" data-testid="connect-model-row">
        <span className="shrink-0 text-body text-secondary">{t('llm.field.model')}</span>
        <input
          type="text"
          value={value}
          data-testid="connect-model-input"
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          autoComplete="off"
          className="min-w-0 flex-1 bg-transparent font-mono text-caption text-primary outline-none placeholder:text-muted"
        />
      </label>
    );
  }

  return (
    <div data-testid="connect-model-row">
      <button
        type="button"
        data-testid="connect-model-toggle"
        aria-expanded={open}
        onClick={onToggle}
        className="flex h-[42px] w-full items-center gap-2 rounded-[11px] px-3 text-left"
      >
        <span className="shrink-0 text-body text-secondary">{t('llm.field.model')}</span>
        <span className={`min-w-0 flex-1 truncate font-medium text-primary ${font}`} data-testid="connect-model-value">
          {labels[value] || value}
        </span>
        {open ? (
          <ChevronUp aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-muted" />
        ) : (
          <ChevronDown aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-muted" />
        )}
      </button>
      {open && <ModelList models={models} labels={labels} value={value} onChange={onChange} font={font} />}
    </div>
  );
}

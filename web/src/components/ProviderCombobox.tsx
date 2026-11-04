// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ProviderCombobox — the wizard's searchable provider picker (spec 138; it
 * replaced the chip row behind LLMProviderSettings' wizard-only
 * `providerPicker="cards"`). Type to filter the registry by display name or
 * key; Custom / Self-Hosted is always the last option, so a provider that is
 * not listed still has somewhere to go.
 *
 * ARIA 1.2 combobox: the input owns focus, the listbox is referenced by
 * aria-controls and the active option by aria-activedescendant. ArrowUp/Down
 * move, Enter picks, Escape closes and restores the selected name; options
 * select on mousedown so the input never blurs first.
 */

import { useId, useMemo, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { useT } from '../i18n/useT';

export interface ProviderOption {
  key: string;
  label: string;
}

interface ProviderComboboxProps {
  options: ProviderOption[];
  /** Always offered, after the filtered options. */
  custom: ProviderOption;
  value: string;
  onChange: (key: string) => void;
  label: string;
}

export default function ProviderCombobox({ options, custom, value, onChange, label }: ProviderComboboxProps) {
  const t = useT();
  const id = useId();
  const listId = `${id}-list`;
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState<string | null>(null);
  const [active, setActive] = useState(0);

  const selectedLabel = [...options, custom].find((o) => o.key === value)?.label ?? '';
  const matches = useMemo(() => {
    const q = (query ?? '').trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => o.label.toLowerCase().includes(q) || o.key.toLowerCase().includes(q));
  }, [options, query]);
  const visible = [...matches, custom];
  const activeIndex = Math.min(active, visible.length - 1);

  const close = () => {
    setOpen(false);
    setQuery(null);
  };
  const pick = (key: string) => {
    onChange(key);
    close();
  };
  const openList = () => {
    if (open) return;
    setOpen(true);
    setActive(Math.max(0, visible.findIndex((o) => o.key === value)));
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) { openList(); return; }
      setActive(Math.min(activeIndex + 1, visible.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (!open) { openList(); return; }
      setActive(Math.max(activeIndex - 1, 0));
    } else if (e.key === 'Enter') {
      if (!open) return;
      e.preventDefault();
      pick(visible[activeIndex].key);
    } else if (e.key === 'Escape') {
      if (!open) return;
      e.preventDefault();
      e.stopPropagation();
      close();
    }
  };

  return (
    <div className="relative">
      <input
        type="text"
        role="combobox"
        aria-label={label}
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open ? `${id}-opt-${activeIndex}` : undefined}
        autoComplete="off"
        spellCheck={false}
        data-testid="provider-combobox"
        value={query ?? selectedLabel}
        placeholder={t('llm.provider.search')}
        onFocus={(e) => {
          openList();
          e.currentTarget.select();
        }}
        onClick={openList}
        onBlur={close}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
        className="w-full text-sm bg-sidebar border border-border rounded-lg pl-3 pr-8 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
      />
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-secondary"
      />
      {open && (
        <div className="absolute left-0 right-0 top-full z-20 mt-1 overflow-hidden surface-float-pop">
          {matches.length === 0 && (
            <p className="px-3 pt-2 pb-1 text-xs text-secondary" data-testid="provider-combobox-empty">
              {t('llm.provider.noMatch')}
            </p>
          )}
          <ul id={listId} role="listbox" aria-label={label} className="max-h-60 overflow-y-auto py-1">
            {visible.map((o, i) => (
              <li
                key={o.key}
                id={`${id}-opt-${i}`}
                role="option"
                aria-selected={o.key === value}
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(o.key);
                }}
                onMouseEnter={() => setActive(i)}
                className={`cursor-pointer px-3 py-1.5 text-sm max-md:min-h-[44px] max-md:flex max-md:items-center ${
                  i === activeIndex ? 'bg-accent/10 text-primary' : 'text-primary'
                } ${o.key === value ? 'font-medium' : ''} ${o.key === custom.key ? 'border-t border-border' : ''}`}
              >
                {o.label}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

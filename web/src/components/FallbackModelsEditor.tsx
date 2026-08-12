// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useId, useState } from 'react';
import { ChevronDown, ChevronRight, Plus, X } from 'lucide-react';
import type { FallbackModelEntry, ProviderRegistry } from '../types';
import { useT } from '../i18n/useT';

interface FallbackModelsEditorProps {
  models: FallbackModelEntry[];
  onChange: (models: FallbackModelEntry[]) => void;
  providers: ProviderRegistry;
}

const EMPTY_ENTRY: FallbackModelEntry = {
  provider: 'openrouter',
  model: '',
  sdk: 'openai',
};

export default function FallbackModelsEditor({
  models,
  onChange,
  providers,
}: FallbackModelsEditorProps) {
  const [expanded, setExpanded] = useState(models.length > 0);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState<FallbackModelEntry>({ ...EMPTY_ENTRY });
  const t = useT();
  // Unique per mounted editor — global and project settings can both be on
  // screen, and two <datalist> elements sharing an id would cross-wire.
  const modelListId = `${useId()}fallback-model-options`;
  const suggestedModels = providers[draft.provider]?.suggested_models ?? [];

  function handleAdd() {
    if (!draft.model.trim()) return;
    const entry: FallbackModelEntry = {
      provider: draft.provider,
      model: draft.model.trim(),
      sdk: draft.sdk,
    };
    if (draft.api_key?.trim()) entry.api_key = draft.api_key.trim();
    if (draft.base_url?.trim()) entry.base_url = draft.base_url.trim();
    onChange([...models, entry]);
    setDraft({ ...EMPTY_ENTRY });
    setAdding(false);
  }

  function handleRemove(idx: number) {
    onChange(models.filter((_, i) => i !== idx));
  }

  function handleProviderChange(key: string) {
    const info = providers[key];
    setDraft({
      ...draft,
      provider: key,
      sdk: info?.sdk || 'openai',
    });
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm font-medium text-primary hover:text-accent transition-all duration-150 w-full text-left mb-2"
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4" />
        ) : (
          <ChevronRight className="w-4 h-4" />
        )}
        <span>{t('fallback.heading')}</span>
        {!expanded && models.length > 0 && (
          <span className="text-secondary font-normal ml-1">
            {t('fallback.configuredCount', { n: models.length })}
          </span>
        )}
      </button>

      {expanded && (
        <div className="space-y-3 ml-6">
          <p className="text-xs text-secondary">
            {t('fallback.intro')}
          </p>

          {/* Existing entries */}
          {models.map((entry, idx) => {
            const displayProvider =
              providers[entry.provider]?.display_name || entry.provider;
            return (
              <div
                key={idx}
                className="flex items-center justify-between bg-sidebar border border-border rounded-lg px-3 py-2"
              >
                <div className="min-w-0 mr-2">
                  <span className="text-sm font-medium text-primary block truncate">
                    {entry.model}
                  </span>
                  <span className="text-xs text-secondary block truncate">
                    {displayProvider}
                    {entry.api_key ? ` ${t('fallback.customKey')}` : ''}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => handleRemove(idx)}
                  className="shrink-0 text-secondary hover:text-error transition-colors p-1 max-md:min-w-[44px] max-md:min-h-[44px] flex items-center justify-center"
                  title={t('fallback.remove')}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            );
          })}

          {/* Add form */}
          {adding ? (
            <div className="border border-border rounded-lg p-3 space-y-3">
              {/* Provider */}
              <div>
                <label className="block text-xs font-medium text-primary mb-1">
                  {t('llm.field.provider')}
                </label>
                <select
                  value={draft.provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
                >
                  {Object.entries(providers).map(([key, info]) => (
                    <option key={key} value={key}>
                      {info.display_name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Model */}
              <div>
                <label className="block text-xs font-medium text-primary mb-1">
                  {t('llm.field.model')}
                </label>
                <input
                  type="text"
                  list={modelListId}
                  value={draft.model}
                  onChange={(e) =>
                    setDraft({ ...draft, model: e.target.value })
                  }
                  placeholder={t('fallback.model.placeholder')}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                />
                {/* Curated per-provider list; the input stays free-text so an
                    unlisted or newer model ID is still enterable. */}
                <datalist id={modelListId}>
                  {suggestedModels.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>

              {/* API Key (optional) */}
              <div>
                <label className="block text-xs font-medium text-primary mb-1">
                  {t('llm.field.apiKey')}{' '}
                  <span className="text-secondary font-normal">
                    {t('fallback.apiKey.optional')}
                  </span>
                </label>
                <input
                  type="password"
                  value={draft.api_key || ''}
                  onChange={(e) =>
                    setDraft({ ...draft, api_key: e.target.value })
                  }
                  placeholder={t('fallback.apiKey.placeholder')}
                  className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                />
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleAdd}
                  disabled={!draft.model.trim()}
                  className="text-sm font-medium text-white bg-accent rounded-lg px-4 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50"
                >
                  {t('fallback.add')}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAdding(false);
                    setDraft({ ...EMPTY_ENTRY });
                  }}
                  className="text-sm font-medium text-secondary border border-border rounded-lg px-4 py-1.5 hover:text-primary transition-all duration-150"
                >
                  {t('fallback.cancel')}
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setAdding(true)}
              className="flex items-center gap-1.5 text-sm text-secondary hover:text-accent transition-all duration-150"
            >
              <Plus className="w-4 h-4" />
              {t('fallback.addCta')}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

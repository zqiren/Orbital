// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The fallback chain, as an ordered list of credential cards (spec 082 §3.9).
 *
 * Each rung used to be a provider + model + optional key + optional endpoint
 * typed by hand, which is the same "four fields the daemon has to pair back
 * together" problem the cards replace. A rung is now one card reference, so a
 * fallback can only ever be a setup that already exists and has been tested.
 *
 * A rung with no card yet emits `{card_id: null}`; the daemon's chain builder
 * skips null entries, so an unfinished row is inert rather than an error.
 */
import { useState } from 'react';
import { ChevronDown, ChevronRight, Plus, X } from 'lucide-react';
import type { CredentialCard, FallbackModelEntry } from '../types';
import { useT } from '../i18n/useT';
import CardPicker from './CardPicker';

interface FallbackModelsEditorProps {
  models: FallbackModelEntry[];
  onChange: (models: FallbackModelEntry[]) => void;
  cards: CredentialCard[];
  defaultCardId: string | null;
}

export default function FallbackModelsEditor({
  models,
  onChange,
  cards,
  defaultCardId,
}: FallbackModelsEditorProps) {
  const [expanded, setExpanded] = useState(models.length > 0);
  const t = useT();

  function setAt(idx: number, cardId: string | null) {
    onChange(models.map((entry, i) => (i === idx ? { card_id: cardId } : { card_id: entry.card_id ?? null })));
  }

  function handleRemove(idx: number) {
    onChange(
      models.filter((_, i) => i !== idx).map((entry) => ({ card_id: entry.card_id ?? null })),
    );
  }

  function handleAdd() {
    onChange([...models.map((entry) => ({ card_id: entry.card_id ?? null })), { card_id: null }]);
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        // Section-heading weight — see the note on LLMProviderSettings'
        // disclosure button; a collapsed section still has to read as one.
        className="flex items-center gap-2 text-[15px] font-semibold leading-6 text-primary hover:text-accent transition-all duration-150 w-full text-left mb-2"
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 shrink-0" />
        )}
        <span>{t('fallback.heading')}</span>
        {!expanded && models.length > 0 && (
          <span className="text-[13px] text-secondary font-normal ml-1">
            {t('fallback.configuredCount', { n: models.length })}
          </span>
        )}
      </button>

      {expanded && (
        <div className="space-y-3 ml-6">
          <p className="text-xs text-secondary">{t('fallback.intro')}</p>

          {models.map((entry, idx) => (
            <div key={idx} className="flex items-start gap-2" data-testid={`fallback-row-${idx}`}>
              {/* The rung's position is the chain order — numbered, because
                  "tried second" is the only thing a row's place means. */}
              <span className="text-xs text-secondary/70 font-mono pt-2.5 w-4 shrink-0">
                {idx + 1}
              </span>
              <div className="min-w-0 flex-1">
                <CardPicker
                  cards={cards}
                  defaultCardId={defaultCardId}
                  value={entry.card_id ?? null}
                  onChange={(cardId) => setAt(idx, cardId)}
                  allowGlobalDefault={false}
                  hideHealth
                  data-testid={`fallback-picker-${idx}`}
                />
              </div>
              <button
                type="button"
                onClick={() => handleRemove(idx)}
                data-testid={`fallback-remove-${idx}`}
                className="shrink-0 text-secondary hover:text-error transition-colors p-1 mt-1.5 max-md:min-w-[44px] max-md:min-h-[44px] flex items-center justify-center"
                title={t('fallback.remove')}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}

          <button
            type="button"
            onClick={handleAdd}
            data-testid="fallback-add"
            className="flex items-center gap-1.5 text-sm text-secondary hover:text-accent transition-all duration-150"
          >
            <Plus className="w-4 h-4" />
            {t('fallback.addCta')}
          </button>
        </div>
      )}
    </div>
  );
}

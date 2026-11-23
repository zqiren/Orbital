// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The create-project heads-up: shown only when the Orbital agent CAN'T start
 * (no key on the default card). The happy path renders nothing: a "using the
 * global defaults" card asked users to read a paragraph to learn that nothing
 * was needed here.
 *
 * Batch 1123: the provider form this file used to hold (provider dropdown,
 * key, model combobox, Advanced, Test Connection, Save) is gone. Connecting a
 * provider is the connect panel (`connect/ConnectPanel`), used by the
 * wizard's key step and by Settings → Add / Edit.
 */
import { useEffect, useState } from 'react';
import { api } from '../config';
import { useT } from '../i18n/useT';

interface LLMSettingsResponse {
  /** Keyless mode (2026-10-21): the wizard's "Skip for now" choice. */
  setup?: { api_key_skipped?: boolean };
  llm: { api_key_set: boolean };
}

export default function LLMProviderSettings() {
  const t = useT();
  const [loaded, setLoaded] = useState(false);
  const [keySet, setKeySet] = useState(false);
  // Keyless mode: the user skipped Orbital's own key, so the heads-up must not
  // tell them to "set up your API key before creating a project" — projects
  // work, pinned to their agent.
  const [skipped, setSkipped] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api<LLMSettingsResponse>('/api/v2/settings')
      .then((data) => {
        if (cancelled) return;
        setKeySet(!!data?.llm?.api_key_set);
        setSkipped(!!data?.setup?.api_key_skipped);
      })
      .catch(() => { /* not configured */ })
      .finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, []);

  if (!loaded || keySet) return null;
  return (
    <div className="border border-warning/30 rounded-lg p-4 bg-warning/5">
      <p className="text-body text-primary font-medium mb-1">{t('llm.provider.heading')}</p>
      <p className="text-body text-secondary">
        {skipped ? t('llm.wizard.keyless') : t('llm.wizard.notConfigured')}
      </p>
    </div>
  );
}

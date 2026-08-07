// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef } from 'react';
import { Check, X, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import type { ProviderRegistry, ProviderInfo } from '../types';
import { api } from '../config';
import { useT } from '../i18n/useT';

interface LLMSettingsResponse {
  llm: {
    api_key_set: boolean;
    api_key_masked: string;
    base_url: string | null;
    model: string | null;
    sdk: string;
    provider: string;
  };
}

export interface LLMValues {
  provider?: string;
  model: string;
  api_key?: string;
  base_url?: string;
  sdk: 'openai' | 'anthropic';
}

interface LLMProviderSettingsProps {
  mode: 'global' | 'project' | 'wizard';
  /** For project mode: existing project LLM values */
  projectValues?: {
    provider?: string;
    model: string;
    api_key?: string;
    base_url?: string | null;
    sdk?: string;
  };
  /** Called whenever values change (project/global modes) */
  onChange?: (values: LLMValues) => void;
  /** Hide the Save button (caller handles save via saveRef) */
  hideSaveButton?: boolean;
  /** Ref to expose save function to parent */
  saveRef?: React.MutableRefObject<(() => Promise<boolean>) | null>;
  /** Provider picker style. 'cards' is wizard-only (Spec 17); everywhere else
   * keeps the compact dropdown. Default 'dropdown'. */
  providerPicker?: 'dropdown' | 'cards';
}

export const CUSTOM_PROVIDER_KEY = '__custom__';

/** Explicit fresh-state default (Spec 17 §9) — works from both CN and global
 * networks, single endpoint, nothing to misconfigure. Never 'custom'. */
const DEFAULT_PROVIDER = 'deepseek';

/** Fixed provider ordering for everyone — CN-friendly providers first, no
 * locale logic (Spec 17 §9, "i dont want to over complicate this"). Unknown
 * registry keys are appended after the known ones in registry order; any
 * 'custom' entry (registry key or the CUSTOM_PROVIDER_KEY sentinel) is always
 * last. Shared by the dropdown and the wizard's preset cards. */
const PROVIDER_ORDER = [
  'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax',
  'openai', 'anthropic', 'google', 'xai', 'mistral', 'groq', 'together', 'openrouter',
];

export function orderProviders(keys: string[]): string[] {
  const isCustom = (k: string) => k === 'custom' || k === CUSTOM_PROVIDER_KEY;
  const known = PROVIDER_ORDER.filter((k) => keys.includes(k));
  const unknown = keys.filter((k) => !PROVIDER_ORDER.includes(k) && !isCustom(k));
  const custom = keys.filter(isCustom);
  return [...known, ...unknown, ...custom];
}

/** Region default for a newly-selected provider (Spec 17 §9): dual-endpoint
 * providers default to 'china' (launch audience is the Chinese X community);
 * everyone else defaults to 'global'. Does NOT apply to saved-config
 * derivation (base_url === china_base_url), which is handled separately. */
export function regionDefaultForProvider(
  info: Pick<ProviderInfo, 'china_base_url'> | undefined,
): 'global' | 'china' {
  return info?.china_base_url ? 'china' : 'global';
}

export default function LLMProviderSettings({
  mode,
  projectValues,
  onChange,
  hideSaveButton,
  saveRef,
  providerPicker = 'dropdown',
}: LLMProviderSettingsProps) {
  const t = useT();
  // Collapsed state for project mode
  const [expanded, setExpanded] = useState(mode !== 'project');

  // Provider registry
  const [providers, setProviders] = useState<ProviderRegistry>({});
  const [providersLoading, setProvidersLoading] = useState(true);

  // Global defaults (fetched for all modes)
  const [globalSettings, setGlobalSettings] = useState<LLMSettingsResponse['llm'] | null>(null);
  const [globalLoaded, setGlobalLoaded] = useState(false);

  // Field state
  const [provider, setProvider] = useState('');
  const [sdk, setSdk] = useState<'openai' | 'anthropic'>('openai');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [region, setRegion] = useState<'global' | 'china'>('global');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Global mode save state
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState('');

  // Model list state
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [modelSource, setModelSource] = useState<'none' | 'api' | 'suggested' | 'freetext'>('none');
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelInputValue, setModelInputValue] = useState('');
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  // Test connection state
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');

  const modelComboRef = useRef<HTMLDivElement>(null);

  // ---- Data fetching ----

  // Fetch global settings (for global mode: to populate fields; for project/wizard: to know if configured)
  useEffect(() => {
    let cancelled = false;
    async function fetchGlobal() {
      try {
        const data = await api<LLMSettingsResponse>('/api/v2/settings');
        if (!cancelled) {
          setGlobalSettings(data.llm);
        }
      } catch {
        // not configured
      } finally {
        if (!cancelled) setGlobalLoaded(true);
      }
    }
    fetchGlobal();
    return () => { cancelled = true; };
  }, []);

  // Fetch providers
  useEffect(() => {
    let cancelled = false;
    async function fetchProviders() {
      try {
        const data = await api<ProviderRegistry>('/api/v2/providers');
        if (!cancelled) {
          setProviders(data);
        }
      } catch {
        // registry not available
      } finally {
        if (!cancelled) setProvidersLoading(false);
      }
    }
    fetchProviders();
    return () => { cancelled = true; };
  }, []);

  // Initialize fields once providers + global settings loaded
  const [initialized, setInitialized] = useState(false);
  useEffect(() => {
    if (initialized) return;
    if (providersLoading || !globalLoaded) return;

    const providerKeys = Object.keys(providers);
    // Explicit default (never 'custom'); guard against DEFAULT_PROVIDER being
    // absent from the registry by falling back to the first known key.
    const defaultProviderKey = providers[DEFAULT_PROVIDER] ? DEFAULT_PROVIDER : providerKeys[0];
    // Track the provider we resolve to so we can seed the model picker below.
    let resolvedProvider = '';

    if (mode === 'global') {
      // Populate from global settings
      if (globalSettings) {
        const gp = globalSettings.provider || '';
        // Seeded only by the no-saved-provider branch below, where
        // globalSettings.base_url is never set — used to avoid storing ''
        // (breaks the backend's OpenAI client, which treats '' unlike None).
        let defaultBaseUrlSeed = '';
        if (gp && gp !== 'custom' && providers[gp]) {
          setProvider(gp);
          resolvedProvider = gp;
          setSdk(providers[gp].sdk);
          // Detect region
          const info = providers[gp];
          if (info.china_base_url && globalSettings.base_url === info.china_base_url) {
            setRegion('china');
          }
        } else if (gp) {
          // Unknown provider, or the registry's literal 'custom' entry — both
          // resolve to the CUSTOM_PROVIDER_KEY sentinel, the single Custom
          // affordance in the picker.
          setProvider(CUSTOM_PROVIDER_KEY);
          resolvedProvider = CUSTOM_PROVIDER_KEY;
          setShowAdvanced(true);
        } else if (defaultProviderKey) {
          setProvider(defaultProviderKey);
          resolvedProvider = defaultProviderKey;
          setSdk(providers[defaultProviderKey].sdk);
          defaultBaseUrlSeed = resolveBaseUrl(providers[defaultProviderKey], 'global');
        }
        setModel(globalSettings.model || '');
        setModelInputValue(globalSettings.model || '');
        setBaseUrl(globalSettings.base_url || defaultBaseUrlSeed);
        if (globalSettings.sdk) setSdk(globalSettings.sdk as 'openai' | 'anthropic');
      } else if (defaultProviderKey) {
        setProvider(defaultProviderKey);
        resolvedProvider = defaultProviderKey;
        setSdk(providers[defaultProviderKey].sdk);
        setBaseUrl(resolveBaseUrl(providers[defaultProviderKey], 'global'));
      }
    } else if (mode === 'project') {
      // Populate from project values, fallback to global
      const pv = projectValues;
      let projectBaseUrlSeed = '';
      if (pv?.provider && pv.provider !== 'custom' && providers[pv.provider]) {
        setProvider(pv.provider);
        resolvedProvider = pv.provider;
        setSdk(providers[pv.provider].sdk);
        const info = providers[pv.provider];
        if (info.china_base_url && pv.base_url === info.china_base_url) {
          setRegion('china');
        } else if (!pv.base_url) {
          // Saved provider with no saved endpoint: seed the provider's
          // region-default URL instead of leaving it empty — an empty
          // base_url falls back cross-provider on the backend, pairing
          // this project's key with another provider's endpoint (the
          // wrong-region 401 trap).
          const defaultRegion = regionDefaultForProvider(info);
          setRegion(defaultRegion);
          projectBaseUrlSeed = resolveBaseUrl(info, defaultRegion);
        }
      } else if (pv?.provider) {
        // Unknown provider, or the registry's literal 'custom' entry — both
        // resolve to the CUSTOM_PROVIDER_KEY sentinel, the single Custom
        // affordance in the picker.
        setProvider(CUSTOM_PROVIDER_KEY);
        resolvedProvider = CUSTOM_PROVIDER_KEY;
        setShowAdvanced(true);
      } else if (defaultProviderKey) {
        setProvider(defaultProviderKey);
        resolvedProvider = defaultProviderKey;
        setSdk(providers[defaultProviderKey].sdk);
      }
      setModel(pv?.model || '');
      setModelInputValue(pv?.model || '');
      setBaseUrl(pv?.base_url || projectBaseUrlSeed);
      if (pv?.sdk) setSdk(pv.sdk as 'openai' | 'anthropic');
    }
    // wizard mode: no fields to initialize

    // Seed the model dropdown from the catalog so it's populated before any key.
    if (resolvedProvider) seedModelsFromCatalog(resolvedProvider);

    setInitialized(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providersLoading, globalLoaded, initialized, mode]);

  // Notify parent of changes (project mode)
  useEffect(() => {
    if (mode !== 'project' || !initialized) return;
    onChange?.({
      provider: provider === CUSTOM_PROVIDER_KEY ? undefined : provider || undefined,
      model: model.trim() || modelInputValue.trim(),
      api_key: apiKey.trim() || undefined,
      base_url: baseUrl.trim() || undefined,
      sdk,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, model, modelInputValue, apiKey, baseUrl, sdk, initialized]);

  // ---- Helper functions ----

  function resolveBaseUrl(info: ProviderInfo, r: 'global' | 'china'): string {
    if (r === 'china' && info.china_base_url) return info.china_base_url;
    return info.base_url || '';
  }

  // Seed the model picker from the catalog's suggested_models so a dropdown is
  // available immediately, with no network call and before any API key. A live
  // /models fetch upgrades this to modelSource='api' when it succeeds. Providers
  // with no catalog (custom) fall back to free-text entry.
  function seedModelsFromCatalog(key: string) {
    if (key === CUSTOM_PROVIDER_KEY) {
      setModelOptions([]);
      setModelSource('freetext');
      return;
    }
    const info: ProviderInfo | undefined = providers[key];
    if (info && info.suggested_models.length > 0) {
      setModelOptions(info.suggested_models);
      setModelSource('suggested');
    } else {
      setModelOptions([]);
      setModelSource('freetext');
    }
  }

  const currentProviderHasChinaUrl =
    provider !== CUSTOM_PROVIDER_KEY && !!providers[provider]?.china_base_url;

  // Fixed order, shared by the dropdown and the wizard's preset cards. The
  // registry's literal 'custom' entry is excluded — the CUSTOM_PROVIDER_KEY
  // sentinel below is the single Custom affordance; rendering both produced
  // duplicate "Custom / Self-Hosted" chips/options.
  const orderedProviderKeys = orderProviders(
    Object.keys(providers).filter((k) => k !== 'custom'),
  );

  function handleProviderChange(key: string) {
    setProvider(key);
    setModel('');
    setModelInputValue('');
    setTestStatus('idle');
    setTestMessage('');
    // Seed the dropdown from the catalog immediately (no key required).
    seedModelsFromCatalog(key);
    if (key === CUSTOM_PROVIDER_KEY) {
      setRegion('global');
      setBaseUrl('');
      setSdk('openai');
      setShowAdvanced(true);
    } else {
      const info: ProviderInfo | undefined = providers[key];
      // Spec 17: a newly-selected dual-endpoint provider defaults to the China
      // region (no saved base_url for it yet). Saved-config derivation (the
      // base_url === china_base_url check in the init effect) is untouched.
      const defaultRegion = regionDefaultForProvider(info);
      setRegion(defaultRegion);
      if (info) {
        setBaseUrl(resolveBaseUrl(info, defaultRegion));
        setSdk(info.sdk);
      }
    }
  }

  function handleRegionChange(r: 'global' | 'china') {
    setRegion(r);
    const info = providers[provider];
    if (info) {
      setBaseUrl(resolveBaseUrl(info, r));
    }
    setTestStatus('idle');
    setTestMessage('');
  }

  // Fetch models when provider + key available
  useEffect(() => {
    if (!initialized) return;
    if (!provider || provider === CUSTOM_PROVIDER_KEY) return;
    // Respect an explicit free-text choice (escape hatch / empty catalog):
    // don't yank the user back into a dropdown by a background fetch.
    if (modelSource === 'freetext') return;

    // Need either a new key typed in, or existing key (global or project)
    const hasKey = apiKey.trim() || globalSettings?.api_key_set || (mode === 'project' && projectValues?.api_key);
    if (!hasKey) return;

    const info = providers[provider];
    if (!info) return;

    // Do NOT clear the catalog-seeded options here — keep the suggested dropdown
    // usable while the live fetch is in flight. A success upgrades to 'api'; a
    // failure re-seeds the suggested list (or free-text for empty catalogs).
    let cancelled = false;
    setModelsLoading(true);

    async function fetchModels() {
      try {
        const body: Record<string, unknown> = {
          provider,
          base_url: baseUrl.trim() || undefined,
        };
        if (apiKey.trim()) {
          body.api_key = apiKey.trim();
        }
        // Backend returns { models: string[] }.
        const result = await api<{ models?: string[] }>('/api/v2/providers/models', {
          method: 'POST',
          body: JSON.stringify(body),
        });
        const models = result?.models ?? [];
        if (!cancelled && models.length > 0) {
          setModelOptions(models);
          setModelSource('api');
        } else {
          throw new Error('empty');
        }
      } catch {
        if (cancelled) return;
        // Fall back to the catalog's suggested models (free-text if none).
        seedModelsFromCatalog(provider);
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    }

    const timer = setTimeout(fetchModels, 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey, provider, providers, initialized]);

  // Close model dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (modelComboRef.current && !modelComboRef.current.contains(e.target as Node)) {
        setShowModelDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const filteredModels = modelOptions.filter((m) =>
    m.toLowerCase().includes(modelInputValue.toLowerCase()),
  );

  function selectModel(value: string) {
    setModel(value);
    setModelInputValue(value);
    setShowModelDropdown(false);
    setTestStatus('idle');
    setTestMessage('');
  }

  function handleModelInputChange(value: string) {
    setModelInputValue(value);
    setModel(value);
    setShowModelDropdown(true);
    setTestStatus('idle');
    setTestMessage('');
  }

  // Escape hatch: switch the picker to free-text entry for a custom model name.
  function enterCustomModel() {
    setModelSource('freetext');
    setModelOptions([]);
    setModel('');
    setModelInputValue('');
    setShowModelDropdown(false);
    setTestStatus('idle');
    setTestMessage('');
    // Focus the (now plain) model input so the user can type immediately.
    requestAnimationFrame(() => {
      modelComboRef.current?.querySelector('input')?.focus();
    });
  }

  // Test connection
  async function handleTestConnection() {
    setTestStatus('testing');
    setTestMessage('');
    try {
      const body: Record<string, unknown> = {
        provider: provider === CUSTOM_PROVIDER_KEY ? undefined : provider,
        model: model.trim(),
        base_url: baseUrl.trim() || undefined,
        sdk,
      };
      if (apiKey.trim()) {
        body.api_key = apiKey.trim();
      }
      const result = await api<{ status: string; message?: string; error?: string }>(
        '/api/v2/providers/test',
        {
          method: 'POST',
          body: JSON.stringify(body),
        },
      );
      if (result.status === 'ok') {
        setTestStatus('success');
        const displayProvider =
          provider === CUSTOM_PROVIDER_KEY
            ? t('llm.test.customProvider')
            : providers[provider]?.display_name || provider;
        setTestMessage(t('llm.test.success', { provider: displayProvider, model }));
      } else {
        setTestStatus('error');
        setTestMessage(result.error || result.message || t('llm.test.failed'));
      }
    } catch (err: unknown) {
      setTestStatus('error');
      const msg = err instanceof Error ? err.message : t('llm.test.failedAlt');
      setTestMessage(msg);
    }
  }

  // API key status
  const [apiKeyStatus, setApiKeyStatus] = useState<{ configured: boolean; source: string } | null>(null);

  useEffect(() => {
    if (mode !== 'global') return;
    let cancelled = false;
    api<{ configured: boolean; source: string }>('/api/v2/settings/api-key/status')
      .then(data => { if (!cancelled) setApiKeyStatus(data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [mode]);

  async function handleDeleteApiKey() {
    if (!confirm(t('llm.apiKey.deleteConfirm'))) return;
    try {
      await api('/api/v2/settings/api-key', { method: 'DELETE' });
      setApiKeyStatus({ configured: false, source: 'none' });
      setGlobalSettings(prev => prev ? { ...prev, api_key_set: false, api_key_masked: '' } : prev);
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err.message : t('llm.apiKey.deleteError'));
    }
  }

  // Core save logic (shared between form submit and external saveRef)
  async function doSave(): Promise<boolean> {
    setSaving(true);
    setSaved(false);
    setSaveError('');
    try {
      // Save API key via dedicated secure endpoint if provided
      if (apiKey.trim()) {
        await api('/api/v2/settings/api-key', {
          method: 'PUT',
          body: JSON.stringify({ api_key: apiKey.trim() }),
        });
        setApiKeyStatus({ configured: true, source: 'keyring' });
      }

      // Save other settings via generic endpoint (without the API key)
      const body: Record<string, string | undefined> = {};
      body.llm_base_url = baseUrl.trim();
      body.llm_model = (model.trim() || modelInputValue.trim());
      body.llm_sdk = sdk;
      body.llm_provider = provider === CUSTOM_PROVIDER_KEY ? 'custom' : provider;

      const data = await api<LLMSettingsResponse>('/api/v2/settings', {
        method: 'PUT',
        body: JSON.stringify(body),
      });
      setGlobalSettings(data.llm);
      setApiKey('');
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      return true;
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err.message : t('llm.saveError'));
      return false;
    } finally {
      setSaving(false);
    }
  }

  // Expose save function to parent via ref
  useEffect(() => {
    if (saveRef) saveRef.current = doSave;
    return () => { if (saveRef) saveRef.current = null; };
  });

  // Global mode: save to /api/v2/settings (and dedicated api-key endpoint)
  async function handleGlobalSave(ev: React.FormEvent) {
    ev.preventDefault();
    await doSave();
  }

  const canTestConnection = !!model.trim();

  // ---- Wizard mode: the create-project modal only wants a heads-up when the
  // agent CAN'T start (no global API key) — the happy-path "using global
  // defaults" card asked users to read a paragraph to learn "nothing needed
  // here", so it (and the loading spinner) render nothing at all. Only the
  // not-configured warning is a visible card. ----
  if (mode === 'wizard') {
    if (!globalLoaded) return null;
    if (globalSettings?.api_key_set) return null;
    return (
      <div className="border border-warning/30 rounded-lg p-4 bg-warning/5">
        <p className="text-sm text-primary font-medium mb-1">{t('llm.provider.heading')}</p>
        <p className="text-sm text-secondary">
          {t('llm.wizard.notConfigured')}
        </p>
      </div>
    );
  }

  // ---- Loading state ----
  if (providersLoading || !globalLoaded) {
    return (
      <div className="flex items-center gap-2 text-sm text-secondary py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        {t('llm.loading')}
      </div>
    );
  }

  // ---- Project mode: collapsible header ----
  const projectHeader = mode === 'project' ? (
    <button
      type="button"
      onClick={() => setExpanded(!expanded)}
      className="flex items-center gap-2 text-sm font-medium text-primary hover:text-accent transition-all duration-150 w-full text-left mb-3"
    >
      {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      <span>{t('llm.provider.heading')}</span>
      {!expanded && (
        <span className="text-secondary font-normal ml-1">
          {t('llm.project.usingDefault', {
            provider: projectValues?.provider && providers[projectValues.provider]
              ? providers[projectValues.provider].display_name
              : globalSettings?.provider && providers[globalSettings.provider]
                ? providers[globalSettings.provider].display_name
                : t('llm.project.globalFallback'),
          })}
        </span>
      )}
    </button>
  ) : null;

  // ---- Global mode header ----
  const globalHeader = mode === 'global' ? (
    <div className="mb-4">
      <h2 className="text-sm font-semibold text-primary mb-1">{t('llm.global.heading')}</h2>
      <p className="text-xs text-secondary">
        {t('llm.global.subhead')}
      </p>
    </div>
  ) : null;

  // ---- Shared fields JSX ----
  const fieldsJSX = (
    <div className="space-y-5">
      {mode === 'project' && (
        <p className="text-xs text-secondary -mt-2">
          {t('llm.project.overrideHint')}
        </p>
      )}

      {/* Provider */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.provider')}</label>
        {providerPicker === 'cards' ? (
          <div className="flex flex-wrap gap-2">
            {orderedProviderKeys.map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => handleProviderChange(key)}
                className={`text-sm px-3 py-1.5 rounded-full border transition-all duration-150 ${
                  provider === key
                    ? 'bg-accent text-white border-accent'
                    : 'bg-sidebar text-secondary border-border hover:text-primary hover:border-secondary/40'
                }`}
              >
                {providers[key].display_name}
              </button>
            ))}
            <button
              type="button"
              onClick={() => handleProviderChange(CUSTOM_PROVIDER_KEY)}
              className={`text-sm px-3 py-1.5 rounded-full border transition-all duration-150 ${
                provider === CUSTOM_PROVIDER_KEY
                  ? 'bg-accent text-white border-accent'
                  : 'bg-sidebar text-secondary border-border hover:text-primary hover:border-secondary/40'
              }`}
            >
              {t('llm.provider.custom')}
            </button>
          </div>
        ) : (
          <select
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          >
            {orderedProviderKeys.map((key) => (
              <option key={key} value={key}>{providers[key].display_name}</option>
            ))}
            <option value={CUSTOM_PROVIDER_KEY}>{t('llm.provider.custom')}</option>
          </select>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.notes && (
          <p className="text-xs text-secondary mt-1">{providers[provider].notes}</p>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.no_china_endpoint && (
          <p className="text-xs text-secondary/70 mt-1">{t('llm.provider.noChinaEndpoint')}</p>
        )}
      </div>

      {/* Region toggle */}
      {currentProviderHasChinaUrl && (
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.region')}</label>
          <div className="inline-flex rounded-lg border border-border overflow-hidden">
            <button
              type="button"
              onClick={() => handleRegionChange('global')}
              className={`text-sm px-4 py-1.5 transition-all duration-150 ${
                region === 'global'
                  ? 'bg-accent text-white'
                  : 'bg-sidebar text-secondary hover:text-primary'
              }`}
            >
              {t('llm.region.global')}
            </button>
            <button
              type="button"
              onClick={() => handleRegionChange('china')}
              className={`text-sm px-4 py-1.5 transition-all duration-150 ${
                region === 'china'
                  ? 'bg-accent text-white'
                  : 'bg-sidebar text-secondary hover:text-primary'
              }`}
            >
              {t('llm.region.china')}
            </button>
          </div>
        </div>
      )}

      {/* API Key */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.apiKey')}</label>
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.console_url && (
          <div className="mb-1.5">
            <a
              href={providers[provider].console_url!}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              {t('llm.apiKey.getKey')}
            </a>
            <p className="text-xs text-secondary/70 mt-0.5">{t('llm.apiKey.howTo')}</p>
          </div>
        )}
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={
            globalSettings?.api_key_set
              ? t('llm.apiKey.replacePlaceholder')
              : t('llm.apiKey.placeholder')
          }
          className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
        />
        {mode === 'global' && globalSettings?.api_key_set && !apiKey && (
          <div className="flex items-center gap-3 mt-1">
            <p className="text-xs text-secondary">
              {t('llm.apiKey.current', { masked: globalSettings.api_key_masked })}
              {apiKeyStatus?.source && apiKeyStatus.source !== 'none' && (
                <span className="ml-1 text-secondary/60">({apiKeyStatus.source})</span>
              )}
            </p>
            <button
              type="button"
              onClick={handleDeleteApiKey}
              className="text-xs text-error hover:text-error/80 transition-colors"
            >
              {t('llm.apiKey.remove')}
            </button>
          </div>
        )}
      </div>

      {/* Model combobox */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.model')}</label>
        {modelsLoading ? (
          <div className="flex items-center gap-2 text-sm text-secondary py-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            {t('llm.model.fetching')}
          </div>
        ) : (
          <div className="relative" ref={modelComboRef}>
            <input
              type="text"
              value={modelInputValue}
              onChange={(e) => handleModelInputChange(e.target.value)}
              onFocus={() => {
                if (modelOptions.length > 0) setShowModelDropdown(true);
              }}
              placeholder={
                modelSource === 'freetext'
                  ? t('llm.model.freetextPlaceholder')
                  : t('llm.model.selectPlaceholder')
              }
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            {showModelDropdown &&
              modelSource !== 'freetext' &&
              (filteredModels.length > 0 || modelOptions.length > 0) && (
              <ul className="absolute z-10 w-full mt-1 max-h-48 overflow-y-auto bg-sidebar border border-border rounded-lg shadow-lg">
                {filteredModels.map((m) => (
                  <li key={m}>
                    <button
                      type="button"
                      onClick={() => selectModel(m)}
                      className={`w-full text-left text-sm px-3 py-2 hover:bg-accent/5 hover:text-primary transition-all duration-150 ${
                        m === model ? 'text-accent font-medium' : 'text-primary'
                      }`}
                    >
                      {m}
                    </button>
                  </li>
                ))}
                {modelInputValue &&
                  !modelOptions.includes(modelInputValue) && (
                    <li>
                      <button
                        type="button"
                        onClick={() => selectModel(modelInputValue)}
                        className="w-full text-left text-sm px-3 py-2 text-secondary hover:bg-accent/5 hover:text-primary transition-all duration-150 italic"
                      >
                        {t('llm.model.useCustom', { value: modelInputValue })}
                      </button>
                    </li>
                  )}
                {/* Escape hatch: always offer free-text entry as the last option. */}
                <li className="border-t border-border">
                  <button
                    type="button"
                    onClick={enterCustomModel}
                    className="w-full text-left text-sm px-3 py-2 text-secondary hover:bg-accent/5 hover:text-primary transition-all duration-150 italic"
                  >
                    {t('llm.model.enterCustom')}
                  </button>
                </li>
              </ul>
            )}
            {modelSource === 'api' && (
              <p className="text-xs text-secondary mt-1">
                {t('llm.model.source.api')}
              </p>
            )}
            {modelSource === 'suggested' && (
              <p className="text-xs text-secondary mt-1">
                {t('llm.model.source.suggested')}
              </p>
            )}
            {modelSource === 'freetext' && (
              <p className="text-xs text-secondary mt-1">
                {t('llm.model.source.freetext')}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Advanced */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-secondary hover:text-primary transition-all duration-150"
        >
          {showAdvanced ? '\u25BE' : '\u25B8'} {t('llm.advanced')}
        </button>
        {showAdvanced && (
          <div className="mt-3 space-y-3">
            <div>
              <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.baseUrl')}</label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={t('llm.baseUrl.placeholder')}
                className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
            </div>
            {provider === CUSTOM_PROVIDER_KEY && (
              <div>
                <label className="block text-sm font-medium text-primary mb-1.5">{t('llm.field.sdk')}</label>
                <select
                  value={sdk}
                  onChange={(e) => setSdk(e.target.value as 'openai' | 'anthropic')}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
                >
                  <option value="openai">{t('llm.sdk.openai')}</option>
                  <option value="anthropic">{t('llm.sdk.anthropic')}</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Test Connection */}
      {canTestConnection && (
        <div>
          <button
            type="button"
            onClick={handleTestConnection}
            disabled={testStatus === 'testing'}
            className="inline-flex items-center gap-2 text-sm font-medium text-secondary border border-border rounded-lg px-4 py-2 hover:border-secondary/40 hover:text-primary transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px] max-md:justify-center"
          >
            {testStatus === 'testing' ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t('llm.testing')}
              </>
            ) : (
              t('llm.test')
            )}
          </button>
          {testStatus === 'success' && (
            <p className="flex items-center gap-1.5 text-sm text-success mt-2">
              <Check className="w-4 h-4" />
              {testMessage}
            </p>
          )}
          {testStatus === 'error' && (
            <p className="flex items-center gap-1.5 text-sm text-error mt-2">
              <X className="w-4 h-4" />
              {testMessage}
            </p>
          )}
        </div>
      )}

      {/* Global mode: save button */}
      {mode === 'global' && !hideSaveButton && (
        <div className="flex items-center gap-3 pt-2">
          <button
            type="submit"
            disabled={saving}
            className="inline-flex items-center gap-2 bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px] max-md:justify-center"
          >
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t('llm.saving')}
              </>
            ) : (
              t('settings.save')
            )}
          </button>
          {saved && (
            <span className="flex items-center gap-1 text-sm text-success">
              <Check className="w-4 h-4" />
              {t('settings.saved')}
            </span>
          )}
          {saveError && (
            <span className="text-sm text-error">{saveError}</span>
          )}
        </div>
      )}
    </div>
  );

  // ---- Render ----
  if (mode === 'global') {
    return (
      <form onSubmit={handleGlobalSave}>
        {globalHeader}
        {fieldsJSX}
      </form>
    );
  }

  // project mode
  return (
    <div>
      {projectHeader}
      {expanded && fieldsJSX}
    </div>
  );
}

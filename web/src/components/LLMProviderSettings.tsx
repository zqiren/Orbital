// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef } from 'react';
import { Check, X, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import type { ProviderRegistry, ProviderInfo } from '../types';
import { api } from '../config';

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
}

const CUSTOM_PROVIDER_KEY = '__custom__';

export default function LLMProviderSettings({
  mode,
  projectValues,
  onChange,
  hideSaveButton,
  saveRef,
}: LLMProviderSettingsProps) {
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

    if (mode === 'global') {
      // Populate from global settings
      if (globalSettings) {
        const gp = globalSettings.provider || '';
        if (gp && providers[gp]) {
          setProvider(gp);
          setSdk(providers[gp].sdk);
          // Detect region
          const info = providers[gp];
          if (info.china_base_url && globalSettings.base_url === info.china_base_url) {
            setRegion('china');
          }
        } else if (gp && !providers[gp]) {
          setProvider(CUSTOM_PROVIDER_KEY);
          setShowAdvanced(true);
        } else if (providerKeys.length > 0) {
          setProvider(providerKeys[0]);
          setSdk(providers[providerKeys[0]].sdk);
        }
        setModel(globalSettings.model || '');
        setModelInputValue(globalSettings.model || '');
        setBaseUrl(globalSettings.base_url || '');
        if (globalSettings.sdk) setSdk(globalSettings.sdk as 'openai' | 'anthropic');
      } else if (providerKeys.length > 0) {
        setProvider(providerKeys[0]);
        setSdk(providers[providerKeys[0]].sdk);
        setBaseUrl(resolveBaseUrl(providers[providerKeys[0]], 'global'));
      }
    } else if (mode === 'project') {
      // Populate from project values, fallback to global
      const pv = projectValues;
      if (pv?.provider && providers[pv.provider]) {
        setProvider(pv.provider);
        setSdk(providers[pv.provider].sdk);
        const info = providers[pv.provider];
        if (info.china_base_url && pv.base_url === info.china_base_url) {
          setRegion('china');
        }
      } else if (pv?.provider && !providers[pv.provider]) {
        setProvider(CUSTOM_PROVIDER_KEY);
        setShowAdvanced(true);
      } else if (providerKeys.length > 0) {
        setProvider(providerKeys[0]);
        setSdk(providers[providerKeys[0]].sdk);
      }
      setModel(pv?.model || '');
      setModelInputValue(pv?.model || '');
      setBaseUrl(pv?.base_url || '');
      if (pv?.sdk) setSdk(pv.sdk as 'openai' | 'anthropic');
    }
    // wizard mode: no fields to initialize

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

  const currentProviderHasChinaUrl =
    provider !== CUSTOM_PROVIDER_KEY && !!providers[provider]?.china_base_url;

  function handleProviderChange(key: string) {
    setProvider(key);
    setModel('');
    setModelInputValue('');
    setModelOptions([]);
    setModelSource('none');
    setTestStatus('idle');
    setTestMessage('');
    setRegion('global');
    if (key === CUSTOM_PROVIDER_KEY) {
      setBaseUrl('');
      setSdk('openai');
      setShowAdvanced(true);
    } else {
      const info: ProviderInfo | undefined = providers[key];
      if (info) {
        setBaseUrl(resolveBaseUrl(info, 'global'));
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

    // Need either a new key typed in, or existing key (global or project)
    const hasKey = apiKey.trim() || globalSettings?.api_key_set || (mode === 'project' && projectValues?.api_key);
    if (!hasKey) return;

    const info = providers[provider];
    if (!info) return;

    let cancelled = false;
    setModelsLoading(true);
    setModelSource('none');
    setModelOptions([]);

    async function fetchModels() {
      try {
        const body: Record<string, unknown> = {
          provider,
          base_url: baseUrl.trim() || undefined,
        };
        if (apiKey.trim()) {
          body.api_key = apiKey.trim();
        }
        const result = await api<string[]>('/api/v2/providers/models', {
          method: 'POST',
          body: JSON.stringify(body),
        });
        if (!cancelled && result && result.length > 0) {
          setModelOptions(result);
          setModelSource('api');
        } else {
          throw new Error('empty');
        }
      } catch {
        if (cancelled) return;
        const info = providers[provider];
        if (info && info.suggested_models.length > 0) {
          setModelOptions(info.suggested_models);
          setModelSource('suggested');
        } else {
          setModelOptions([]);
          setModelSource('freetext');
        }
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
            ? 'custom provider'
            : providers[provider]?.display_name || provider;
        setTestMessage(`Connected to ${displayProvider} using ${model}`);
      } else {
        setTestStatus('error');
        setTestMessage(result.error || result.message || 'Connection failed.');
      }
    } catch (err: unknown) {
      setTestStatus('error');
      const msg = err instanceof Error ? err.message : 'Connection test failed.';
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
    if (!confirm('Remove the stored API key?')) return;
    try {
      await api('/api/v2/settings/api-key', { method: 'DELETE' });
      setApiKeyStatus({ configured: false, source: 'none' });
      setGlobalSettings(prev => prev ? { ...prev, api_key_set: false, api_key_masked: '' } : prev);
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err.message : 'Failed to delete API key.');
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
      setSaveError(err instanceof Error ? err.message : 'Failed to save settings.');
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

  // ---- Wizard mode: just show a link ----
  if (mode === 'wizard') {
    if (!globalLoaded) {
      return (
        <div className="flex items-center gap-2 text-sm text-secondary py-2">
          <Loader2 className="w-4 h-4 animate-spin" />
          Checking LLM configuration...
        </div>
      );
    }
    if (globalSettings?.api_key_set) {
      const displayProvider = globalSettings.provider && providers[globalSettings.provider]
        ? providers[globalSettings.provider].display_name
        : globalSettings.provider || 'Custom';
      const displayModel = globalSettings.model || 'not set';
      return (
        <div className="border border-border rounded-lg p-4 bg-sidebar/50">
          <p className="text-sm text-primary font-medium mb-1">LLM Provider</p>
          <p className="text-sm text-secondary">
            Using global defaults: {displayProvider} / {displayModel}
          </p>
          <p className="text-xs text-secondary mt-2">
            Change defaults in{' '}
            <span className="text-accent cursor-default">Global Settings</span>.
          </p>
        </div>
      );
    }
    return (
      <div className="border border-warning/30 rounded-lg p-4 bg-warning/5">
        <p className="text-sm text-primary font-medium mb-1">LLM Provider</p>
        <p className="text-sm text-secondary">
          No LLM provider configured yet. Set up your API key and model in{' '}
          <span className="text-accent cursor-default">Global Settings</span>{' '}
          before creating a project.
        </p>
      </div>
    );
  }

  // ---- Loading state ----
  if (providersLoading || !globalLoaded) {
    return (
      <div className="flex items-center gap-2 text-sm text-secondary py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading LLM settings...
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
      <span>LLM Provider</span>
      {!expanded && (
        <span className="text-secondary font-normal ml-1">
          (using {projectValues?.provider && providers[projectValues.provider]
            ? providers[projectValues.provider].display_name
            : globalSettings?.provider && providers[globalSettings.provider]
              ? providers[globalSettings.provider].display_name
              : 'global'} default)
        </span>
      )}
    </button>
  ) : null;

  // ---- Global mode header ----
  const globalHeader = mode === 'global' ? (
    <div className="mb-4">
      <h2 className="text-sm font-semibold text-primary mb-1">Default LLM Settings</h2>
      <p className="text-xs text-secondary">
        Used by all projects unless overridden in project settings.
      </p>
    </div>
  ) : null;

  // ---- Shared fields JSX ----
  const fieldsJSX = (
    <div className="space-y-5">
      {mode === 'project' && (
        <p className="text-xs text-secondary -mt-2">
          Leave blank to use global defaults. Only fill in to override for this project.
        </p>
      )}

      {/* Provider */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">Provider</label>
        <select
          value={provider}
          onChange={(e) => handleProviderChange(e.target.value)}
          className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
        >
          {Object.entries(providers).map(([key, info]) => (
            <option key={key} value={key}>{info.display_name}</option>
          ))}
          <option value={CUSTOM_PROVIDER_KEY}>Custom / Self-Hosted</option>
        </select>
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.notes && (
          <p className="text-xs text-secondary mt-1">{providers[provider].notes}</p>
        )}
      </div>

      {/* Region toggle */}
      {currentProviderHasChinaUrl && (
        <div>
          <label className="block text-sm font-medium text-primary mb-1.5">Region</label>
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
              Global
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
              China
            </button>
          </div>
        </div>
      )}

      {/* API Key */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">API Key</label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={
            globalSettings?.api_key_set
              ? 'Type a new key to replace the current one'
              : 'sk-...'
          }
          className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
        />
        {mode === 'global' && globalSettings?.api_key_set && !apiKey && (
          <div className="flex items-center gap-3 mt-1">
            <p className="text-xs text-secondary">
              Current key: {globalSettings.api_key_masked}
              {apiKeyStatus?.source && apiKeyStatus.source !== 'none' && (
                <span className="ml-1 text-secondary/60">({apiKeyStatus.source})</span>
              )}
            </p>
            <button
              type="button"
              onClick={handleDeleteApiKey}
              className="text-xs text-error hover:text-error/80 transition-colors"
            >
              Remove key
            </button>
          </div>
        )}
      </div>

      {/* Model combobox */}
      <div>
        <label className="block text-sm font-medium text-primary mb-1.5">Model</label>
        {modelsLoading ? (
          <div className="flex items-center gap-2 text-sm text-secondary py-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Fetching available models...
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
                  ? 'Type model name, e.g. gpt-4o'
                  : 'Select or type a model name'
              }
              className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
            />
            {showModelDropdown && filteredModels.length > 0 && (
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
                {modelSource === 'suggested' &&
                  modelInputValue &&
                  !modelOptions.includes(modelInputValue) && (
                    <li>
                      <button
                        type="button"
                        onClick={() => selectModel(modelInputValue)}
                        className="w-full text-left text-sm px-3 py-2 text-secondary hover:bg-accent/5 hover:text-primary transition-all duration-150 italic"
                      >
                        Use &quot;{modelInputValue}&quot;
                      </button>
                    </li>
                  )}
              </ul>
            )}
            {modelSource === 'suggested' && (
              <p className="text-xs text-secondary mt-1">
                Could not fetch live models. Showing suggested models for this provider. You can also type a custom model name.
              </p>
            )}
            {modelSource === 'freetext' && (
              <p className="text-xs text-secondary mt-1">
                Enter the model identifier to use with this provider.
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
          {showAdvanced ? '\u25BE' : '\u25B8'} Advanced
        </button>
        {showAdvanced && (
          <div className="mt-3 space-y-3">
            <div>
              <label className="block text-sm font-medium text-primary mb-1.5">Base URL</label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.openai.com/v1"
                className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
            </div>
            {provider === CUSTOM_PROVIDER_KEY && (
              <div>
                <label className="block text-sm font-medium text-primary mb-1.5">SDK</label>
                <select
                  value={sdk}
                  onChange={(e) => setSdk(e.target.value as 'openai' | 'anthropic')}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
                >
                  <option value="openai">OpenAI-compatible</option>
                  <option value="anthropic">Anthropic</option>
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
                Testing...
              </>
            ) : (
              'Test Connection'
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
                Saving...
              </>
            ) : (
              'Save'
            )}
          </button>
          {saved && (
            <span className="flex items-center gap-1 text-sm text-success">
              <Check className="w-4 h-4" />
              Saved
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

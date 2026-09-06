// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useRef, type ReactNode } from 'react';
import { Check, X, Loader2 } from 'lucide-react';
import type {
  CardMutationResponse,
  CardTestResult,
  CredentialCard,
  ProviderRegistry,
  ProviderInfo,
  TokendanceSigninResponse,
} from '../types';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { Locale } from '../i18n/locales';
import watchaLogo from '../assets/watcha-logo.png';
import Select from './Select';

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

interface LLMProviderSettingsProps {
  /**
   * 'global' is the credential-card form (spec 082 §3.2) — the same fields it
   * always had, saving through the card routes instead of the single global
   * key slot. 'wizard' is the create-project heads-up card. The old 'project'
   * mode is gone: a project picks a card (CardPicker), it no longer carries a
   * provider/model/endpoint/key of its own.
   */
  mode: 'global' | 'wizard';
  /** Editing this existing card; absent ⇒ the form creates a new one. */
  card?: CredentialCard | null;
  /** Called after a successful POST/PUT (and after TokenDance provisioning)
   *  with the saved card and its connection-test result. */
  onCardSaved?: (result: CardMutationResponse) => void;
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

/** Fixed provider ordering — CN-friendly providers first (Spec 17 §9). Unknown
 * registry keys are appended after the known ones in registry order; any
 * 'custom' entry (registry key or the CUSTOM_PROVIDER_KEY sentinel) is always
 * last. Shared by the dropdown and the wizard's preset cards.
 *
 * Spec 47 adds the single locale-aware exception: TokenDance is a
 * China-mainland-only router, so it sorts to the top for the zh UI locale and
 * below the international block for en. No geo-detection — a wrong locale
 * guess only affects sort position, the provider still works if selected. */
const PROVIDER_ORDER = [
  'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax',
  'openai', 'anthropic', 'google', 'xai', 'mistral', 'groq', 'together', 'openrouter',
  // Aggregators last, next to OpenRouter. Zen (pay-as-you-go credits) before
  // Go (the $10/mo subscription) — one account and one key serve both.
  'opencode-zen', 'opencode-go',
];

export function providerOrderForLocale(locale: Locale): string[] {
  return locale === 'zh'
    ? ['tokendance', ...PROVIDER_ORDER]
    : [...PROVIDER_ORDER, 'tokendance'];
}

export function orderProviders(keys: string[], locale: Locale = 'en'): string[] {
  const order = providerOrderForLocale(locale);
  const isCustom = (k: string) => k === 'custom' || k === CUSTOM_PROVIDER_KEY;
  const known = order.filter((k) => keys.includes(k));
  const unknown = keys.filter((k) => !order.includes(k) && !isCustom(k));
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

/** Collapsed-by-default wrapper for the long explanatory hint paragraphs
 * (bug #49). Native <details> keeps it state-free — this file already carries
 * three useState toggles, and the disclosure needs no programmatic control.
 * Short safety captions (china-only / no-China-endpoint) stay outside it. */
function HintDetails({ className = '', children }: { className?: string; children: ReactNode }) {
  const t = useT();
  return (
    <details className={`text-xs ${className}`}>
      <summary className="cursor-pointer text-secondary hover:text-primary">
        {t('llm.hints.details')}
      </summary>
      <div className="mt-1">{children}</div>
    </details>
  );
}

export default function LLMProviderSettings({
  mode,
  card,
  onCardSaved,
  hideSaveButton,
  saveRef,
  providerPicker = 'dropdown',
}: LLMProviderSettingsProps) {
  const t = useT();
  const { locale } = useLocale();

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

  // Card save state. `saveTest` is the connection test the save ran (spec 082
  // D9): a failed test still saves the card, so the result is shown here
  // rather than turned into a save failure.
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveTest, setSaveTest] = useState<CardTestResult | null>(null);
  // Card name. Renaming lives in this form because Edit is one of the
  // card's three actions — the list carries no rename affordance of its
  // own, since clicking a card is what selects it.
  const [cardName, setCardName] = useState('');

  // Model list state
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [modelSource, setModelSource] = useState<'none' | 'api' | 'suggested' | 'freetext'>('none');
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelInputValue, setModelInputValue] = useState('');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  // Set only by the "enter a custom model" escape hatch. The fetch effect used
  // to gate on `modelSource === 'freetext'`, which conflated a deliberate
  // choice with "this provider has no catalog" — and the latter is exactly the
  // case (Custom / self-hosted) that most needs a live list.
  const [freetextPinned, setFreetextPinned] = useState(false);

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

    if (mode === 'global' && card) {
      // Editing an existing card: the card IS the configuration — never fall
      // back to the global settings block here, or an edit would silently
      // adopt the default card's provider.
      const cp = card.provider;
      if (cp && cp !== 'custom' && providers[cp]) {
        setProvider(cp);
        resolvedProvider = cp;
        setSdk(providers[cp].sdk);
        setRegion(card.region === 'china' ? 'china' : 'global');
        setBaseUrl(card.base_url || resolveBaseUrl(providers[cp], card.region));
      } else {
        setProvider(CUSTOM_PROVIDER_KEY);
        resolvedProvider = CUSTOM_PROVIDER_KEY;
        setShowAdvanced(true);
        setBaseUrl(card.base_url || '');
      }
      if (card.sdk) setSdk(card.sdk);
      setModel(card.model || '');
      setModelInputValue(card.model || '');
      setCardName(card.name || '');
    } else if (mode === 'global') {
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
    }
    // wizard mode: no fields to initialize

    // Seed the model dropdown from the catalog so it's populated before any key.
    if (resolvedProvider) seedModelsFromCatalog(resolvedProvider);

    setInitialized(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providersLoading, globalLoaded, initialized, mode]);

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

  // Once a TokenDance key is saved (saved provider is tokendance and the
  // global slot holds a key), the one-click button retires — TokenDance then
  // behaves like every other provider; top-up happens on their site via the
  // regular console link. Browsing to the card while another provider's key
  // occupies the slot still offers the one-click signin.
  const tokendanceKeyActive = card
    ? card.provider === 'tokendance' && card.key_set
    : globalSettings?.provider === 'tokendance' && !!globalSettings?.api_key_set;

  // Fixed order, shared by the dropdown and the wizard's preset cards. The
  // registry's literal 'custom' entry is excluded — the CUSTOM_PROVIDER_KEY
  // sentinel below is the single Custom affordance; rendering both produced
  // duplicate "Custom / Self-Hosted" chips/options.
  const orderedProviderKeys = orderProviders(
    Object.keys(providers).filter((k) => k !== 'custom'),
    locale,
  );

  function handleProviderChange(key: string) {
    setProvider(key);
    setModel('');
    setModelInputValue('');
    setFreetextPinned(false);
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

  // Mirror modelSource into a ref for the async fetch closure below: a failed
  // refresh must not downgrade a list an earlier fetch already loaded. Synced
  // in an effect (not during render) so the write happens on commit — the
  // closure that reads it runs long after, when the request settles.
  const modelSourceRef = useRef(modelSource);
  useEffect(() => {
    modelSourceRef.current = modelSource;
  }, [modelSource]);

  // A stored key is never round-tripped through the browser. When one exists
  // the request names the card (`card_id`) and the daemon resolves the key
  // from the Keychain; a freshly typed key goes in the body instead.
  const storedKeyCardId = card?.key_set ? card.id : undefined;

  // Fetch models when provider + endpoint (+ key, where one is needed)
  useEffect(() => {
    if (!initialized) return;
    if (!provider) return;
    // Respect an explicit free-text choice: don't yank the user back into a
    // dropdown by a background fetch.
    if (freetextPinned) return;

    const isCustom = provider === CUSTOM_PROVIDER_KEY;
    const info = isCustom ? undefined : providers[provider];
    if (!isCustom && !info) return;
    // The registry already records who serves a model list. z.ai has no
    // /v1/models at all (nginx 404), and MiniMax/Qwen are flagged too — honor
    // the flag instead of paying a round trip that can only fail.
    if (info && info.supports_model_list === false) return;
    // Custom / self-hosted is reachable only through a typed endpoint.
    if (isCustom && !baseUrl.trim()) return;

    // Need a key typed in, or one already stored (global or project). Local
    // servers reached through Custom (LM Studio / Ollama) need none.
    const hasKey = apiKey.trim() || storedKeyCardId || globalSettings?.api_key_set;
    if (!hasKey && !isCustom) return;

    // Do NOT clear the catalog-seeded options here — keep the suggested dropdown
    // usable while the live fetch is in flight. A success upgrades to 'api'; a
    // failure re-seeds the suggested list (or free-text for empty catalogs).
    let cancelled = false;
    const controller = new AbortController();
    setModelsLoading(true);

    async function fetchModels() {
      try {
        const body: Record<string, unknown> = {
          provider: isCustom ? 'custom' : provider,
          base_url: baseUrl.trim() || undefined,
          // The Custom provider picks its own protocol, and the registry's
          // `custom` entry always says "openai" — the backend can't infer it.
          sdk,
        };
        if (apiKey.trim()) {
          body.api_key = apiKey.trim();
        } else if (storedKeyCardId) {
          // Spec 082 §3.8: card_id is the alternative to a raw key — the
          // daemon reads provider/endpoint/sdk/key off the card.
          body.card_id = storedKeyCardId;
        }
        // Backend returns { models: string[] }.
        const result = await api<{ models?: string[] }>('/api/v2/providers/models', {
          method: 'POST',
          body: JSON.stringify(body),
          signal: controller.signal,
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
        // A failed refresh must never destroy a live list an earlier fetch
        // loaded — that downgrade was the visible "loads, then snaps back to
        // the built-in defaults".
        if (modelSourceRef.current === 'api') return;
        // Otherwise fall back to the catalog's suggested models (free-text if none).
        seedModelsFromCatalog(provider);
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    }

    const timer = setTimeout(fetchModels, 400);
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey, provider, providers, initialized, baseUrl, sdk, freetextPinned, storedKeyCardId]);

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
    setFreetextPinned(true);
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
      } else if (storedKeyCardId) {
        // Test the card being edited against the fields on screen: card_id
        // supplies the key, the body supplies the model the user just picked.
        body.card_id = storedKeyCardId;
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

  // Legacy global-slot key removal — reachable only from the wizard mount,
  // where there is no card to delete instead. Inside the card modal the
  // affordance is Delete card (the shim would hit the DEFAULT card, which is
  // not necessarily the one on screen).
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

  // TokenDance one-click key provisioning (Spec 47 Tier 2). Blocking POST:
  // the daemon opens the system browser for consent (up to ~5 min) and the
  // response arrives only after the key is minted, validated, and stored.
  const [tokendanceSigninBusy, setTokendanceSigninBusy] = useState(false);
  const [tokendanceSigninMsg, setTokendanceSigninMsg] =
    useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  // Post-signin collapsed summary (wizard mount only): after a successful
  // one-click signin the full form folds into a "ready to use" card so the
  // user sees the working defaults instead of a wall of fields.
  const [postSigninCollapsed, setPostSigninCollapsed] = useState(false);

  async function handleTokendanceSignin() {
    setTokendanceSigninBusy(true);
    setTokendanceSigninMsg(null);
    try {
      // Spec 082 §3.6: signin now provisions a CARD. It sets the default only
      // when there isn't one, and it never touches another card's key — so
      // the follow-up `PUT /settings` this used to do (which wrote the global
      // slot, and would now write whichever card happens to be default) is
      // gone. Re-connecting an existing TokenDance card names it by id so the
      // fresh key lands on that card and every project pointing at it follows.
      const res = await api<TokendanceSigninResponse>(
        '/api/v2/providers/tokendance/signin',
        {
          method: 'POST',
          body: JSON.stringify(
            card?.provider === 'tokendance' ? { card_id: card.id } : {},
          ),
        },
      );
      setApiKeyStatus({ configured: true, source: 'keyring' });
      setApiKey('');
      // Values come from the response, never read back from state set inside
      // this handler (React 19 batching).
      const info = providers['tokendance'];
      const chosenModel =
        res.card?.model || model.trim() || modelInputValue.trim() || info?.default_model || '';
      setGlobalSettings(prev => ({
        api_key_set: true,
        api_key_masked: res.api_key_masked,
        base_url: res.card?.base_url ?? prev?.base_url ?? info?.base_url ?? null,
        model: chosenModel,
        sdk: res.card?.sdk ?? prev?.sdk ?? info?.sdk ?? 'openai',
        provider: 'tokendance',
      }));
      if (chosenModel) {
        setModel(chosenModel);
        setModelInputValue(chosenModel);
      }
      if (providerPicker === 'cards') setPostSigninCollapsed(true);
      if (res.card) {
        onCardSaved?.({ card: res.card, test: res.test ?? null });
      }
      setTokendanceSigninMsg({ kind: 'ok', text: t('llm.tokendance.signin.success') });
    } catch (err: unknown) {
      setTokendanceSigninMsg({
        kind: 'err',
        text: err instanceof Error && err.message
          ? err.message
          : t('llm.tokendance.signin.failed'),
      });
    } finally {
      setTokendanceSigninBusy(false);
    }
  }

  /**
   * Save = create or update a credential card, and the daemon runs the
   * connection test as part of it (spec 082 D9). A failed test is NOT a save
   * failure: the card is stored either way and the result is surfaced here in
   * green or red, so a provider outage can never block saving.
   */
  async function doSave(): Promise<boolean> {
    setSaving(true);
    setSaved(false);
    setSaveError('');
    setSaveTest(null);
    try {
      const isCustom = provider === CUSTOM_PROVIDER_KEY;
      const info = isCustom ? undefined : providers[provider];
      const typedBaseUrl = baseUrl.trim();
      const body: Record<string, unknown> = {
        provider: isCustom ? 'custom' : provider,
        region,
        model: model.trim() || modelInputValue.trim(),
        sdk,
      };
      // A registry provider's endpoint comes from provider + region; only a
      // deliberate deviation (a proxy typed under Advanced) is worth storing
      // on the card, and Custom always carries its own.
      if (isCustom) {
        body.base_url = typedBaseUrl;
      } else if (typedBaseUrl && info && typedBaseUrl !== resolveBaseUrl(info, region)) {
        body.base_url = typedBaseUrl;
      }
      if (apiKey.trim()) body.api_key = apiKey.trim();
      // Only when edited: an untouched name stays auto-generated, so it
      // keeps following provider/model the way the daemon intends.
      if (cardName.trim() && cardName.trim() !== (card?.name ?? '')) {
        body.name = cardName.trim();
      }

      const res = card
        ? await api<CardMutationResponse>(
            `/api/v2/settings/cards/${encodeURIComponent(card.id)}`,
            { method: 'PUT', body: JSON.stringify(body) },
          )
        : await api<CardMutationResponse>('/api/v2/settings/cards', {
            method: 'POST',
            body: JSON.stringify(body),
          });

      setSaveTest(res.test ?? null);
      if (apiKey.trim()) setApiKeyStatus({ configured: true, source: 'keyring' });
      setApiKey('');
      setSaved(true);
      onCardSaved?.(res);
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

  // What the "current key" line shows: the edited card's own mask, or — on
  // the card-less (wizard) mount — the default card's, via the derived `llm`
  // block. Never a raw key: the daemon only ever serves the mask.
  const storedKeyMasked = card
    ? (card.key_set ? card.key_masked : '')
    : (globalSettings?.api_key_set ? globalSettings.api_key_masked : '');
  const storedKeySource = card ? card.key_source : apiKeyStatus?.source;

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

  // ---- Card form header ----
  // Inside the Add/Edit modal the dialog title already names the surface, so
  // the section heading is only drawn for a standalone mount (the wizard).
  const globalHeader = !card ? (
    <div className="mb-4">
      <h2 className="text-sm font-semibold text-primary mb-1">{t('llm.global.heading')}</h2>
      <HintDetails>
        <p className="text-xs text-secondary">
          {t('llm.global.subhead')}
        </p>
      </HintDetails>
    </div>
  ) : null;

  // ---- Shared fields JSX ----
  const fieldsJSX = (
    <div className="space-y-5">
      {/* Name — only when editing a saved card. A new card is named after its
          provider and model by the daemon, which beats anything typed before
          those are chosen. */}
      {card && (
        <div>
          <label htmlFor="card-name-field" className="block text-sm font-medium text-primary mb-1.5">
            {t('llm.field.name')}
          </label>
          <input
            id="card-name-field"
            type="text"
            value={cardName}
            data-testid="card-name-field"
            onChange={(e) => setCardName(e.target.value)}
            placeholder={card.name}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          />
        </div>
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
          <Select
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
          >
            {orderedProviderKeys.map((key) => (
              <option key={key} value={key}>{providers[key].display_name}</option>
            ))}
            <option value={CUSTOM_PROVIDER_KEY}>{t('llm.provider.custom')}</option>
          </Select>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.notes && (
          <HintDetails className="mt-1">
            <p className="text-xs text-secondary">{providers[provider].notes}</p>
          </HintDetails>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.no_china_endpoint && (
          <p className="text-xs text-secondary/70 mt-1">{t('llm.provider.noChinaEndpoint')}</p>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.china_only && (
          <p className="text-xs text-secondary/70 mt-1">{t('llm.provider.chinaOnly')}</p>
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
        {provider === 'tokendance' && mode === 'global' && (
          <div className="mb-2">
            {!tokendanceKeyActive ? (
              <>
                <button
                  type="button"
                  data-testid="tokendance-signin"
                  onClick={handleTokendanceSignin}
                  disabled={tokendanceSigninBusy}
                  className="inline-flex items-center gap-2 text-sm font-medium rounded-lg px-4 py-2 bg-card text-primary border border-border hover:border-[#8bdc7e] hover:shadow-sm disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-150"
                >
                  {tokendanceSigninBusy ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <img
                      src={watchaLogo}
                      alt=""
                      className="w-5 h-5 rounded-full"
                      aria-hidden="true"
                    />
                  )}
                  {tokendanceSigninBusy
                    ? t('llm.tokendance.signin.waiting')
                    : t('llm.tokendance.signin.button')}
                </button>
                <p className="text-xs text-secondary/70 mt-1" data-testid="tokendance-sponsor">
                  {t('llm.tokendance.sponsor')}
                </p>
              </>
            ) : (
              // A minted key can expire (user-chosen expiry at consent) or be
              // revoked in their console; the retired big button left no way
              // back to one-click. Same handler — the signin flow overwrites
              // the stored key, so re-running it IS the re-provision path.
              <button
                type="button"
                data-testid="tokendance-reconnect"
                onClick={handleTokendanceSignin}
                disabled={tokendanceSigninBusy}
                className="text-xs text-accent hover:underline disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {tokendanceSigninBusy
                  ? t('llm.tokendance.signin.waiting')
                  : t('llm.tokendance.signin.reconnect')}
              </button>
            )}
            {tokendanceSigninBusy && (
              <p className="text-xs text-secondary mt-1">{t('llm.tokendance.signin.hint')}</p>
            )}
            {tokendanceSigninMsg && (
              <p
                data-testid="tokendance-signin-msg"
                className={`text-xs mt-1 ${tokendanceSigninMsg.kind === 'ok' ? 'text-success' : 'text-error'}`}
              >
                {tokendanceSigninMsg.text}
              </p>
            )}
          </div>
        )}
        {provider !== CUSTOM_PROVIDER_KEY && providers[provider]?.console_url && (
          <HintDetails className="mb-1.5">
            <a
              href={providers[provider].console_url!}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              {t('llm.apiKey.getKey')}
            </a>
            <p className="text-xs text-secondary/70 mt-0.5">{t('llm.apiKey.howTo')}</p>
          </HintDetails>
        )}
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          disabled={card?.read_only}
          placeholder={
            storedKeyMasked
              ? t('llm.apiKey.replacePlaceholder')
              : t('llm.apiKey.placeholder')
          }
          className="w-full text-sm font-mono bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 disabled:opacity-50"
        />
        {card?.read_only && (
          <p className="text-xs text-secondary/70 mt-1" data-testid="card-readonly-note">
            {t('cards.env.readOnly')}
          </p>
        )}
        {storedKeyMasked && !apiKey && (
          <div className="flex items-center gap-3 mt-1">
            <p className="text-xs text-secondary">
              {t('llm.apiKey.current', { masked: storedKeyMasked })}
              {storedKeySource && storedKeySource !== 'none' && (
                <span className="ml-1 text-secondary/60">({storedKeySource})</span>
              )}
            </p>
            {/* Only the legacy (card-less) mount offers key removal: inside
                the card modal the affordance is Delete card, and the shim
                would hit the DEFAULT card rather than the one on screen. */}
            {!card && (
              <button
                type="button"
                onClick={handleDeleteApiKey}
                className="text-xs text-error hover:text-error/80 transition-colors"
              >
                {t('llm.apiKey.remove')}
              </button>
            )}
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
            {modelSource !== 'none' && (
              <HintDetails className="mt-1">
                <p className="text-xs text-secondary">
                  {modelSource === 'api' && t('llm.model.source.api')}
                  {modelSource === 'suggested' && t('llm.model.source.suggested')}
                  {modelSource === 'freetext' && t('llm.model.source.freetext')}
                </p>
              </HintDetails>
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
                <Select
                  value={sdk}
                  onChange={(e) => setSdk(e.target.value as 'openai' | 'anthropic')}
                  className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-accent transition-all duration-150"
                >
                  <option value="openai">{t('llm.sdk.openai')}</option>
                  <option value="anthropic">{t('llm.sdk.anthropic')}</option>
                </Select>
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

      {/* Save = create/update the card, with the connection test (spec 082 §3.2) */}
      {!hideSaveButton && (
        <div className="pt-2">
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={saving || card?.read_only}
              data-testid="card-save"
              className="inline-flex items-center gap-2 bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px] max-md:justify-center"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {t('cards.form.saving')}
                </>
              ) : (
                t('settings.save')
              )}
            </button>
            {saved && !saveTest && (
              <span className="flex items-center gap-1 text-sm text-success">
                <Check className="w-4 h-4" />
                {t('settings.saved')}
              </span>
            )}
            {saveError && (
              <span className="text-sm text-error" data-testid="card-save-error">
                {saveError}
              </span>
            )}
          </div>
          {/* The save-test verdict, shown inline BEFORE the modal closes. A
              failed test is red but the card is saved either way (D9), so the
              line says both things rather than reading as a lost edit. */}
          {saveTest && (
            <p
              data-testid="card-save-test"
              className={`flex items-start gap-1.5 text-sm mt-2 ${
                saveTest.ok ? 'text-success' : 'text-error'
              }`}
            >
              {saveTest.ok ? (
                <Check className="w-4 h-4 shrink-0 mt-0.5" />
              ) : (
                <X className="w-4 h-4 shrink-0 mt-0.5" />
              )}
              <span>
                {saveTest.ok
                  ? saveTest.message || t('cards.save.verified')
                  : t('cards.save.savedWithError', {
                      message: saveTest.message || t('cards.health.genericError'),
                    })}
              </span>
            </p>
          )}
        </div>
      )}
    </div>
  );

  // ---- Post-signin summary (wizard mount) ----
  // After one-click provisioning the form folds into a "ready to use" card:
  // the defaults are already persisted, so the user only needs to know it
  // works — and where to click if they want something different.
  const summaryModel = globalSettings?.model || '';
  const summaryModelLabel =
    providers['tokendance']?.models?.[summaryModel]?.display_name || summaryModel;
  const signinSummaryJSX = (
    <div
      data-testid="tokendance-signin-summary"
      className="border border-success/30 bg-success/5 rounded-lg p-4"
    >
      <div className="flex items-start gap-3">
        <img
          src={watchaLogo}
          alt=""
          className="w-8 h-8 rounded-full shrink-0"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-sm font-medium text-primary">
            <Check className="w-4 h-4 text-success shrink-0" />
            {t('llm.tokendance.summary.title')}
          </p>
          <p className="text-xs text-secondary mt-1">
            {t('llm.tokendance.summary.body', { model: summaryModelLabel })}
          </p>
          {globalSettings?.api_key_masked && (
            <p className="text-xs text-secondary/70 mt-1">
              {t('llm.apiKey.current', { masked: globalSettings.api_key_masked })}
            </p>
          )}
        </div>
      </div>
      <button
        type="button"
        data-testid="tokendance-summary-change"
        onClick={() => setPostSigninCollapsed(false)}
        className="mt-3 text-xs text-accent hover:underline"
      >
        {t('llm.tokendance.summary.change')}
      </button>
    </div>
  );

  // ---- Render ----
  return (
    <form onSubmit={handleGlobalSave}>
      {globalHeader}
      {postSigninCollapsed ? signinSummaryJSX : fieldsJSX}
    </form>
  );
}

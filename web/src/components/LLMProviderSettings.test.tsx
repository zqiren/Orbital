// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * LLMProviderSettings unit tests — Spec 17 Tier 1 (provider onboarding).
 *
 * Covers:
 *  - orderProviders: fixed CN-first order, unknowns after knowns, 'custom'
 *    (registry key and the CUSTOM_PROVIDER_KEY sentinel) always last;
 *    Spec 47 locale exception: tokendance top for zh, after the
 *    international block for en;
 *  - regionDefaultForProvider: pure region-default logic;
 *  - default provider resolution: fresh install -> DeepSeek (never Custom),
 *    saved provider wins, unknown-saved provider -> Custom;
 *  - region toggle defaults to China when a dual-endpoint provider is newly
 *    selected via the dropdown;
 *  - the no-China-endpoint caption and the "Get your API key" console link
 *    render for the right providers and are hidden for Custom;
 *  - the wizard-only preset-card picker renders chips in the fixed order and
 *    selects exactly like the dropdown.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { ProviderInfo, ProviderRegistry } from '../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import LLMProviderSettings, {
  orderProviders,
  regionDefaultForProvider,
  CUSTOM_PROVIDER_KEY,
} from './LLMProviderSettings';

afterEach(() => cleanup());

function makeProvider(overrides: Partial<ProviderInfo> = {}): ProviderInfo {
  return {
    display_name: 'Test Provider',
    base_url: 'https://api.test.example/v1',
    supports_model_list: false,
    sdk: 'openai',
    suggested_models: [],
    notes: '',
    ...overrides,
  };
}

const REGISTRY: ProviderRegistry = {
  deepseek: makeProvider({
    display_name: 'DeepSeek',
    base_url: 'https://api.deepseek.com',
    console_url: 'https://platform.deepseek.com/api_keys',
  }),
  moonshot: makeProvider({
    display_name: 'Moonshot (Kimi)',
    base_url: 'https://api.moonshot.ai/v1',
    china_base_url: 'https://api.moonshot.cn/v1',
    console_url: 'https://platform.moonshot.cn/console/api-keys',
  }),
  openai: makeProvider({
    display_name: 'OpenAI',
    base_url: 'https://api.openai.com/v1',
    console_url: 'https://platform.openai.com/api-keys',
    no_china_endpoint: true,
  }),
  anthropic: makeProvider({
    display_name: 'Anthropic',
    base_url: 'https://api.anthropic.com',
    console_url: 'https://console.anthropic.com/settings/keys',
    no_china_endpoint: true,
    sdk: 'anthropic',
  }),
  tokendance: makeProvider({
    display_name: 'TokenDance (词元跳动)',
    base_url: 'https://tokendance.space/gateway/v1',
    console_url: 'https://tokendance.space/keys',
    china_only: true,
    default_model: 'deepseek-v4-flash',
    models: { 'deepseek-v4-flash': { display_name: 'DeepSeek V4 Flash' } },
  }),
  // Production registry shape: GET /api/v2/providers serves a literal 'custom'
  // entry alongside the CUSTOM_PROVIDER_KEY sentinel the picker adds itself.
  custom: makeProvider({
    display_name: 'Custom / Self-Hosted',
    base_url: '',
  }),
};

interface MockSettings {
  api_key_set?: boolean;
  api_key_masked?: string;
  base_url?: string | null;
  model?: string | null;
  sdk?: string;
  provider?: string;
}

function mockApi(opts: { settings?: MockSettings; providers?: ProviderRegistry } = {}) {
  apiMock.mockImplementation(async (path: string) => {
    if (path === '/api/v2/settings') {
      return {
        llm: {
          api_key_set: false,
          api_key_masked: '',
          base_url: null,
          model: null,
          sdk: 'openai',
          provider: '',
          ...opts.settings,
        },
      };
    }
    if (path === '/api/v2/providers') {
      return opts.providers ?? REGISTRY;
    }
    if (path === '/api/v2/settings/api-key/status') {
      return { configured: false, source: 'none' };
    }
    return {};
  });
}

beforeEach(() => {
  apiMock.mockReset();
});

// ---- orderProviders ----

describe('orderProviders', () => {
  it('applies the fixed CN-first order for known providers', () => {
    const scrambled = [
      'openai', 'anthropic', 'deepseek', 'moonshot', 'zhipu', 'qwen',
      'minimax', 'google', 'xai', 'mistral', 'groq', 'together', 'openrouter',
    ];
    expect(orderProviders(scrambled)).toEqual([
      'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax',
      'openai', 'anthropic', 'google', 'xai', 'mistral', 'groq', 'together', 'openrouter',
    ]);
  });

  it('only includes keys present in the input, in fixed order', () => {
    expect(orderProviders(['xai', 'deepseek', 'openai'])).toEqual(['deepseek', 'openai', 'xai']);
  });

  it('places unknown registry keys after the known ones, in original relative order', () => {
    const result = orderProviders(['openai', 'some_new_provider', 'deepseek', 'another_one']);
    expect(result).toEqual(['deepseek', 'openai', 'some_new_provider', 'another_one']);
  });

  it('always places a literal "custom" registry key last', () => {
    expect(orderProviders(['custom', 'openai', 'deepseek'])).toEqual(['deepseek', 'openai', 'custom']);
  });

  it('always places the CUSTOM_PROVIDER_KEY sentinel last', () => {
    expect(orderProviders([CUSTOM_PROVIDER_KEY, 'openai', 'deepseek'])).toEqual([
      'deepseek', 'openai', CUSTOM_PROVIDER_KEY,
    ]);
  });

  // Spec 47: the single locale-aware exception for the China-only router.
  it('en locale (default) sorts tokendance after the international block', () => {
    expect(orderProviders(['tokendance', 'openai', 'deepseek', 'openrouter'])).toEqual([
      'deepseek', 'openai', 'openrouter', 'tokendance',
    ]);
  });

  it('zh locale sorts tokendance to the top', () => {
    expect(orderProviders(['openai', 'deepseek', 'tokendance'], 'zh')).toEqual([
      'tokendance', 'deepseek', 'openai',
    ]);
  });

  it('tokendance is a known key in en order: before unknowns, custom still last', () => {
    expect(orderProviders(['custom', 'brand_new', 'tokendance', 'deepseek'])).toEqual([
      'deepseek', 'tokendance', 'brand_new', 'custom',
    ]);
  });
});

// ---- regionDefaultForProvider ----

describe('regionDefaultForProvider', () => {
  it('returns china for a provider with a china_base_url', () => {
    expect(regionDefaultForProvider({ china_base_url: 'https://api.moonshot.cn/v1' })).toBe('china');
  });

  it('returns global for a provider without one', () => {
    expect(regionDefaultForProvider({ china_base_url: undefined })).toBe('global');
    expect(regionDefaultForProvider({ china_base_url: null })).toBe('global');
  });

  it('returns global when info is undefined', () => {
    expect(regionDefaultForProvider(undefined)).toBe('global');
  });
});

// ---- Default provider resolution ----

describe('LLMProviderSettings — default provider resolution (Spec 17 §9)', () => {
  it('fresh install (no saved provider) defaults to DeepSeek, never Custom', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('deepseek');
    });
  });

  it('a saved, known provider wins over the default', async () => {
    mockApi({ settings: { provider: 'openai', base_url: 'https://api.openai.com/v1' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('openai');
    });
  });

  it('an unknown saved provider lands on Custom, not the default', async () => {
    mockApi({ settings: { provider: 'legacy-provider-not-in-registry' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe(CUSTOM_PROVIDER_KEY);
    });
  });

  // Final-review Finding 2: when globalSettings exists but has no saved
  // provider/base_url (fresh install), the default-provider branch pre-selects
  // DeepSeek — but base_url must seed from DeepSeek's registry base_url, not
  // fall through to ''. An empty string is a legitimate "inherit from global"
  // sentinel in *project* mode only; in global mode it gets saved as-is and
  // passed raw to the backend's OpenAI client, breaking the request.
  it('seeds the Advanced base URL field from the default provider, not empty, on a fresh install', async () => {
    mockApi({ settings: { provider: '', base_url: null } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('deepseek');
    });

    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));

    const baseUrlInput = await screen.findByDisplayValue(REGISTRY.deepseek.base_url!);
    expect(baseUrlInput).toBeTruthy();
  });
});

// ---- Region defaults to China on new selection ----

describe('LLMProviderSettings — region defaults to China on new selection (Spec 17 §9)', () => {
  it('defaults the region toggle to China when switching to a dual-endpoint provider', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" />);
    const select = (await screen.findAllByRole('combobox'))[0] as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('deepseek'));

    fireEvent.change(select, { target: { value: 'moonshot' } });

    const chinaBtn = await screen.findByText('China');
    const globalBtn = screen.getByText('Global');
    expect(chinaBtn.className).toContain('bg-accent');
    expect(globalBtn.className).not.toContain('bg-accent');
  });

  it('does not show a region toggle for a provider with no china_base_url', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" />);
    const select = (await screen.findAllByRole('combobox'))[0] as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('deepseek'));

    fireEvent.change(select, { target: { value: 'openai' } });
    expect(screen.queryByText('China')).toBeNull();
  });

  it('leaves saved-config region derivation untouched (china base_url -> China toggle on load)', async () => {
    mockApi({
      settings: { provider: 'moonshot', base_url: 'https://api.moonshot.cn/v1' },
    });
    render(<LLMProviderSettings mode="global" />);
    const chinaBtn = await screen.findByText('China');
    expect(chinaBtn.className).toContain('bg-accent');
  });
});

// ---- Caption + console link ----

describe('LLMProviderSettings — no-China-endpoint caption + "Get your API key" link', () => {
  it('shows the caption for a no_china_endpoint provider (OpenAI)', async () => {
    mockApi({ settings: { provider: 'openai', base_url: 'https://api.openai.com/v1' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() =>
      expect(
        screen.getByText('No mainland-China endpoint — requires global network access.'),
      ).toBeTruthy(),
    );
  });

  it('shows the China-mainland-only caption for a china_only provider (TokenDance)', async () => {
    mockApi({
      settings: { provider: 'tokendance', base_url: 'https://tokendance.space/gateway/v1' },
    });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() =>
      expect(
        screen.getByText('China mainland only — may be unreachable outside mainland China.'),
      ).toBeTruthy(),
    );
  });

  it('hides the caption for a provider without the flag (DeepSeek)', async () => {
    mockApi({ settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('deepseek');
    });
    expect(
      screen.queryByText('No mainland-China endpoint — requires global network access.'),
    ).toBeNull();
  });

  it('renders the "Get your API key" link with the provider console_url', async () => {
    mockApi({ settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' } });
    render(<LLMProviderSettings mode="global" />);
    const link = await screen.findByText('Get your API key ↗') as HTMLAnchorElement;
    expect(link.getAttribute('href')).toBe('https://platform.deepseek.com/api_keys');
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
    expect(screen.getByText(/Create an account/)).toBeTruthy();
  });

  it('hides the console link and caption entirely for Custom', async () => {
    mockApi({ settings: { provider: 'legacy-provider-not-in-registry' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe(CUSTOM_PROVIDER_KEY);
    });
    expect(screen.queryByText('Get your API key ↗')).toBeNull();
    expect(
      screen.queryByText('No mainland-China endpoint — requires global network access.'),
    ).toBeNull();
  });
});

// ---- Wizard-only preset cards ----

describe('LLMProviderSettings — preset cards (providerPicker="cards", wizard-only)', () => {
  it('renders provider chips in the fixed order instead of a <select>, and selects on click', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" hideSaveButton providerPicker="cards" />);
    await waitFor(() => expect(screen.getByText('DeepSeek')).toBeTruthy());

    expect(screen.queryByRole('combobox')).toBeNull();

    const openaiChip = screen.getByText('OpenAI');
    expect(openaiChip.className).not.toContain('bg-accent');
    fireEvent.click(openaiChip);
    await waitFor(() => expect(openaiChip.className).toContain('bg-accent'));
  });

  it('defaults to the dropdown picker when the prop is omitted', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getAllByRole('combobox')[0]).toBeTruthy());
  });
});

// ---- Duplicate Custom entry (registry literal 'custom' vs. the sentinel) ----
//
// GET /api/v2/providers serves a literal 'custom' registry entry (display_name
// "Custom / Self-Hosted"), matching the REGISTRY fixture above. Without the
// fix, that entry rendered alongside the CUSTOM_PROVIDER_KEY sentinel option,
// producing two identical "Custom / Self-Hosted" chips/options.

describe('LLMProviderSettings — no duplicate Custom entry', () => {
  it('renders exactly one Custom option in dropdown mode', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('deepseek');
    });
    expect(screen.getAllByText('Custom / Self-Hosted')).toHaveLength(1);
  });

  it('renders exactly one Custom chip in cards mode', async () => {
    mockApi({ settings: { provider: '' } });
    render(<LLMProviderSettings mode="global" hideSaveButton providerPicker="cards" />);
    await waitFor(() => expect(screen.getByText('DeepSeek')).toBeTruthy());
    expect(screen.getAllByText('Custom / Self-Hosted')).toHaveLength(1);
  });

  it('a saved provider="custom" resolves to the sentinel Custom, shows Advanced, and keeps the saved base_url', async () => {
    mockApi({
      settings: {
        provider: 'custom',
        base_url: 'https://my-llm.example/v1',
        model: 'my-model',
      },
    });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe(CUSTOM_PROVIDER_KEY);
    });
    // Only one Custom option ever renders (the sentinel) — same fixed-fixture
    // assertion as above, re-checked against the saved-custom-config path.
    expect(screen.getAllByText('Custom / Self-Hosted')).toHaveLength(1);
    // Advanced is expanded (as the pre-existing unknown-provider branch does),
    // and the saved base_url is preserved rather than cleared.
    expect(screen.getByDisplayValue('https://my-llm.example/v1')).toBeTruthy();
  });
});

// ---- wizard mode gating (backlog #25: create-project modal shows only the
// no-api-key warning — the loading state and the happy-path "using global
// defaults" card render nothing) ----

describe('LLMProviderSettings — wizard mode gating (backlog #25)', () => {
  it('renders nothing while global settings are still loading', () => {
    // Never-resolving api call keeps globalLoaded=false for the assertion window.
    apiMock.mockImplementation(() => new Promise(() => {}));
    const { container } = render(<LLMProviderSettings mode="wizard" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing (no happy-path card) once a global API key is configured', async () => {
    mockApi({ settings: { api_key_set: true, provider: 'deepseek', model: 'deepseek-chat' } });
    const { container } = render(<LLMProviderSettings mode="wizard" />);
    await waitFor(() => expect(apiMock).toHaveBeenCalledWith('/api/v2/settings'));
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText(/using global defaults/i)).toBeNull();
  });

  it('renders only the not-configured warning when no global API key is set', async () => {
    mockApi({ settings: { api_key_set: false } });
    render(<LLMProviderSettings mode="wizard" />);
    await waitFor(() => {
      expect(screen.getByText(/No LLM provider configured yet/)).toBeTruthy();
    });
  });
});

// ---- Collapsed hint paragraphs (bug #49) ----
//
// The long explanatory paragraphs now sit inside native <details> closed by
// default; the short china-only / no-China-endpoint safety captions stay
// always-visible. jsdom keeps <details> children in the DOM either way, so
// these assert the wrapper and its `open` state rather than text presence.

describe('LLMProviderSettings — hint paragraphs collapse by default (bug #49)', () => {
  function closedDetailsFor(text: string | RegExp): HTMLDetailsElement {
    const details = screen.getByText(text).closest('details');
    expect(details).toBeTruthy();
    expect((details as HTMLDetailsElement).open).toBe(false);
    return details as HTMLDetailsElement;
  }

  it('collapses the global subhead, the API-key how-to and the model-source explainer', async () => {
    mockApi({ settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getByText('Get your API key ↗')).toBeTruthy());

    closedDetailsFor('Used by all projects unless overridden in project settings.');
    closedDetailsFor(/Create an account/);
    // Empty suggested_models in the fixture -> modelSource 'freetext'.
    closedDetailsFor('Enter the model identifier to use with this provider.');

    // Every collapsed block is labelled with the shared summary string.
    expect(screen.getAllByText('Details').length).toBeGreaterThanOrEqual(3);
  });

  it('collapses the provider notes blurb', async () => {
    mockApi({
      settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' },
      providers: {
        ...REGISTRY,
        deepseek: makeProvider({
          display_name: 'DeepSeek',
          base_url: 'https://api.deepseek.com',
          notes: 'Cheap and fast; reasoning model available.',
        }),
      },
    });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() =>
      expect(screen.getByText('Cheap and fast; reasoning model available.')).toBeTruthy(),
    );
    closedDetailsFor('Cheap and fast; reasoning model available.');
  });

  it('keeps the region-availability warnings outside the disclosure', async () => {
    mockApi({ settings: { provider: 'openai', base_url: 'https://api.openai.com/v1' } });
    render(<LLMProviderSettings mode="global" />);
    const caption = await screen.findByText(
      'No mainland-China endpoint — requires global network access.',
    );
    expect(caption.closest('details')).toBeNull();
  });

  it('opens on demand, revealing the hint text', async () => {
    mockApi({ settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getByText('Get your API key ↗')).toBeTruthy());

    const details = closedDetailsFor(/Create an account/);
    fireEvent.click(details.querySelector('summary')!);
    expect(details.open).toBe(true);
  });
});

// ---- Spec 47 Tier 2: TokenDance one-click signin ----

describe('LLMProviderSettings — TokenDance one-click signin (Spec 47 Tier 2)', () => {
  const TD_SETTINGS = {
    provider: 'tokendance',
    base_url: 'https://tokendance.space/gateway/v1',
  };

  /** The TokenDance card the daemon provisions (spec 082 §3.6): model comes
   *  from the registry's default_model, never from the form. */
  const TD_CARD = {
    id: 'card_td',
    name: 'TokenDance (词元跳动) · deepseek-v4-flash',
    provider: 'tokendance',
    region: 'global' as const,
    base_url: null,
    sdk: null,
    model: 'deepseek-v4-flash',
    created_at: '2026-09-04T10:00:00+00:00',
    verified_at: '2026-09-04T10:00:01+00:00',
    last_used_at: null,
    last_error: null,
    key_set: true,
    key_masked: 'sk-t...9876',
    key_source: 'keychain' as const,
    is_default: true,
    read_only: false,
  };

  function mockApiWithSignin(
    signinImpl: () => Promise<unknown>,
    { keySet = false }: { keySet?: boolean } = {},
  ) {
    apiMock.mockImplementation(async (path: string, opts?: RequestInit) => {
      if (path === '/api/v2/providers/tokendance/signin' && opts?.method === 'POST') {
        return signinImpl();
      }
      if (path === '/api/v2/settings') {
        return {
          llm: {
            api_key_set: keySet,
            api_key_masked: keySet ? 'sk-t...1111' : '',
            model: null,
            sdk: 'openai',
            ...TD_SETTINGS,
          },
        };
      }
      if (path === '/api/v2/providers') return REGISTRY;
      if (path === '/api/v2/settings/api-key/status') {
        return keySet
          ? { configured: true, source: 'keyring' }
          : { configured: false, source: 'none' };
      }
      return {};
    });
  }

  /** Any legacy `PUT /settings` write. Spec 082 retired it from this flow:
   *  it wrote the single global slot, and would now write whichever card
   *  happens to be default — not necessarily the TokenDance one. */
  function settingsPutBody(): Record<string, unknown> | null {
    const call = apiMock.mock.calls.find(
      ([path, opts]) => path === '/api/v2/settings' && (opts as RequestInit)?.method === 'PUT',
    );
    return call ? JSON.parse(String((call[1] as RequestInit).body)) : null;
  }

  it('renders the signin button for tokendance in global mode only', async () => {
    mockApi({ settings: TD_SETTINGS });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getByTestId('tokendance-signin')).toBeTruthy());
  });

  it('does not render the button for other providers', async () => {
    mockApi({ settings: { provider: 'deepseek', base_url: 'https://api.deepseek.com' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('deepseek');
    });
    expect(screen.queryByTestId('tokendance-signin')).toBeNull();
  });

  it('click → POST, busy state disables the button, success persists defaults and retires the button', async () => {
    let release!: (v: unknown) => void;
    mockApiWithSignin(() => new Promise((res) => { release = res; }));
    render(<LLMProviderSettings mode="global" />);
    const btn = await screen.findByTestId('tokendance-signin');

    fireEvent.click(btn);
    // In flight: disabled + browser hint visible.
    await waitFor(() => expect((btn as HTMLButtonElement).disabled).toBe(true));
    expect(screen.getByText(/browser window that just opened/)).toBeTruthy();
    expect(apiMock).toHaveBeenCalledWith(
      '/api/v2/providers/tokendance/signin',
      expect.objectContaining({ method: 'POST' }),
    );

    release({ api_key_set: true, api_key_masked: 'sk-t...9876', card: TD_CARD, test: { ok: true, status: null, code: null, message: 'ok' } });
    await waitFor(() => expect(screen.getByText('API key created and saved.')).toBeTruthy());
    // One tap lands in a usable configuration — but as a CARD now. The
    // provisioning route persists provider/model/endpoint itself, so the
    // legacy global-slot write is gone: making it would overwrite whichever
    // card is default.
    expect(settingsPutBody()).toBeNull();
    expect(
      (screen.getByDisplayValue('deepseek-v4-flash') as HTMLInputElement).value,
    ).toBe('deepseek-v4-flash');
    // The saved TokenDance key retires the one-click button; the stored-key
    // line reflects the mask without a page reload.
    expect(screen.queryByTestId('tokendance-signin')).toBeNull();
    expect(screen.getByText(/sk-t\.\.\.9876/)).toBeTruthy();
  });

  it("the provisioned card's model wins over whatever was typed in the form", async () => {
    mockApiWithSignin(async () => ({
      api_key_set: true,
      api_key_masked: 'sk-t...9876',
      card: TD_CARD,
      test: { ok: true, status: null, code: null, message: 'ok' },
    }));
    render(<LLMProviderSettings mode="global" />);
    const btn = await screen.findByTestId('tokendance-signin');
    // Fixture catalog has no suggested_models → free-text model input.
    fireEvent.change(screen.getByPlaceholderText(/model name/i), {
      target: { value: 'glm-5.2' },
    });
    fireEvent.click(btn);
    await waitFor(() => expect(screen.getByText('API key created and saved.')).toBeTruthy());
    // Spec 082 §3.6: the card the daemon creates carries the registry's
    // default model, and the card is the configuration. Typing a model before
    // one-click no longer changes what gets provisioned — the user edits the
    // card afterwards to change it.
    expect(
      (screen.getByDisplayValue('deepseek-v4-flash') as HTMLInputElement).value,
    ).toBe('deepseek-v4-flash');
  });

  it('re-connect names the card being re-keyed, so no other card is touched', async () => {
    mockApiWithSignin(async () => ({
      api_key_set: true,
      api_key_masked: 'sk-t...9876',
      card: TD_CARD,
      test: { ok: true, status: null, code: null, message: 'ok' },
    }));
    render(
      <LLMProviderSettings
        mode="global"
        card={{ ...TD_CARD, key_masked: 'sk-t...1111' }}
      />,
    );
    fireEvent.click(await screen.findByTestId('tokendance-reconnect'));
    await waitFor(() => expect(screen.getByText('API key created and saved.')).toBeTruthy());
    const call = apiMock.mock.calls.find(
      ([path]) => path === '/api/v2/providers/tokendance/signin',
    );
    expect(JSON.parse(String((call![1] as RequestInit).body))).toEqual({ card_id: 'card_td' });
  });

  it('button is absent when the saved provider is tokendance with a key set', async () => {
    mockApi({ settings: { ...TD_SETTINGS, api_key_set: true, api_key_masked: 'sk-t...1111' } });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('tokendance');
    });
    expect(screen.queryByTestId('tokendance-signin')).toBeNull();
    // …but the low-key re-connect link takes its place: an expired or
    // console-revoked minted key must keep a one-click path back.
    expect(screen.getByTestId('tokendance-reconnect')).toBeTruthy();
  });

  it('re-connect re-runs the signin flow and refreshes the stored key', async () => {
    mockApiWithSignin(
      async () => ({ api_key_set: true, api_key_masked: 'sk-t...9876' }),
      { keySet: true },
    );
    render(<LLMProviderSettings mode="global" />);
    const link = await screen.findByTestId('tokendance-reconnect');
    expect(screen.queryByTestId('tokendance-signin')).toBeNull();
    fireEvent.click(link);
    await waitFor(() => expect(screen.getByText('API key created and saved.')).toBeTruthy());
    expect(apiMock).toHaveBeenCalledWith(
      '/api/v2/providers/tokendance/signin',
      expect.objectContaining({ method: 'POST' }),
    );
    // The fresh key's mask replaces the stale one without a reload.
    expect(screen.getByText(/sk-t\.\.\.9876/)).toBeTruthy();
  });

  it("still offers signin when another provider's key occupies the global slot", async () => {
    mockApi({
      settings: {
        provider: 'deepseek',
        base_url: 'https://api.deepseek.com',
        api_key_set: true,
        api_key_masked: 'sk-d...2222',
      },
    });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getAllByRole('combobox')[0]).toBeTruthy());
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'tokendance' } });
    await waitFor(() => expect(screen.getByTestId('tokendance-signin')).toBeTruthy());
  });

  it('shows the Watcha sponsor caption under the signin button', async () => {
    mockApi({ settings: TD_SETTINGS });
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(screen.getByTestId('tokendance-sponsor')).toBeTruthy());
  });

  it('cards mount folds into the ready-to-use summary after signin; Adjust reopens the form', async () => {
    mockApiWithSignin(async () => ({ api_key_set: true, api_key_masked: 'sk-t...9876' }));
    render(<LLMProviderSettings mode="global" providerPicker="cards" hideSaveButton />);
    fireEvent.click(await screen.findByTestId('tokendance-signin'));

    const summary = await screen.findByTestId('tokendance-signin-summary');
    // Shows the working default by its catalog display name + the masked key.
    expect(summary.textContent).toContain('DeepSeek V4 Flash');
    expect(summary.textContent).toContain('sk-t...9876');
    // The form (and its signin button) is folded away.
    expect(screen.queryByTestId('tokendance-signin')).toBeNull();

    fireEvent.click(screen.getByTestId('tokendance-summary-change'));
    await waitFor(() =>
      expect(screen.queryByTestId('tokendance-signin-summary')).toBeNull(),
    );
    // Reopened form is populated with the persisted default model.
    expect(
      (screen.getByDisplayValue('deepseek-v4-flash') as HTMLInputElement).value,
    ).toBe('deepseek-v4-flash');
  });

  it('failure surfaces the backend error message', async () => {
    mockApiWithSignin(() =>
      Promise.reject(new Error('sign-in timed out waiting for the browser redirect')),
    );
    render(<LLMProviderSettings mode="global" />);
    const btn = await screen.findByTestId('tokendance-signin');
    fireEvent.click(btn);
    await waitFor(() =>
      expect(
        screen.getByText('sign-in timed out waiting for the browser redirect'),
      ).toBeTruthy(),
    );
    expect(screen.getByTestId('tokendance-signin-msg').className).toContain('text-error');
  });
});

// ---- Model-list fetch (live /models picker) ----
//
// Regression cluster: the picker loaded a live list and then silently reverted
// to the static suggested list. Root cause was the post-save key clear (the
// backend now falls back to the stored key), but four frontend behaviors kept
// the picker degraded: a failed refresh clobbered a good list, the registry's
// supports_model_list flag was never read, base_url changes never refetched,
// and the Custom provider — the one place a user can't know model ids by
// heart — never fetched at all.

const LIST_REGISTRY: ProviderRegistry = {
  deepseek: makeProvider({
    display_name: 'DeepSeek',
    base_url: 'https://api.deepseek.com',
    supports_model_list: true,
    suggested_models: ['suggested-only'],
  }),
  minimax: makeProvider({
    display_name: 'MiniMax',
    base_url: 'https://api.minimax.io/v1',
    supports_model_list: false,
    suggested_models: ['MiniMax-M3'],
  }),
  custom: makeProvider({ display_name: 'Custom / Self-Hosted', base_url: '' }),
};

/** api() mock that also answers /api/v2/providers/models via `onModels`. */
function mockApiWithModels(
  settings: MockSettings,
  onModels: (body: Record<string, unknown>) => Promise<{ models: string[] }>,
) {
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/settings') {
      return {
        llm: {
          api_key_set: true,
          api_key_masked: 'sk-***',
          base_url: null,
          model: null,
          sdk: 'openai',
          provider: '',
          ...settings,
        },
      };
    }
    if (path === '/api/v2/providers') return LIST_REGISTRY;
    if (path === '/api/v2/settings/api-key/status') {
      return { configured: true, source: 'keyring' };
    }
    if (path === '/api/v2/providers/models') {
      return onModels(JSON.parse((options?.body as string) ?? '{}'));
    }
    return {};
  });
}

const modelCalls = () =>
  apiMock.mock.calls.filter((c) => c[0] === '/api/v2/providers/models');

/** Let the 400ms fetch debounce elapse (plus slack) with real timers. */
const afterDebounce = () => new Promise((r) => setTimeout(r, 550));

describe('LLMProviderSettings — model list fetch', () => {
  it('skips the request entirely when the registry says supports_model_list=false', async () => {
    mockApiWithModels(
      { provider: 'minimax', base_url: 'https://api.minimax.io/v1' },
      async () => ({ models: ['should-never-be-requested'] }),
    );
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('minimax');
    });
    await afterDebounce();
    expect(modelCalls()).toHaveLength(0);
  });

  it('fetches models for the Custom provider, sending its base_url and sdk', async () => {
    mockApiWithModels(
      { provider: 'custom', base_url: 'http://localhost:1234/v1' },
      async () => ({ models: ['local-llama'] }),
    );
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe(CUSTOM_PROVIDER_KEY);
    });
    await waitFor(() => expect(modelCalls().length).toBeGreaterThan(0));
    const body = JSON.parse(modelCalls()[0][1].body as string);
    expect(body.base_url).toBe('http://localhost:1234/v1');
    expect(body.sdk).toBe('openai');
  });

  it('keeps an already-loaded live list when a later refresh fails', async () => {
    let calls = 0;
    mockApiWithModels(
      { provider: 'deepseek', base_url: 'https://api.deepseek.com' },
      async () => {
        calls += 1;
        if (calls === 1) return { models: ['live-a', 'live-b'] };
        throw new Error('401');
      },
    );
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(calls).toBe(1));

    // Force a refetch by editing the endpoint, and make it fail.
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));
    const baseUrlInput = await screen.findByDisplayValue('https://api.deepseek.com');
    fireEvent.change(baseUrlInput, { target: { value: 'https://api.deepseek.com/' } });
    await waitFor(() => expect(calls).toBe(2));

    // The live list must survive; it must NOT be replaced by suggested_models.
    const modelInput = screen.getByPlaceholderText(/model/i);
    fireEvent.focus(modelInput);
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'live-a' })).toBeTruthy();
    });
    expect(screen.queryByRole('button', { name: 'suggested-only' })).toBeNull();
  });

  it('refetches when the base URL changes (region switch / manual edit)', async () => {
    mockApiWithModels(
      { provider: 'deepseek', base_url: 'https://api.deepseek.com' },
      async () => ({ models: ['live-a'] }),
    );
    render(<LLMProviderSettings mode="global" />);
    await waitFor(() => expect(modelCalls().length).toBe(1));

    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }));
    const baseUrlInput = await screen.findByDisplayValue('https://api.deepseek.com');
    fireEvent.change(baseUrlInput, { target: { value: 'https://api.deepseek.cn' } });

    await waitFor(() => expect(modelCalls().length).toBe(2));
    const body = JSON.parse(modelCalls()[1][1].body as string);
    expect(body.base_url).toBe('https://api.deepseek.cn');
  });
});

// ---- Spec 082: the form saves a credential card, test included ----

describe('LLMProviderSettings — card save (spec 082 §3.2)', () => {
  const CARD = {
    id: 'card_or',
    name: 'DeepSeek · deepseek-chat',
    provider: 'deepseek',
    region: 'global' as const,
    base_url: null,
    sdk: null,
    model: 'deepseek-chat',
    created_at: '2026-09-04T10:00:00+00:00',
    verified_at: null,
    last_used_at: null,
    last_error: null,
    key_set: true,
    key_masked: 'sk-o...wxyz',
    key_source: 'keychain' as const,
    is_default: false,
    read_only: false,
  };

  /** Card routes on top of the standard registry/settings mocks. */
  function mockCardApi(
    postResult: () => Promise<unknown> = async () => ({
      card: CARD,
      test: { ok: true, status: null, code: null, message: 'Connected to DeepSeek using x' },
    }),
  ) {
    apiMock.mockImplementation(async (path: string, opts?: RequestInit) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') return postResult();
      if (path.startsWith('/api/v2/settings/cards/') && opts?.method === 'PUT') return postResult();
      if (path === '/api/v2/settings') {
        return {
          llm: {
            api_key_set: false,
            api_key_masked: '',
            base_url: null,
            model: null,
            sdk: 'openai',
            provider: '',
          },
        };
      }
      if (path === '/api/v2/providers') return REGISTRY;
      if (path === '/api/v2/settings/api-key/status') {
        return { configured: false, source: 'none' };
      }
      return {};
    });
  }

  function cardCall(method: string) {
    return apiMock.mock.calls.find(
      ([path, opts]) =>
        String(path).startsWith('/api/v2/settings/cards') &&
        (opts as RequestInit)?.method === method,
    );
  }

  it('Save POSTs a new card with provider, region, model and the typed key', async () => {
    mockCardApi();
    const onCardSaved = vi.fn();
    render(<LLMProviderSettings mode="global" onCardSaved={onCardSaved} />);
    await screen.findByText('API Key');

    fireEvent.change(screen.getByPlaceholderText(/sk-/), { target: { value: 'sk-test-key' } });
    fireEvent.change(screen.getByPlaceholderText(/model name/i), {
      target: { value: 'deepseek-chat' },
    });
    fireEvent.click(screen.getByTestId('card-save'));

    await waitFor(() => expect(cardCall('POST')).toBeTruthy());
    const body = JSON.parse(String((cardCall('POST')![1] as RequestInit).body));
    expect(body).toMatchObject({
      provider: 'deepseek',
      region: 'global',
      model: 'deepseek-chat',
      api_key: 'sk-test-key',
    });
    // A registry provider's endpoint is provider+region, not a copied string —
    // storing it would resurrect the "key from one provider, endpoint from
    // another" pairing the cards exist to prevent.
    expect(body.base_url).toBeUndefined();
    await waitFor(() => expect(onCardSaved).toHaveBeenCalled());
  });

  it('shows the save-test verdict inline on success', async () => {
    mockCardApi();
    render(<LLMProviderSettings mode="global" />);
    await screen.findByText('API Key');
    fireEvent.change(screen.getByPlaceholderText(/model name/i), {
      target: { value: 'deepseek-chat' },
    });
    fireEvent.click(screen.getByTestId('card-save'));
    const verdict = await screen.findByTestId('card-save-test');
    expect(verdict.textContent).toContain('Connected to DeepSeek');
  });

  it('a failed test still saves the card and says so, in red', async () => {
    mockCardApi(async () => ({
      card: { ...CARD, last_error: { status: 401, code: 'invalid_api_key', message: 'Invalid API key', at: '2026-09-04T10:00:00+00:00' } },
      test: { ok: false, status: 401, code: 'invalid_api_key', message: 'Invalid API key' },
    }));
    const onCardSaved = vi.fn();
    render(<LLMProviderSettings mode="global" onCardSaved={onCardSaved} />);
    await screen.findByText('API Key');
    fireEvent.change(screen.getByPlaceholderText(/model name/i), {
      target: { value: 'deepseek-chat' },
    });
    fireEvent.click(screen.getByTestId('card-save'));

    const verdict = await screen.findByTestId('card-save-test');
    // D9: an outage or a bad key must never block saving — the card exists,
    // and the failure is where the user will look for it.
    expect(verdict.textContent).toContain('Saved, but the connection test failed');
    expect(verdict.textContent).toContain('Invalid API key');
    expect(verdict.className).toContain('text-error');
    expect(onCardSaved).toHaveBeenCalled();
    expect(screen.queryByTestId('card-save-error')).toBeNull();
  });

  it('editing an existing card PUTs to that card id and populates from it', async () => {
    mockCardApi();
    // Moonshot, deliberately NOT the fresh-install default provider: the form
    // has to be the CARD's configuration, and 'deepseek' would be the same
    // value either branch produced.
    render(
      <LLMProviderSettings
        mode="global"
        card={{ ...CARD, provider: 'moonshot', model: 'kimi-k2.5' }}
      />,
    );
    await waitFor(() => {
      const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
      expect(select.value).toBe('moonshot');
    });
    expect(((await screen.findByDisplayValue('kimi-k2.5')) as HTMLInputElement).value)
      .toBe('kimi-k2.5');
    // The card's own masked key is shown — never the global slot's.
    expect(screen.getByText(/sk-o\.\.\.wxyz/)).toBeTruthy();

    fireEvent.click(screen.getByTestId('card-save'));
    await waitFor(() => expect(cardCall('PUT')).toBeTruthy());
    expect(String(cardCall('PUT')![0])).toBe('/api/v2/settings/cards/card_or');
  });

  it('the env card is read-only: no key entry, no save', async () => {
    mockCardApi();
    render(
      <LLMProviderSettings
        mode="global"
        card={{ ...CARD, id: 'env', read_only: true, key_source: 'environment' }}
      />,
    );
    await screen.findByTestId('card-readonly-note');
    expect((screen.getByTestId('card-save') as HTMLButtonElement).disabled).toBe(true);
  });

  it('a save failure (not a test failure) surfaces as a save error', async () => {
    mockCardApi(async () => {
      throw new Error('missing model');
    });
    render(<LLMProviderSettings mode="global" />);
    await screen.findByText('API Key');
    fireEvent.change(screen.getByPlaceholderText(/model name/i), {
      target: { value: 'deepseek-chat' },
    });
    fireEvent.click(screen.getByTestId('card-save'));
    expect((await screen.findByTestId('card-save-error')).textContent).toContain('missing model');
    expect(screen.queryByTestId('card-save-test')).toBeNull();
  });
});

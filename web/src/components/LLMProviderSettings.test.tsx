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

// ---- Project mode: saved provider with no endpoint seeds region default ----

describe('LLMProviderSettings — project mode seeds endpoint for saved provider (401 trap)', () => {
  it('seeds the region-default base_url when a project saved a provider but no endpoint', async () => {
    mockApi({});
    const onChange = vi.fn();
    render(
      <LLMProviderSettings
        mode="project"
        projectValues={{ provider: 'moonshot', model: 'kimi-k2.5', sdk: 'openai' }}
        onChange={onChange}
      />,
    );
    // An empty base_url would fall back cross-provider on the backend —
    // the seed must be the provider's region-default endpoint (China for
    // dual-endpoint providers, per Spec 17 §9).
    await waitFor(() => {
      const calls = onChange.mock.calls;
      expect(calls.length).toBeGreaterThan(0);
      expect(calls[calls.length - 1][0].base_url).toBe('https://api.moonshot.cn/v1');
    });
  });

  it('leaves a saved custom endpoint untouched', async () => {
    mockApi({});
    const onChange = vi.fn();
    render(
      <LLMProviderSettings
        mode="project"
        projectValues={{
          provider: 'moonshot',
          model: 'kimi-k2.5',
          base_url: 'https://my-proxy.example/v1',
          sdk: 'openai',
        }}
        onChange={onChange}
      />,
    );
    await waitFor(() => {
      const calls = onChange.mock.calls;
      expect(calls.length).toBeGreaterThan(0);
      expect(calls[calls.length - 1][0].base_url).toBe('https://my-proxy.example/v1');
    });
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

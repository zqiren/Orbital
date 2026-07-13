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

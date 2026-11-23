// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import type { CatalogProvider, ProviderInfo } from '../../types';
import {
  catalogFromRegistry, consoleUrl, hostOf, initialRegion, matchesProvider, shortName, TILE_ORDER,
} from './providerMeta';

const kimi: CatalogProvider = {
  id: 'moonshot', display_name: 'Moonshot (Kimi)', auth: 'api_key', signin: null, key_entry: true,
  regions: [
    { id: 'global', label: 'International', console_url: 'https://platform.moonshot.ai/console/api-keys' },
    { id: 'china', label: 'China', console_url: 'https://platform.moonshot.cn/console/api-keys' },
  ],
  default_region: 'global', console_url: 'https://platform.moonshot.ai/console/api-keys',
  supports_model_list: true, default_model: 'kimi-k3', china_only: false, custom_endpoint: false,
};
const td: CatalogProvider = { ...kimi, id: 'tokendance', display_name: 'TokenDance (词元跳动)', regions: [], default_region: 'china', china_only: true, console_url: 'https://tokendance.space/keys' };

describe('providerMeta', () => {
  it('the tiles are the owner\'s order, the same for both languages', () => {
    expect([...TILE_ORDER]).toEqual(['tokendance', 'chatgpt', 'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax', 'openai', 'anthropic']);
  });

  it('short tile names, never translated', () => {
    expect(shortName(kimi, 'moonshot')).toBe('Kimi');
    expect(shortName(td, 'tokendance')).toBe('TokenDance');
    expect(shortName(undefined, 'groq')).toBe('groq');
  });

  it('a two-region provider opens on the UI language\'s region; one region is just that', () => {
    expect(initialRegion(kimi, 'zh')).toBe('china');
    expect(initialRegion(kimi, 'en')).toBe('global');
    expect(initialRegion(td, 'en')).toBe('china');
  });

  it('"Get a key" follows the region', () => {
    expect(consoleUrl(kimi, 'china')).toBe('https://platform.moonshot.cn/console/api-keys');
    expect(consoleUrl(kimi, 'global')).toBe('https://platform.moonshot.ai/console/api-keys');
    expect(consoleUrl(td, 'global')).toBe('https://tokendance.space/keys');
    expect(hostOf('https://platform.minimaxi.com/user-center/basic-information/interface-key')).toBe('platform.minimaxi.com');
  });

  it('search matches id, name and Chinese aliases', () => {
    expect(matchesProvider('qwen', 'Alibaba Cloud Qwen (DashScope)', '通义')).toBe(true);
    expect(matchesProvider('moonshot', 'Moonshot (Kimi)', '月之暗面')).toBe(true);
    expect(matchesProvider('hunyuan', 'Tencent Hunyuan (HY)', '腾讯')).toBe(true);
    expect(matchesProvider('deepseek', 'DeepSeek', 'deep')).toBe(true);
    expect(matchesProvider('deepseek', 'DeepSeek', '腾讯')).toBe(false);
    expect(matchesProvider('custom', '自定义 / 本地模型', 'ollama')).toBe(true);
  });

  it('an older daemon\'s raw registry still yields a catalog (no ChatGPT)', () => {
    const reg: Record<string, ProviderInfo> = {
      moonshot: { display_name: 'Moonshot (Kimi)', base_url: 'https://api.moonshot.ai/v1', china_base_url: 'https://api.moonshot.cn/v1', supports_model_list: true, sdk: 'openai', suggested_models: ['kimi-k3'], notes: '' },
      tokendance: { display_name: 'TokenDance', base_url: 'x', supports_model_list: true, sdk: 'openai', suggested_models: [], notes: '', china_only: true, default_model: 'deepseek-v4-flash' },
    };
    const cat = catalogFromRegistry(reg);
    expect(cat.find((p) => p.id === 'moonshot')!.regions.map((r) => r.id)).toEqual(['global', 'china']);
    expect(cat.find((p) => p.id === 'tokendance')!.signin).toBe('tokendance');
    expect(cat.some((p) => p.id === 'chatgpt')).toBe(false);
  });
});

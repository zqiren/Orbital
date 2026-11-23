// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * What the connect panel knows about providers beyond the catalog: the tile
 * order, the short tile labels, and search aliases. None of this is UI
 * chrome (provider names are never translated), so it is data here rather
 * than catalog strings.
 */
import type { CatalogProvider, ProviderRegistry } from '../../types';

/** The ChatGPT sign-in provider: its own tile, separate from OpenAI. */
export const CHATGPT = 'chatgpt';
/** Custom / self-hosted. */
export const CUSTOM = 'custom';

/** The tiles, in the owner's order, the same in both languages. */
export const TILE_ORDER = [
  'tokendance', CHATGPT, 'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax', 'openai', 'anthropic',
] as const;

/** How many tiles a narrow panel shows before "More" (4 columns, 2 rows). */
export const NARROW_TILE_COUNT = 7;

/** Lists this long or shorter show every row and no search. */
export const MODEL_SEARCH_THRESHOLD = 12;

/** A list this long gets a search and a fixed-height scroll area. */
export function needsSearch(count: number): boolean {
  return count > MODEL_SEARCH_THRESHOLD;
}

const SHORT_NAME: Record<string, string> = {
  tokendance: 'TokenDance',
  [CHATGPT]: 'ChatGPT',
  deepseek: 'DeepSeek',
  moonshot: 'Kimi',
  zhipu: 'GLM',
  qwen: 'Qwen',
  minimax: 'MiniMax',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Gemini',
  openrouter: 'OpenRouter',
};

/** Extra words a search matches on, Chinese names included. */
const ALIASES: Record<string, string> = {
  tokendance: '词元跳动 观猹 watcha',
  [CHATGPT]: 'openai gpt plus pro team',
  openai: 'gpt chatgpt',
  anthropic: 'claude',
  deepseek: '深度求索',
  moonshot: 'kimi 月之暗面',
  zhipu: 'glm 智谱 清言 z.ai zai',
  qwen: '阿里云 阿里 通义 千问 百炼 dashscope alibaba aliyun',
  minimax: '海螺 稀宇',
  hunyuan: '腾讯 腾讯云 混元 tencent',
  google: 'gemini 谷歌',
  xai: 'grok',
  'opencode-zen': 'opencode',
  'opencode-go': 'opencode',
  [CUSTOM]: '自定义 本地 local self-hosted ollama lm studio vllm',
};

export type Region = 'china' | 'global';

export function shortName(p: CatalogProvider | undefined, id: string): string {
  return SHORT_NAME[id] ?? p?.display_name ?? id;
}

export function hasRegions(p: CatalogProvider | undefined): boolean {
  return !!p && p.regions.length > 1;
}

/** The region a newly picked provider opens on: the UI language decides
 *  between two regions (zh → China, en → Global); one region is just that. */
export function initialRegion(p: CatalogProvider | undefined, locale: 'en' | 'zh'): Region {
  if (!p) return 'global';
  if (hasRegions(p)) return locale === 'zh' ? 'china' : 'global';
  return p.default_region;
}

/** The "Get a key" console for a provider in a region; null when none. */
export function consoleUrl(p: CatalogProvider | undefined, region: Region): string | null {
  if (!p) return null;
  if (hasRegions(p)) {
    const r = p.regions.find((x) => x.id === region);
    if (r?.console_url) return r.console_url;
  }
  return p.console_url ?? null;
}

/** The host shown under "Get a key", e.g. platform.minimaxi.com. */
export function hostOf(url: string): string {
  try {
    const u = new URL(url);
    return u.host + (u.pathname.length > 1 && u.pathname.length < 24 ? u.pathname : '');
  } catch {
    return url;
  }
}

/** Search over id, name and aliases (Chinese aliases included). */
export function matchesProvider(id: string, name: string, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return `${id} ${name} ${ALIASES[id] ?? ''}`.toLowerCase().includes(q);
}

/**
 * The catalog derived from the raw registry, for a daemon that predates
 * GET /providers/catalog. ChatGPT is only offered by the catalog route: an
 * older daemon cannot sign in to it.
 */
export function catalogFromRegistry(reg: ProviderRegistry): CatalogProvider[] {
  return Object.entries(reg).map(([id, info]) => {
    const two = !!info.china_base_url;
    const isCustom = id === CUSTOM;
    return {
      id,
      display_name: info.display_name,
      auth: isCustom ? 'none' : 'api_key',
      signin: id === 'tokendance' ? 'tokendance' : null,
      key_entry: true,
      regions: two
        ? [
            { id: 'global', label: 'International', console_url: null },
            { id: 'china', label: 'China', console_url: info.console_url ?? null },
          ]
        : [],
      default_region: info.china_only ? 'china' : 'global',
      console_url: info.console_url ?? null,
      supports_model_list: info.supports_model_list,
      default_model: info.default_model ?? info.suggested_models?.[0] ?? null,
      china_only: !!info.china_only,
      custom_endpoint: isCustom,
      sdk: info.sdk,
    } satisfies CatalogProvider;
  });
}

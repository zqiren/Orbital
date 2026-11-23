// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * The connect panel's rules (batch 1123, option C):
 *  - the key is tested on paste (trimmed), Enter (not while an IME composes),
 *    blur with an untested key, and Next / Save — never on typing alone;
 *  - Next is enabled whenever a key is present; with an untested key it tests
 *    first and saves only on a pass; a fail stays under the field and editing
 *    the key clears it;
 *  - a click that blurs the field (Next, a tile) lands and is not swallowed;
 *  - switching region marks the key untested and hides the model row;
 *  - the model row appears only after a pass; lists over 12 get a search;
 *  - one-click sign-in: waiting → account + model row; refusal offers a key.
 */
import { render, screen, waitFor, cleanup, fireEvent, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { CatalogProvider, CredentialCard } from '../../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../../config', async (orig) => ({ ...(await orig<typeof import('../../config')>()), api: apiMock }));

import ConnectPanel, { type ConnectFooterCtx } from './ConnectPanel';
import { resetCatalogCache } from './connectApi';
import { MODEL_SEARCH_THRESHOLD, needsSearch } from './providerMeta';

const cat = (o: Partial<CatalogProvider> & { id: string }): CatalogProvider => ({
  display_name: o.id, auth: 'api_key', signin: null, key_entry: true, regions: [],
  default_region: 'global', console_url: `https://${o.id}.example/keys`, supports_model_list: true,
  default_model: null, china_only: false, custom_endpoint: false, ...o,
});
const CATALOG: CatalogProvider[] = [
  cat({ id: 'openai', display_name: 'OpenAI', default_model: 'gpt-6.2-sol' }),
  cat({ id: 'chatgpt', display_name: 'ChatGPT', auth: 'signin', signin: 'chatgpt', key_entry: false, console_url: null }),
  cat({ id: 'anthropic', display_name: 'Anthropic' }),
  cat({ id: 'deepseek', display_name: 'DeepSeek', default_model: 'deepseek-v4-pro' }),
  cat({
    id: 'moonshot', display_name: 'Moonshot (Kimi)', default_model: 'kimi-k3',
    regions: [
      { id: 'global', label: 'International', console_url: 'https://platform.moonshot.ai/keys' },
      { id: 'china', label: 'China', console_url: 'https://platform.moonshot.cn/keys' },
    ],
  }),
  cat({ id: 'zhipu', display_name: 'Zhipu AI (GLM)' }),
  cat({ id: 'qwen', display_name: 'Alibaba Cloud Qwen (DashScope)' }),
  cat({ id: 'minimax', display_name: 'MiniMax' }),
  cat({ id: 'tokendance', display_name: 'TokenDance (词元跳动)', signin: 'tokendance', china_only: true }),
  cat({ id: 'openrouter', display_name: 'OpenRouter' }),
  cat({ id: 'hunyuan', display_name: 'Tencent Hunyuan (HY)' }),
  cat({ id: 'custom', display_name: 'Custom / Self-Hosted', auth: 'none', custom_endpoint: true, console_url: null }),
];
const LONG = Array.from({ length: 30 }, (_, i) => `vendor/model-${i}`);

const card = (o: Partial<CredentialCard>): CredentialCard => ({
  id: 'card_1', name: 'x', provider: 'deepseek', region: 'global', base_url: null, sdk: null, model: '',
  created_at: '', verified_at: null, last_used_at: null, last_error: null, key_set: true,
  key_masked: 'sk-a...wxyz', key_source: 'keychain', is_default: true, read_only: false, ...o,
});

interface Calls { check: Record<string, unknown>[]; save: { method: string; path: string; body: Record<string, unknown> }[] }
let calls: Calls;
/** Resolvers for a held check, so a test can act while it runs. */
let holdCheck: ((v: unknown) => void) | null;

function mockApi(opts: { pass?: (b: Record<string, unknown>) => boolean; hold?: boolean; signin?: 'ok' | 'refused' | 'hold'; models?: string[] } = {}) {
  apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    if (path === '/api/v2/providers/catalog') return { providers: CATALOG };
    if (path === '/api/v2/providers/check') {
      calls.check.push(body);
      if (opts.hold) await new Promise((r) => { holdCheck = r; });
      const ok = opts.pass ? opts.pass(body) : !String(body.api_key ?? '').includes('bad');
      if (!ok) return { ok: false, status: 401, code: 'invalid_api_key', message: 'Incorrect API key provided: sk-****bad.', models: [], default_model: null };
      const models = opts.models ?? (body.provider === 'openrouter' ? LONG : ['m-default', 'm-other']);
      return { ok: true, status: 200, code: null, message: 'ok', models, default_model: models[0] };
    }
    if (path === '/api/v2/settings/cards' || path.startsWith('/api/v2/settings/cards/')) {
      calls.save.push({ method: init?.method ?? 'GET', path, body });
      return { card: card({ id: 'card_saved', provider: body.provider ?? 'chatgpt', model: body.model }), test: { ok: true, status: 200, code: null, message: '' } };
    }
    if (path === '/api/v2/providers/chatgpt/signin') {
      if (opts.signin === 'hold') return new Promise(() => {});
      if (opts.signin === 'refused') {
        const { ApiError } = await vi.importActual<typeof import('../../config')>('../../config');
        throw new ApiError(403, JSON.stringify({ code: 'not_eligible', message: 'Business workspaces cannot.' }));
      }
      const c = card({ id: 'card_cg', provider: 'chatgpt', model: 'gpt-6.2-sol', auth: 'signin', key_set: false, key_masked: '', account: { email: 'a@example.com' } });
      return { card: c, test: { ok: true, status: 200, code: null, message: '' }, account: { email: 'a@example.com' } };
    }
    if (path === '/api/v2/providers/chatgpt/signin/cancel') return { cancelled: true };
    if (path === '/api/v2/providers/models') return { models: ['gpt-6.2-sol', 'gpt-6.1-terra'], details: [{ id: 'gpt-6.2-sol', display_name: 'GPT-6.2 Sol' }] };
    return {};
  });
}

const footer = ({ canSubmit, busy, submit, error }: ConnectFooterCtx) => (
  <div>
    <button type="button" data-testid="next" disabled={!canSubmit} aria-busy={busy} onClick={submit}>Next</button>
    {error && <p data-testid="footer-error">{error}</p>}
  </div>
);

function renderPanel(props: Partial<React.ComponentProps<typeof ConnectPanel>> = {}) {
  const onDone = vi.fn();
  const onCardStored = vi.fn();
  render(<ConnectPanel onDone={onDone} onCardStored={onCardStored} footer={footer} {...props} />);
  return { onDone, onCardStored };
}

const keyField = () => screen.getByTestId('connect-key') as HTMLInputElement;
const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

beforeEach(() => {
  apiMock.mockReset();
  resetCatalogCache();
  calls = { check: [], save: [] };
  holdCheck = null;
  localStorage.setItem('orbital.locale', 'en');
});
afterEach(() => cleanup());

describe('ConnectPanel — tiles', () => {
  it('shows the owner\'s tiles in order, then More; nothing below until a pick', async () => {
    mockApi();
    renderPanel();
    await screen.findByTestId('connect-tile-tokendance');
    const ids = screen.getAllByTestId(/^connect-tile-/).map((el) => el.dataset.testid!.replace('connect-tile-', ''));
    expect(ids).toEqual(['tokendance', 'chatgpt', 'deepseek', 'moonshot', 'zhipu', 'qwen', 'minimax', 'openai', 'anthropic']);
    expect(screen.getByTestId('connect-more')).toBeInTheDocument();
    expect(screen.queryByTestId('connect-key')).toBeNull();
    expect(screen.getByTestId('next')).toBeDisabled();
  });

  it('More searches every provider by id, name and Chinese alias', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-more'));
    await userEvent.type(screen.getByTestId('connect-more-search'), '腾讯');
    expect(screen.getByTestId('connect-pick-hunyuan')).toBeInTheDocument();
    expect(screen.queryByTestId('connect-pick-openai')).toBeNull();
    await userEvent.click(screen.getByTestId('connect-pick-hunyuan'));
    expect(screen.getByTestId('connect-provider-hunyuan')).toBeInTheDocument();
  });
});

describe('ConnectPanel — when the key is tested', () => {
  it('typing alone never tests', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    await userEvent.type(keyField(), 'sk-typed-key');
    await act(async () => { await new Promise((r) => setTimeout(r, 600)); });
    expect(calls.check).toHaveLength(0);
    expect(screen.queryByTestId('connect-model-row')).toBeNull();
    // ...but Next is already live: a key is present.
    expect(screen.getByTestId('next')).toBeEnabled();
  });

  it('paste tests at once, with surrounding whitespace trimmed', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    keyField().focus();
    await userEvent.paste('   sk-pasted  \n');
    await waitFor(() => expect(calls.check).toHaveLength(1));
    expect(calls.check[0]).toMatchObject({ provider: 'deepseek', api_key: 'sk-pasted' });
    expect(keyField().value).toBe('sk-pasted');
    await screen.findByTestId('connect-key-ok');
    expect(screen.getByTestId('connect-model-value')).toHaveTextContent('m-default');
  });

  it('Enter tests; an Enter that confirms an IME candidate does not', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    await userEvent.type(keyField(), 'sk-abc');
    fireEvent.keyDown(keyField(), { key: 'Enter', isComposing: true });
    fireEvent.keyDown(keyField(), { key: 'Enter', keyCode: 229 });
    await flush();
    expect(calls.check).toHaveLength(0);
    fireEvent.keyDown(keyField(), { key: 'Enter' });
    await waitFor(() => expect(calls.check).toHaveLength(1));
  });

  it('leaving the field with an untested key tests it; with no key, nothing', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    keyField().focus();
    keyField().blur();
    await flush();
    expect(calls.check).toHaveLength(0);
    await userEvent.type(keyField(), 'sk-abc');
    await userEvent.tab();
    await waitFor(() => expect(calls.check).toHaveLength(1));
    // A tested key is not re-tested by another blur.
    await screen.findByTestId('connect-key-ok');
    keyField().focus();
    keyField().blur();
    await flush();
    expect(calls.check).toHaveLength(1);
  });

  it('Next with an untested key tests first, then saves on the pass with the preselected model', async () => {
    mockApi();
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    await userEvent.type(keyField(), 'sk-good');
    // The click blurs the field: it must land, and test exactly once.
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(calls.check).toHaveLength(1);
    expect(calls.save).toHaveLength(1);
    expect(calls.save[0]).toMatchObject({ method: 'POST', body: { provider: 'deepseek', model: 'm-default', api_key: 'sk-good' } });
  });

  it('Next with a failing key saves nothing: the reason stays under the field; editing clears it', async () => {
    mockApi();
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-openai'));
    await userEvent.type(keyField(), 'sk-bad-key');
    await userEvent.click(screen.getByTestId('next'));
    const reason = await screen.findByTestId('connect-key-error');
    expect(reason).toHaveTextContent('Invalid key · 401');
    // The provider's sentence sits behind the sign, not on screen.
    expect(screen.queryByText(/Incorrect API key provided/)).toBeNull();
    await userEvent.hover(screen.getByTestId('connect-key-error-sign'));
    expect(screen.getByText(/Incorrect API key provided/)).toBeInTheDocument();
    expect(calls.save).toHaveLength(0);
    expect(onDone).not.toHaveBeenCalled();
    // Never a dead end: Next stays live, and editing the key clears the reason.
    expect(screen.getByTestId('next')).toBeEnabled();
    await userEvent.type(keyField(), 'x');
    expect(screen.queryByTestId('connect-key-error')).toBeNull();
  });

  it('Next pressed while a test runs waits for it, then saves once', async () => {
    mockApi({ hold: true });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    keyField().focus();
    await userEvent.paste('sk-good');
    await waitFor(() => expect(holdCheck).not.toBeNull());
    await userEvent.click(screen.getByTestId('next'));
    expect(screen.getByTestId('next')).toHaveAttribute('aria-busy', 'true');
    expect(calls.check).toHaveLength(1);
    await act(async () => { holdCheck!(null); });
    await waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(calls.check).toHaveLength(1);
    expect(calls.save).toHaveLength(1);
  });

  it('a tile clicked while the field holds an untested key switches provider, and tests nothing', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    await userEvent.type(keyField(), 'sk-for-deepseek');
    await userEvent.click(screen.getByTestId('connect-tile-anthropic'));
    await flush();
    expect(screen.getByTestId('connect-provider-anthropic')).toBeInTheDocument();
    expect(keyField().value).toBe('');
    expect(calls.check).toHaveLength(0);
  });
});

describe('ConnectPanel — region', () => {
  it('only a two-region provider shows the toggle; switching marks the key untested', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    expect(screen.queryByTestId('connect-region')).toBeNull();
    await userEvent.click(screen.getByTestId('connect-tile-moonshot'));
    expect(screen.getByTestId('connect-region')).toBeInTheDocument();
    // English UI opens on Global, and the ↗ follows the region.
    expect(screen.getByTestId('connect-region-global')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('connect-get-key')).toHaveAttribute('href', 'https://platform.moonshot.ai/keys');
    keyField().focus();
    await userEvent.paste('sk-kimi');
    await screen.findByTestId('connect-key-ok');
    expect(screen.getByTestId('connect-model-row')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('connect-region-china'));
    expect(screen.queryByTestId('connect-key-ok')).toBeNull();
    expect(screen.queryByTestId('connect-model-row')).toBeNull();
    expect(screen.getByTestId('connect-get-key')).toHaveAttribute('href', 'https://platform.moonshot.cn/keys');
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(calls.check).toHaveLength(2));
    expect(calls.check[1]).toMatchObject({ provider: 'moonshot', region: 'china', api_key: 'sk-kimi' });
    await waitFor(() => expect(calls.save[0]?.body).toMatchObject({ region: 'china' }));
  });

  it('a region click that blurs an untested key lands, then tests with the new region', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-moonshot'));
    await userEvent.type(keyField(), 'sk-kimi');
    await userEvent.click(screen.getByTestId('connect-region-china'));
    await waitFor(() => expect(calls.check).toHaveLength(1));
    expect(calls.check[0]).toMatchObject({ region: 'china' });
  });
});

describe('ConnectPanel — model row', () => {
  it('a short list shows every model and no search; picking one closes it', async () => {
    mockApi({ models: ['a', 'b', 'c'] });
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-deepseek'));
    keyField().focus();
    await userEvent.paste('sk-good');
    await userEvent.click(await screen.findByTestId('connect-model-toggle'));
    expect(screen.queryByTestId('connect-model-search')).toBeNull();
    expect(screen.getAllByRole('option')).toHaveLength(3);
    await userEvent.click(screen.getByTestId('connect-model-option-c'));
    expect(screen.queryByTestId('connect-model-list')).toBeNull();
    expect(screen.getByTestId('connect-model-value')).toHaveTextContent('c');
  });

  it('a long list gets a search with a count', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-more'));
    await userEvent.click(screen.getByTestId('connect-pick-openrouter'));
    keyField().focus();
    await userEvent.paste('sk-or');
    await userEvent.click(await screen.findByTestId('connect-model-toggle'));
    await userEvent.type(screen.getByTestId('connect-model-search'), 'model-2');
    // model-2, model-20 … model-29
    expect(screen.getAllByRole('option')).toHaveLength(11);
  });

  it('the search threshold is 12', () => {
    expect(MODEL_SEARCH_THRESHOLD).toBe(12);
    expect(needsSearch(12)).toBe(false);
    expect(needsSearch(13)).toBe(true);
  });
});

describe('ConnectPanel — one-click sign-in', () => {
  it('ChatGPT: one button → waiting with reopen and cancel → cancel returns to the button', async () => {
    mockApi({ signin: 'hold' });
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    expect(screen.queryByTestId('connect-key')).toBeNull();
    await userEvent.click(screen.getByTestId('connect-signin'));
    expect(screen.getByTestId('connect-signin-waiting')).toBeInTheDocument();
    expect(screen.getByTestId('connect-signin-reopen')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('connect-signin-cancel'));
    expect(screen.getByTestId('connect-signin')).toBeInTheDocument();
    expect(apiMock).toHaveBeenCalledWith('/api/v2/providers/chatgpt/signin/cancel', { method: 'POST' });
  });

  it('ChatGPT: a completed sign-in shows the account and model row; Next writes only a changed model', async () => {
    mockApi({ signin: 'ok' });
    const { onDone, onCardStored } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    expect(await screen.findByTestId('connect-account-email')).toHaveTextContent('a@example.com');
    expect(onCardStored).toHaveBeenCalled();
    // Labels from the account catalog.
    await waitFor(() => expect(screen.getByTestId('connect-model-value')).toHaveTextContent('GPT-6.2 Sol'));
    await userEvent.click(screen.getByTestId('connect-model-toggle'));
    await userEvent.click(screen.getByTestId('connect-model-option-gpt-6.1-terra'));
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(calls.save).toEqual([{ method: 'PUT', path: '/api/v2/settings/cards/card_cg', body: { model: 'gpt-6.1-terra' } }]);
  });

  it('ChatGPT: Next straight after sign-in saves nothing and still advances', async () => {
    mockApi({ signin: 'ok' });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    await screen.findByTestId('connect-account');
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalledTimes(1));
    expect(calls.save).toHaveLength(0);
  });

  it('ChatGPT refusal: use an OpenAI key instead, or try another account', async () => {
    mockApi({ signin: 'refused' });
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    await screen.findByTestId('connect-signin-refused');
    expect(screen.getByTestId('connect-signin-other')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('connect-use-openai-key'));
    expect(screen.getByTestId('connect-provider-openai')).toBeInTheDocument();
    expect(screen.getByTestId('connect-tile-openai')).toHaveAttribute('aria-pressed', 'true');
  });

  it('TokenDance opens on sign-in, with "I have a key" for a pasted key', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-tokendance'));
    expect(screen.getByTestId('connect-signin')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('connect-have-key'));
    expect(screen.getByTestId('connect-key')).toBeInTheDocument();
  });
});

describe('ConnectPanel — editing a saved card', () => {
  it('opens connected with no tiles, the model row open; Save with no change writes nothing', async () => {
    mockApi();
    const { onDone } = renderPanel({ card: card({ id: 'card_km', provider: 'moonshot', region: 'china', model: 'kimi-k3' }) });
    await screen.findByTestId('connect-key-ok');
    expect(screen.queryByTestId('connect-tile-deepseek')).toBeNull();
    expect(screen.getByTestId('connect-region-china')).toHaveAttribute('aria-pressed', 'true');
    await screen.findByTestId('connect-model-list');
    await userEvent.click(screen.getByTestId('next'));
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(calls.save).toHaveLength(0);
    expect(calls.check).toHaveLength(0);
  });

  it('changing the model PUTs only the model', async () => {
    mockApi();
    renderPanel({ card: card({ id: 'card_km', provider: 'deepseek', model: 'gpt-6.2-sol' }) });
    await screen.findByTestId('connect-model-list');
    await userEvent.click(screen.getByTestId('connect-model-option-gpt-6.1-terra'));
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(calls.save).toHaveLength(1));
    expect(calls.save[0]).toEqual({ method: 'PUT', path: '/api/v2/settings/cards/card_km', body: { model: 'gpt-6.1-terra' } });
  });
});

describe('ConnectPanel — carried over from the old form', () => {
  it('zh UI: a two-region provider opens on China (en opens on Global, above)', async () => {
    localStorage.setItem('orbital.locale', 'zh');
    mockApi();
    const { LocaleProvider } = await import('../../i18n/LocaleContext');
    render(
      <LocaleProvider>
        <ConnectPanel onDone={vi.fn()} footer={footer} />
      </LocaleProvider>,
    );
    await userEvent.click(await screen.findByTestId('connect-tile-moonshot'));
    expect(screen.getByTestId('connect-region-china')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('connect-region-china').textContent).toBe('中国');
    expect(screen.getByTestId('connect-region-global').textContent).toBe('国际');
  });

  it('Custom appears exactly once, last in More, and asks for an endpoint and a protocol', async () => {
    mockApi({ models: ['local-model'] });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-more'));
    expect(screen.getAllByTestId('connect-pick-custom')).toHaveLength(1);
    const rows = screen.getAllByTestId(/^connect-pick-/);
    expect(rows[rows.length - 1]).toBe(screen.getByTestId('connect-pick-custom'));
    await userEvent.click(screen.getByTestId('connect-pick-custom'));
    // A local server needs no key: Next is live once the endpoint is in.
    expect(screen.getByTestId('next')).toBeDisabled();
    await userEvent.type(screen.getByTestId('connect-custom-url'), 'http://localhost:1234/v1');
    await userEvent.selectOptions(screen.getByTestId('connect-custom-sdk'), 'openai-responses');
    expect(screen.getByTestId('next')).toBeEnabled();
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(calls.check[0]).toMatchObject({ provider: 'custom', base_url: 'http://localhost:1234/v1', sdk: 'openai-responses' });
    expect(calls.save[0].body).toMatchObject({ provider: 'custom', base_url: 'http://localhost:1234/v1', sdk: 'openai-responses', model: 'local-model' });
  });

  it('TokenDance one-click: the daemon stores the card; Next then needs no save (9.png, 2026-09-08)', async () => {
    mockApi();
    apiMock.mockImplementationOnce(async () => ({ providers: CATALOG }));
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
      if (path === '/api/v2/providers/tokendance/signin') {
        return { card: card({ id: 'card_td', provider: 'tokendance', model: 'deepseek-v4-flash', key_masked: 'sk-t...9876' }), test: { ok: true, status: 200, code: null, message: '' } };
      }
      if (path === '/api/v2/providers/models') return { models: ['deepseek-v4-flash', 'glm-5.2'] };
      return base(path, init);
    });
    const { onDone, onCardStored } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-tokendance'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    await screen.findByTestId('connect-account');
    expect(onCardStored).toHaveBeenCalledWith(expect.objectContaining({ id: 'card_td' }));
    await waitFor(() => expect(screen.getByTestId('connect-model-value')).toHaveTextContent('deepseek-v4-flash'));
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ id: 'card_td' })));
    expect(calls.save).toHaveLength(0);
  });

  it('a sign-in that stored a red card shows the reason, and Next is not a dead end', async () => {
    mockApi();
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
      if (path === '/api/v2/providers/chatgpt/signin') {
        return { card: card({ id: 'card_cg', provider: 'chatgpt', model: 'gpt-6.2-sol', auth: 'signin', key_masked: '' }), test: { ok: false, status: 429, code: 'usage_limit_reached', message: 'Your ChatGPT plan’s usage limit is reached.' }, account: { email: 'a@example.com' } };
      }
      return base(path, init);
    });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    expect(await screen.findByTestId('connect-signin-test')).toHaveTextContent('Usage limit reached · 429');
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
  });

  it('sign out deletes the signed-in card and returns to the sign-in button', async () => {
    mockApi({ signin: 'ok' });
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    await userEvent.click(await screen.findByTestId('connect-signout'));
    await screen.findByTestId('connect-signin');
    expect(calls.save).toContainEqual({ method: 'DELETE', path: '/api/v2/settings/cards/card_cg', body: {} });
  });

  it('editing a card whose key stopped working shows why, and Save re-tests the stored key first', async () => {
    mockApi();
    const { onDone } = renderPanel({
      card: card({ id: 'card_bad', provider: 'deepseek', model: 'm-default', last_error: { status: 402, code: 'insufficient_credits', message: 'Insufficient balance', at: '' } }),
    });
    expect(await screen.findByTestId('connect-key-error')).toHaveTextContent('Out of credits · 402');
    expect(keyField().placeholder).toBe('sk-a...wxyz');
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(calls.check[0]).toMatchObject({ card_id: 'card_bad', provider: 'deepseek' });
    expect(calls.check[0]).not.toHaveProperty('api_key');
  });

  it('a long list can take a model the list does not have', async () => {
    mockApi();
    renderPanel();
    await userEvent.click(await screen.findByTestId('connect-more'));
    await userEvent.click(screen.getByTestId('connect-pick-openrouter'));
    keyField().focus();
    await userEvent.paste('sk-or');
    await userEvent.click(await screen.findByTestId('connect-model-toggle'));
    await userEvent.type(screen.getByTestId('connect-model-search'), 'vendor/brand-new');
    await userEvent.click(screen.getByTestId('connect-model-use-typed'));
    expect(screen.getByTestId('connect-model-value')).toHaveTextContent('vendor/brand-new');
  });
});

describe('ConnectPanel — a sign-in OpenAI refuses after storing', () => {
  it('a not_eligible test on the stored card is the refusal; using a key removes that card first', async () => {
    mockApi();
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
      if (path === '/api/v2/providers/chatgpt/signin') {
        return {
          card: card({ id: 'card_red', provider: 'chatgpt', auth: 'signin', key_masked: '' }),
          test: { ok: false, status: 403, code: 'not_eligible', message: 'This ChatGPT account or plan can’t be used by other apps.' },
          account: { email: 'a@example.com' },
        };
      }
      return base(path, init);
    });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-tile-chatgpt'));
    await userEvent.click(screen.getByTestId('connect-signin'));
    await screen.findByTestId('connect-signin-refused');
    expect(screen.getByTestId('next')).toBeDisabled();
    await userEvent.click(screen.getByTestId('connect-use-openai-key'));
    await screen.findByTestId('connect-provider-openai');
    expect(calls.save).toContainEqual({ method: 'DELETE', path: '/api/v2/settings/cards/card_red', body: {} });
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe('ConnectPanel — a provider with no model list', () => {
  it('when the daemon asks for a model, a model field appears in the frame and the retest sends it', async () => {
    mockApi({ pass: (b) => !!b.model });
    const base = apiMock.getMockImplementation()!;
    apiMock.mockImplementation(async (path: string, init?: { method?: string; body?: string }) => {
      if (path === '/api/v2/providers/check') {
        const body = init?.body ? JSON.parse(init.body) : {};
        if (!body.model) {
          calls.check.push(body);
          return { ok: false, status: null, code: 'missing_model', message: 'Type a model name.', models: [], default_model: null };
        }
      }
      return base(path, init);
    });
    const { onDone } = renderPanel();
    await userEvent.click(await screen.findByTestId('connect-more'));
    await userEvent.click(screen.getByTestId('connect-pick-custom'));
    await userEvent.type(screen.getByTestId('connect-custom-url'), 'http://localhost:9/v1');
    await userEvent.click(screen.getByTestId('next'));
    expect(await screen.findByTestId('connect-key-error')).toHaveTextContent('No model');
    await userEvent.type(screen.getByTestId('connect-model-typed'), 'llama3');
    expect(screen.queryByTestId('connect-key-error')).toBeNull();
    await userEvent.click(screen.getByTestId('next'));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(calls.check[1]).toMatchObject({ provider: 'custom', model: 'llama3' });
    expect(calls.save[0].body).toMatchObject({ model: 'llama3' });
  });
});

describe('ConnectPanel — embedded (the Welcome page)', () => {
  it('opens on the preselected provider without taking focus, and reports a pass as connected', async () => {
    mockApi();
    const seen: ConnectFooterCtx[] = [];
    const onEngage = vi.fn();
    render(
      <ConnectPanel
        embedded
        initialProvider="deepseek"
        onEngage={onEngage}
        onDone={vi.fn()}
        footer={(ctx) => { seen.push(ctx); return null; }}
      />,
    );
    // The footer renders while the catalog loads too (the page's Start lives there).
    expect(seen.length).toBeGreaterThan(0);
    expect(seen[0].connected).toBe(false);
    await screen.findByTestId('connect-provider-deepseek');
    expect(document.activeElement).not.toBe(keyField());
    expect(onEngage).not.toHaveBeenCalled();
    keyField().focus();
    expect(onEngage).toHaveBeenCalledTimes(1);
    await userEvent.paste('sk-good');
    await screen.findByTestId('connect-key-ok');
    const last = seen[seen.length - 1];
    expect(last).toMatchObject({ connected: true, provider: 'deepseek', model: 'm-default' });
  });

  it('picking a tile engages; an unknown preset picks nothing', async () => {
    mockApi();
    const onEngage = vi.fn();
    render(<ConnectPanel embedded initialProvider="nope" onEngage={onEngage} onDone={vi.fn()} footer={footer} />);
    await screen.findByTestId('connect-tile-deepseek');
    expect(screen.queryByTestId('connect-key')).toBeNull();
    await userEvent.click(screen.getByTestId('connect-tile-openai'));
    expect(onEngage).toHaveBeenCalled();
  });
});

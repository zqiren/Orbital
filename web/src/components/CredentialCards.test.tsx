// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * CredentialCards unit tests — spec 082 §5 (frontend).
 *
 * Covers: the flat list renders every card with health, the Default badge and
 * the key suffix; Set default / Test / Delete hit the right routes; the
 * default card cannot be deleted (the daemon refuses and the row says why);
 * deleting a referenced card names the projects it moved; the env card is
 * read-only; inline rename PUTs only `name`; and the Add-card modal stays
 * open on the save-test verdict.
 *
 * The api client (web/src/config.ts `api`) is mocked — no network.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { CredentialCard, ProviderRegistry } from '../types';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', async (orig) => ({ ...(await orig<typeof import('../config')>()), api: apiMock }));

import CredentialCards from './CredentialCards';
import { resetCatalogCache } from './connect/connectApi';
import CardList from './CardList';

afterEach(() => cleanup());

function makeCard(overrides: Partial<CredentialCard> = {}): CredentialCard {
  return {
    id: 'card_aaa',
    name: 'DeepSeek · deepseek-chat',
    provider: 'deepseek',
    region: 'global',
    base_url: null,
    sdk: null,
    model: 'deepseek-chat',
    created_at: '2026-09-01T10:00:00+00:00',
    verified_at: null,
    last_used_at: null,
    last_error: null,
    key_set: true,
    key_masked: 'sk-d...1234',
    key_source: 'keychain',
    is_default: false,
    read_only: false,
    ...overrides,
  };
}

const DEFAULT_CARD = makeCard({
  id: 'card_default',
  name: 'OpenCode Go · deepseek-v4-flash',
  provider: 'opencode-go',
  model: 'deepseek-v4-flash',
  key_masked: 'sk-o...wxyz',
  is_default: true,
  verified_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
});

const BAD_CARD = makeCard({
  id: 'card_bad',
  name: 'OpenRouter (launch-vid)',
  provider: 'openrouter',
  model: '',
  last_error: {
    status: 402,
    code: 'insufficient_credits',
    message: 'Insufficient credits',
    at: '2026-09-03T22:53:00+00:00',
  },
});

const PROVIDERS: ProviderRegistry = {
  'opencode-go': {
    display_name: 'OpenCode Go',
    base_url: 'https://opencode.ai/zen/go/v1',
    supports_model_list: true,
    sdk: 'openai',
    suggested_models: [],
    notes: '',
  },
  openrouter: {
    display_name: 'OpenRouter',
    base_url: 'https://openrouter.ai/api/v1',
    supports_model_list: true,
    sdk: 'openai',
    suggested_models: [],
    notes: '',
  },
};

const cat = (id: string, display_name: string, extra: Record<string, unknown> = {}) => ({
  id, display_name, auth: 'api_key', signin: null, key_entry: true, regions: [], default_region: 'global',
  console_url: null, supports_model_list: true, default_model: null, china_only: false, custom_endpoint: false, ...extra,
});
const CATALOG = [
  cat('deepseek', 'DeepSeek', { default_model: 'deepseek-chat' }),
  cat('opencode-go', 'OpenCode Go'),
  cat('openrouter', 'OpenRouter'),
  cat('moonshot', 'Moonshot (Kimi)', {
    regions: [
      { id: 'global', label: 'International', console_url: null },
      { id: 'china', label: 'China', console_url: null },
    ],
  }),
  cat('chatgpt', 'ChatGPT', { auth: 'signin', signin: 'chatgpt', key_entry: false }),
];

/** A passing check with this list, or a failing one with this sentence. */
const checkPass = (models = ['deepseek-chat']) => ({ ok: true, status: 200, code: null, message: '', models, default_model: models[0] });
const checkFail = (message: string, status = 401) => ({ ok: false, status, code: 'invalid_api_key', message, models: [], default_model: null });

/** In the open Add dialog: pick a tile, type a key, press Save. */
async function addWithKey(provider: string, key: string) {
  fireEvent.click(await screen.findByTestId(`connect-tile-${provider}`));
  const field = screen.getByTestId('connect-key');
  fireEvent.change(field, { target: { value: key } });
  fireEvent.click(screen.getByTestId('card-save'));
}

/** GET /settings serves the card list; every other route is per-test. */
function mockCards(cards: CredentialCard[], extra?: (path: string, opts?: RequestInit) => unknown) {
  apiMock.mockImplementation(async (path: string, opts?: RequestInit) => {
    const handled = extra?.(path, opts);
    if (handled !== undefined) return handled;
    if (path === '/api/v2/settings') {
      return {
        credential_cards: cards,
        default_card_id: cards.find((c) => c.is_default)?.id ?? null,
        llm: { api_key_set: true, api_key_masked: '', base_url: null, model: null, sdk: 'openai', provider: '' },
      };
    }
    if (path === '/api/v2/providers') return PROVIDERS;
    if (path === '/api/v2/providers/catalog') return { providers: CATALOG };
    return {};
  });
}

function renderList() {
  return render(<CredentialCards providers={PROVIDERS} />);
}

beforeEach(() => {
  apiMock.mockReset();
  resetCatalogCache();
  vi.spyOn(window, 'confirm').mockReturnValue(true);
});

describe('CredentialCards — the flat list', () => {
  it('renders one row per card, with model, key suffix and health', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD]);
    renderList();

    const row = await screen.findByTestId('card-row-card_default');
    // The name already IS "<Provider> · <model>", so the line under it carries
    // only what the name does not: the key suffix. Repeating the setup was the
    // bulk this redesign removed.
    expect(row.textContent).toContain('OpenCode Go · deepseek-v4-flash');
    expect(row.textContent).toContain('sk-o...wxyz');
    // Health is a SYMBOL, not a sentence: the words are the mark's accessible
    // name and its hover tooltip, so a list of four cards no longer reads like
    // a changelog. Assert the accessible name — that is what a screen reader
    // gets, and what the tooltip shows.
    expect(screen.getByTestId('card-health-card_default').getAttribute('aria-label')).toBe(
      'Verified 2h ago',
    );

    // A card that failed carries the failure, with the clock time — the
    // 2026-09-03 402 is exactly the thing this list exists to show.
    const bad = screen.getByTestId('card-health-card_bad');
    expect(bad.getAttribute('aria-label')).toContain('402 Insufficient credits');
    expect(screen.getByTestId('card-health-tip-card_bad').textContent).toContain(
      '402 Insufficient credits',
    );
    // A migrated card with no model is flagged rather than hidden (D7).
    expect(screen.getByTestId('card-row-card_bad').textContent).toContain('needs a model');
  });

  it('highlights exactly the default card, and every card has the same three actions', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD]);
    renderList();
    // Selection IS the highlight — there is no separate "Set default" control
    // to promote a card, because clicking the card is what promotes it.
    await screen.findByTestId('card-selected-card_default');
    expect(screen.queryByTestId('card-selected-card_bad')).toBeNull();
    expect(screen.queryByTestId('card-set-default-card_bad')).toBeNull();
    expect(
      (screen.getByTestId('card-select-card_default') as HTMLElement).getAttribute('aria-checked'),
    ).toBe('true');
    expect(
      (screen.getByTestId('card-select-card_bad') as HTMLElement).getAttribute('aria-checked'),
    ).toBe('false');
    // Exactly three per card. A fourth is the regression this guards.
    for (const id of ['card_default', 'card_bad']) {
      expect(screen.getByTestId(`card-test-${id}`)).toBeTruthy();
      expect(screen.getByTestId(`card-edit-${id}`)).toBeTruthy();
      expect(screen.getByTestId(`card-delete-${id}`)).toBeTruthy();
    }
  });

  it('says so when there are no cards yet', async () => {
    mockCards([]);
    renderList();
    expect((await screen.findByTestId('cards-list-empty')).textContent).toContain(
      'No providers yet',
    );
  });
});

describe('CredentialCards — row actions', () => {
  it('selecting a card PUTs the default route', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) =>
      path === '/api/v2/settings/cards/card_bad/default' && opts?.method === 'PUT'
        ? { default_card_id: 'card_bad', applied: true }
        : undefined,
    );
    renderList();
    fireEvent.click(await screen.findByTestId('card-select-card_bad'));
    await waitFor(() =>
      expect(apiMock).toHaveBeenCalledWith(
        '/api/v2/settings/cards/card_bad/default',
        expect.objectContaining({ method: 'PUT' }),
      ),
    );
  });

  it('says the environment card still wins when the default was only recorded', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) =>
      path === '/api/v2/settings/cards/card_bad/default' && opts?.method === 'PUT'
        ? { default_card_id: 'env', applied: false }
        : undefined,
    );
    renderList();
    fireEvent.click(await screen.findByTestId('card-select-card_bad'));
    // A click that records a choice but changes nothing that runs has to say
    // so, or it reads as a broken button.
    expect((await screen.findByTestId('card-note-card_bad')).textContent).toContain(
      'AGENT_OS_API_KEY',
    );
  });

  it('Test posts the card test route and refreshes the row health', async () => {
    const healed = { ...BAD_CARD, last_error: null, verified_at: new Date().toISOString() };
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) =>
      path === '/api/v2/settings/cards/card_bad/test' && opts?.method === 'POST'
        ? { card: healed, test: { ok: true, status: null, code: null, message: 'Connected' } }
        : undefined,
    );
    renderList();
    fireEvent.click(await screen.findByTestId('card-test-card_bad'));
    await waitFor(() =>
      expect(screen.getByTestId('card-health-card_bad').getAttribute('aria-label')).toBe(
        'Verified just now',
      ),
    );
  });

  it('Delete calls DELETE and names the projects it moved to the default', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) =>
      path === '/api/v2/settings/cards/card_bad' && opts?.method === 'DELETE'
        ? {
            id: 'card_bad',
            deleted: true,
            reassigned_projects: [{ project_id: 'proj_1', name: 'launch-vid' }],
          }
        : undefined,
    );
    renderList();
    fireEvent.click(await screen.findByTestId('card-delete-card_bad'));
    // D5: moving somebody's projects onto another account is the one delete
    // consequence worth naming out loud.
    expect((await screen.findByTestId('card-note-card_bad')).textContent).toContain('launch-vid');
  });

  it('surfaces the daemon refusing to delete the default card', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards/card_default' && opts?.method === 'DELETE') {
        throw new Error('Set another card as default first.');
      }
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('card-delete-card_default'));
    expect((await screen.findByTestId('card-error-card_default')).textContent).toContain(
      'Set another card as default first.',
    );
  });

  it('a coded refusal reads as a sentence in the UI language, never as raw JSON (F5, G43)', async () => {
    // What api() throws for a 400 with detail {code, message}: the detail
    // JSON-stringified as the Error message.
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards/card_default' && opts?.method === 'DELETE') {
        throw new Error(JSON.stringify({
          code: 'card_is_default',
          message: 'This is the default card. Make another card the default first.',
        }));
      }
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('card-delete-card_default'));
    const line = (await screen.findByTestId('card-error-card_default')).textContent ?? '';
    expect(line).toBe('Make another provider the default before deleting this one.');
    expect(line).not.toContain('{');
  });

  it('deleting the LAST provider succeeds: says what follows, no error, list refreshed to empty (I5)', async () => {
    let cards = [DEFAULT_CARD];
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    apiMock.mockImplementation(async (path: string, opts?: RequestInit) => {
      if (path === '/api/v2/settings/cards/card_default' && opts?.method === 'DELETE') {
        // Daemon contract (batch 1111): the only card may go; the install is
        // then keyless (setup.api_key_skipped: true).
        cards = [];
        return { id: 'card_default', deleted: true, reassigned_projects: [] };
      }
      if (path === '/api/v2/settings') {
        return {
          credential_cards: cards,
          default_card_id: cards.length ? 'card_default' : null,
          setup: { api_key_skipped: cards.length === 0 },
          llm: { api_key_set: cards.length > 0, api_key_masked: '', base_url: null, model: null, sdk: 'openai', provider: '' },
        };
      }
      if (path === '/api/v2/providers') return PROVIDERS;
      return {};
    });
    renderList();
    fireEvent.click(await screen.findByTestId('card-delete-card_default'));
    expect(confirmSpy.mock.calls.at(-1)![0]).toBe(
      'Delete "OpenCode Go · deepseek-v4-flash"? It\'s your only provider: Orbital\'s agent stops working, your other agents still do.',
    );
    await waitFor(() => expect(screen.queryByTestId('card-row-card_default')).toBeNull());
    expect(screen.queryByTestId('card-error-card_default')).toBeNull();
  });

  it('with other cards present, the ordinary confirm is used', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    mockCards([DEFAULT_CARD, BAD_CARD]);
    renderList();
    fireEvent.click(await screen.findByTestId('card-delete-card_bad'));
    expect(confirmSpy.mock.calls.at(-1)![0]).toContain('fall back to the default provider');
  });

  it('an unmapped code shows the daemon\'s own sentence', async () => {
    mockCards([DEFAULT_CARD, BAD_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards/card_bad/test' && opts?.method === 'POST') {
        throw new Error(JSON.stringify({ code: 'keychain_error', message: 'The Keychain is locked.' }));
      }
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('card-test-card_bad'));
    expect((await screen.findByTestId('card-error-card_bad')).textContent).toBe('The Keychain is locked.');
  });

  it('the list carries no rename affordance — renaming belongs to Edit', async () => {
    // Clicking a card is what selects it, so the name cannot also be a rename
    // trigger. The name field lives in the Edit form, one of the three actions.
    mockCards([DEFAULT_CARD]);
    renderList();
    await screen.findByTestId('card-row-card_default');
    expect(screen.queryByTestId('card-name-card_default')).toBeNull();
    expect(screen.queryByTestId('card-name-input-card_default')).toBeNull();
  });

  it('the env card cannot be edited or deleted', async () => {
    const envCard = makeCard({
      id: 'env',
      name: 'Environment key',
      read_only: true,
      key_source: 'environment',
      is_default: true,
    });
    mockCards([envCard]);
    renderList();
    await screen.findByTestId('card-row-env');
    // Inert rather than absent: the row still explains itself, but neither of
    // the two destructive actions is live.
    expect((screen.getByTestId('card-edit-env') as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId('card-delete-env') as HTMLButtonElement).disabled).toBe(true);
    // Testing the environment card is still allowed — it is read-only, not unusable.
    expect((screen.getByTestId('card-test-env') as HTMLButtonElement).disabled).toBe(false);
  });
});

describe('CredentialCards — Add card', () => {
  it('a failing test in the Add dialog stores nothing; the reason stays in the dialog (2026-11-12)', async () => {
    mockCards([DEFAULT_CARD], (path) => {
      if (path === '/api/v2/providers/check') return checkFail('Invalid API key');
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('cards-add'));
    await screen.findByTestId('card-modal');
    // A new card is saved only after its own key PASSED — Save tests first.
    await addWithKey('deepseek', 'sk-new-key');
    await waitFor(() => expect(screen.getByTestId('connect-key-error')).toBeTruthy());
    const posts = apiMock.mock.calls.filter(([p, o]) => p === '/api/v2/settings/cards' && (o as RequestInit)?.method === 'POST');
    expect(posts).toHaveLength(0);
    expect(screen.getByTestId('card-modal')).toBeTruthy();
    // Never a dead end: Save stays live for the next try.
    expect((screen.getByTestId('card-save') as HTMLButtonElement).disabled).toBe(false);
  });

  it('a Save the daemon refuses (its own test failed) shows that verdict and keeps the dialog open', async () => {
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') {
        throw new Error(JSON.stringify({
          code: 'connection_test_failed',
          message: 'Insufficient balance',
          test: { ok: false, status: 402, code: 'insufficient_credits', message: 'Insufficient balance' },
        }));
      }
      if (path === '/api/v2/providers/check') return checkPass();
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('cards-add'));
    await screen.findByTestId('card-modal');
    await addWithKey('deepseek', 'sk-new-key');

    // The refusal is the panel's verdict now, under the key; the sentence is
    // behind its sign and never raw JSON.
    await waitFor(() => expect(screen.getByTestId('connect-key-error')).toBeTruthy());
    expect(screen.getByTestId('card-modal').textContent).not.toContain('{');
    fireEvent.click(screen.getByTestId('connect-key-error-sign'));
    expect(screen.getByTestId('card-modal').textContent).toContain('Insufficient balance');
    expect(screen.getByTestId('card-modal')).toBeTruthy();
  });

  it('a successful save closes the modal by itself — Save means saved and done', async () => {
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') {
        return {
          card: makeCard({ id: 'card_new', name: 'New card' }),
          test: { ok: true, status: null, code: null, message: 'Connected to DeepSeek using deepseek-chat' },
        };
      }
      if (path === '/api/v2/providers/check') return checkPass();
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('cards-add'));
    await screen.findByTestId('card-modal');
    await addWithKey('deepseek', 'sk-new-key');
    // No Done step: the dialog is gone and the list refreshed.
    await waitFor(() => expect(screen.queryByTestId('card-modal')).toBeNull());
    const post = apiMock.mock.calls.find(([p, o]) => p === '/api/v2/settings/cards' && (o as RequestInit)?.method === 'POST');
    expect(JSON.parse(String((post![1] as RequestInit).body))).toMatchObject({ provider: 'deepseek', model: 'deepseek-chat', api_key: 'sk-new-key' });
  });

  it('after a refused Save, fixing the key and saving creates exactly one card and closes the dialog (F4)', async () => {
    const saved = makeCard({ id: 'card_new', name: 'New card', key_masked: 'sk-f...0000' });
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') {
        const body = JSON.parse(String(opts.body));
        if (body.api_key === 'sk-new-key-9999') {
          throw new Error(JSON.stringify({ code: 'connection_test_failed', message: 'Invalid API key' }));
        }
        return { card: saved, test: { ok: true, status: 200, code: null, message: 'ok' } };
      }
      if (path === '/api/v2/providers/check') return checkPass();
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('cards-add'));
    await screen.findByTestId('card-modal');
    await addWithKey('deepseek', 'sk-new-key-9999');
    await waitFor(() => expect(screen.getByTestId('connect-key-error')).toBeTruthy());

    // Fix the key and save again: the refused attempt stored nothing, so this
    // is the one card, and a passing save closes the dialog.
    fireEvent.change(screen.getByTestId('connect-key'), { target: { value: 'sk-fixed-0000' } });
    expect(screen.queryByTestId('connect-key-error')).toBeNull();
    fireEvent.click(screen.getByTestId('card-save'));
    await waitFor(() => expect(screen.queryByTestId('card-modal')).toBeNull());
    const posts = apiMock.mock.calls.filter(([p, o]) => p === '/api/v2/settings/cards' && (o as RequestInit)?.method === 'POST');
    const puts = apiMock.mock.calls.filter(([p, o]) => String(p).startsWith('/api/v2/settings/cards/') && (o as RequestInit)?.method === 'PUT');
    expect(posts).toHaveLength(2);
    expect(JSON.parse(String((posts[1][1] as RequestInit).body)).api_key).toBe('sk-fixed-0000');
    expect(puts).toHaveLength(0);
  });

  it('Edit opens the same panel already connected, the provider logo and name as its title', async () => {
    const kimi = makeCard({ id: 'card_km', name: 'Moonshot (Kimi) · kimi-k3', provider: 'moonshot', region: 'china', model: 'kimi-k3' });
    mockCards([DEFAULT_CARD, kimi], (path) => {
      if (path === '/api/v2/providers/models') return { models: ['kimi-k3', 'kimi-k2.6'] };
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('card-edit-card_km'));
    const modal = await screen.findByTestId('card-modal');
    await screen.findByTestId('connect-model-list');
    expect(modal.querySelector('h2')!.textContent).toBe('Moonshot (Kimi)');
    expect(screen.queryByTestId('connect-tile-deepseek')).toBeNull();
    expect(screen.getByTestId('connect-region-china').getAttribute('aria-pressed')).toBe('true');
  });

  it('a revoked sign-in row shows the red mark and "Sign in again", which opens the dialog signing in', async () => {
    const cg = makeCard({
      id: 'card_cg', name: 'ChatGPT · gpt-6.2-sol', provider: 'chatgpt', model: 'gpt-6.2-sol', key_set: false, key_masked: '',
      auth: 'signin', account: { email: 'a@example.com' }, needs_signin: true,
      last_error: { status: 401, code: 'needs_signin', message: 'Disconnected on OpenAI\u2019s side.', at: '2026-11-23T10:00:00+00:00' },
    });
    mockCards([DEFAULT_CARD, cg], (path) => {
      if (path === '/api/v2/providers/chatgpt/signin') return new Promise(() => {});
      return undefined;
    });
    renderList();
    await screen.findByTestId('card-row-card_cg');
    expect(screen.getByTestId('card-health-card_cg').getAttribute('aria-label')).toContain('Disconnected');
    fireEvent.click(screen.getByTestId('card-resignin-card_cg'));
    await screen.findByTestId('connect-signin-waiting');
    expect(apiMock).toHaveBeenCalledWith('/api/v2/providers/chatgpt/signin', expect.objectContaining({ body: JSON.stringify({ card_id: 'card_cg' }) }));
  });

  it('rows lead with the provider logo and show a region chip for two-region providers', async () => {
    const kimi = makeCard({ id: 'card_km', name: 'Moonshot (Kimi) · kimi-k3', provider: 'moonshot', region: 'global', model: 'kimi-k3' });
    mockCards([DEFAULT_CARD, kimi]);
    renderList();
    await screen.findByTestId('card-region-card_km');
    expect(screen.getByTestId('card-region-card_km').textContent).toBe('Global');
    expect(screen.queryByTestId('card-region-card_default')).toBeNull();
  });

  it('offers no vendor-specific sign-in of its own — only "Add provider"', async () => {
    // TokenDance one-click lives INSIDE the Add flow, next to the key field it
    // replaces, so it is invisible to everyone who cannot use it rather than a
    // top-level button they must understand in order to ignore. Prominence for
    // the market that wants it is ordering (zh sorts it first), not visibility.
    mockCards([DEFAULT_CARD]);
    renderList();
    await screen.findByTestId('cards-add');
    expect(screen.queryByTestId('cards-tokendance')).toBeNull();
    expect(screen.queryByTestId('cards-tokendance-msg')).toBeNull();
  });
});

// 2026-09-19: in Project settings the whole page is one <form>, and CardList
// rendered the Add dialog — itself a <form> — inside it. WebKit and Chromium
// stop a nested form's submit event at the ancestor form, so React never saw
// it: no preventDefault, a native GET submit, a full reload to "Select a
// project", and the card POST never sent. The dialog must live outside any
// host form, and its submit must not reach the host's onSubmit.
describe('CardList — the Add dialog inside a host form (project settings)', () => {
  it('saves the card without submitting the host form', async () => {
    const created = makeCard({ id: 'card_new', name: 'OpenCode Go · glm-5.3', provider: 'opencode-go', model: 'glm-5.3' });
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') {
        return { card: created, test: { ok: true, status: 200, code: null, message: 'Connected' } };
      }
      if (path === '/api/v2/providers/check') return checkPass(['glm-5.3']);
      return undefined;
    });
    const hostSubmit = vi.fn((e: React.FormEvent) => e.preventDefault());
    const onChange = vi.fn();
    render(
      <form data-testid="host-form" onSubmit={hostSubmit}>
        <CardList
          mode="project"
          value={null}
          onChange={onChange}
          cards={[DEFAULT_CARD]}
          defaultCardId={DEFAULT_CARD.id}
          providers={PROVIDERS}
          onRefresh={vi.fn()}
          onCardUpdated={vi.fn()}
          showAdd
        />
      </form>,
    );
    fireEvent.click(screen.getByTestId('card-list-add'));
    const modal = await screen.findByTestId('card-modal');
    // Not a DOM descendant of the host form: the browser must deliver the
    // dialog's own submit event instead of swallowing it.
    expect(screen.getByTestId('host-form').contains(modal)).toBe(false);

    await addWithKey('deepseek', 'sk-same-key');

    await waitFor(() => expect(screen.queryByTestId('card-modal')).toBeNull());
    expect(apiMock).toHaveBeenCalledWith('/api/v2/settings/cards', expect.objectContaining({ method: 'POST' }));
    expect(onChange).toHaveBeenCalledWith('card_new');
    expect(hostSubmit).not.toHaveBeenCalled();
  });
});

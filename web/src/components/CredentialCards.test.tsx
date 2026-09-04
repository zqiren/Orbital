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
vi.mock('../config', () => ({ api: apiMock }));

import CredentialCards from './CredentialCards';

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
    return {};
  });
}

function renderList() {
  return render(<CredentialCards providers={PROVIDERS} />);
}

beforeEach(() => {
  apiMock.mockReset();
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
  it('opens the provider form in a modal and keeps it open on the save verdict', async () => {
    mockCards([DEFAULT_CARD], (path, opts) => {
      if (path === '/api/v2/settings/cards' && opts?.method === 'POST') {
        return {
          card: makeCard({ id: 'card_new', name: 'New card' }),
          test: { ok: false, status: 401, code: 'invalid_api_key', message: 'Invalid API key' },
        };
      }
      if (path === '/api/v2/settings/api-key/status') return { configured: true, source: 'keychain' };
      return undefined;
    });
    renderList();
    fireEvent.click(await screen.findByTestId('cards-add'));
    await screen.findByTestId('card-modal');

    fireEvent.change(await screen.findByPlaceholderText(/model name/), {
      target: { value: 'deepseek-chat' },
    });
    fireEvent.click(screen.getByTestId('card-save'));

    // The verdict is the thing the user opened the modal for, so the modal
    // does not close itself over it (D9: the card is saved regardless).
    expect((await screen.findByTestId('card-save-test')).textContent).toContain('Invalid API key');
    expect(screen.getByTestId('card-modal')).toBeTruthy();
    fireEvent.click(screen.getByTestId('card-modal-done'));
    await waitFor(() => expect(screen.queryByTestId('card-modal')).toBeNull());
  });

  it('the TokenDance one-click button provisions a card and refreshes the list', async () => {
    mockCards([DEFAULT_CARD], (path, opts) =>
      path === '/api/v2/providers/tokendance/signin' && opts?.method === 'POST'
        ? {
            card: makeCard({ id: 'card_td', name: 'TokenDance · deepseek-v4-flash' }),
            test: { ok: true, status: null, code: null, message: 'ok' },
            default_card_id: 'card_default',
            api_key_set: true,
            api_key_masked: 'sk-t...abcd',
          }
        : undefined,
    );
    renderList();
    fireEvent.click(await screen.findByTestId('cards-tokendance'));
    await waitFor(() =>
      expect(screen.getByTestId('cards-tokendance-msg').textContent).toContain(
        'API key created and saved.',
      ),
    );
  });
});

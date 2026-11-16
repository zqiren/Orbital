// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Task sources panel against a mocked daemon (the contract's shapes): the
 * guide while nothing is connected, the connect list from the catalogue, a
 * form whose check fails on a field, a source's switch / catch-up / failure.
 */

import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../../config', async (orig) => {
  const actual = await orig<typeof import('../../config')>();
  return { ...actual, api: apiMock, isRelayMode: false, BASE_URL: '' };
});
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({ on: vi.fn(), off: vi.fn(), connectionState: 'connected', subscribe: vi.fn() }),
}));

import SourcesPanel from './SourcesPanel';
import { __resetSources, refreshSources } from './sourcesStore';
import { ApiError } from '../../config';
import { CALENDAR, EMAIL, FEISHU, GITHUB, TELEGRAM, WEBSITE, detail, item, summary } from './testFixtures';
import type { Project } from '../../types';

type Handler = (init?: { method?: string; body?: string }) => unknown;
let routes: Record<string, Handler> = {};
const calls: Array<{ key: string; body: unknown }> = [];

function setApi(r: Record<string, Handler>) {
  routes = r;
  apiMock.mockImplementation((path: string, init?: { method?: string; body?: string }) => {
    const key = `${init?.method ?? 'GET'} ${path}`;
    calls.push({ key, body: init?.body ? JSON.parse(init.body) : undefined });
    const h = routes[key];
    if (!h) return Promise.resolve({});
    try { return Promise.resolve(h(init)); } catch (e) { return Promise.reject(e); }
  });
}

const CATALOG = {
  platform: 'darwin',
  types: [GITHUB, { ...GITHUB, type: 'gitlab', connect: { ...GITHUB.connect, cli: 'glab', host: 'gitlab.com' } }, EMAIL, CALENDAR, TELEGRAM,
    { ...TELEGRAM, type: 'slack', available: false, unavailable_reason: 'not_built' }, FEISHU, WEBSITE],
};
const SIGNED_IN = { type: 'github', cli: 'gh', state: 'signed_in', accounts: [{ login: 'mira', host: 'github.com', active: true }], sign_in_command: 'gh auth login', install_command: 'brew install gh' };
const SIGNED_OUT = { ...SIGNED_IN, type: 'gitlab', cli: 'glab', state: 'signed_out', accounts: [] };
const PROJECTS = [{ project_id: 'p1', name: 'client-portal', workspace: '/w' } as Project];

async function renderPanel(sources: ReturnType<typeof summary>[], extra: Record<string, Handler> = {}) {
  setApi({
    'GET /api/v2/sources/catalog': () => CATALOG,
    'GET /api/v2/sources': () => ({ sources, waiting: 0 }),
    'GET /api/v2/sources/accounts?type=github': () => SIGNED_IN,
    'GET /api/v2/sources/accounts?type=gitlab': () => SIGNED_OUT,
    ...extra,
  });
  await act(async () => { await refreshSources(); });
  const onOpenSession = vi.fn();
  const onClose = vi.fn();
  await act(async () => {
    render(<SourcesPanel phone={false} projects={PROJECTS} focus={null} onClose={onClose} onOpenSession={onOpenSession} />);
  });
  return { onOpenSession, onClose };
}

beforeEach(() => { apiMock.mockReset(); calls.length = 0; __resetSources(); });
afterEach(() => cleanup());

describe('the guide (nothing connected)', () => {
  it('three beats, then what starts with nothing filled in: a signed-in CLI and Calendar; not a signed-out one', async () => {
    await renderPanel([]);
    const guide = await screen.findByTestId('src-guide');
    expect(guide).toHaveTextContent('Something new in a tool');
    expect(guide).toHaveTextContent('It becomes a Workbench task');
    await waitFor(() => expect(within(guide).getAllByTestId('src-kind-row').map((r) => r.dataset.type)).toEqual(['github', 'calendar']));
    expect(within(guide).getAllByTestId('src-kind-row')[0]).toHaveTextContent('mira');
    expect(screen.queryByTestId('src-plus')).toBeNull();
    fireEvent.click(screen.getByTestId('src-guide-more'));
    expect(screen.getByTestId('src-add')).toBeInTheDocument();
  });
});

describe('the connect list', () => {
  it('groups my accounts, my bots and the website check; a type this daemon cannot run is not listed', async () => {
    await renderPanel([summary({ id: 'a' })]);
    fireEvent.click(screen.getByTestId('src-plus'));
    const rows = screen.getAllByTestId('src-kind-row');
    expect(rows.map((r) => r.dataset.type)).toEqual(['github', 'gitlab', 'email', 'calendar', 'telegram', 'feishu', 'website']);
    expect(screen.queryByText('Coming soon')).toBeNull();
    await waitFor(() => expect(rows.find((r) => r.dataset.type === 'gitlab')).toHaveTextContent('Not signed in'));
  });
});

describe('a connect form', () => {
  it('a refused token marks its field; nothing is created; fixing it and pressing again creates the source', async () => {
    let attempt = 0;
    const created = detail({ id: 'src_t', type: 'telegram', family: 'bot', label: '@mira_bot', kinds: { dm: {}, at: {} } });
    await renderPanel([summary({ id: 'a' })], {
      'POST /api/v2/sources': () => {
        attempt += 1;
        if (attempt === 1) throw new ApiError(400, JSON.stringify({ code: 'verify_failed', field: 'token', reason: 'rejected', message: 'x' }));
        return created;
      },
    });
    fireEvent.click(screen.getByTestId('src-plus'));
    fireEvent.click(screen.getAllByTestId('src-kind-row').find((r) => r.dataset.type === 'telegram')!);
    const form = screen.getByTestId('src-connect');
    expect(within(form).getByTestId('src-agent-card')).toBeInTheDocument();
    expect(within(form).getByTestId('src-setup-link')).toHaveAttribute('href', 'https://t.me/BotFather');
    const submit = within(form).getByTestId('src-submit');
    expect(submit).toBeDisabled();
    const token = within(form).getByTestId('src-field-token');
    expect(token).toHaveAttribute('type', 'password');
    fireEvent.change(token, { target: { value: 'bad' } });
    await act(async () => { fireEvent.click(submit); });
    expect(within(form).getByTestId('src-field-error')).toHaveTextContent('Telegram rejected this token');
    expect(calls.filter((c) => c.key === 'POST /api/v2/sources')[0].body).toMatchObject({ type: 'telegram', secrets: { token: 'bad' }, kinds: { dm: {}, at: {} } });
    fireEvent.change(token, { target: { value: 'good' } });
    expect(within(form).queryByTestId('src-field-error')).toBeNull();
    await act(async () => { fireEvent.click(submit); });
    // Back on the list, the new source first.
    await waitFor(() => expect(screen.getByTestId('src-list')).toBeInTheDocument());
    expect(screen.getAllByTestId('src-row')[0]).toHaveAttribute('data-id', 'src_t');
  });

  it('a CLI form takes the active account; the website check needs only an address', async () => {
    await renderPanel([summary({ id: 'a' })], { 'POST /api/v2/sources': () => detail({ id: 'n' }) });
    fireEvent.click(screen.getByTestId('src-plus'));
    fireEvent.click(screen.getAllByTestId('src-kind-row').find((r) => r.dataset.type === 'website')!);
    const form = screen.getByTestId('src-connect');
    expect(within(form).queryByTestId('src-kinds')).toBeNull();
    expect(within(form).queryByTestId('src-agent-card')).toBeNull();
    fireEvent.change(within(form).getByTestId('src-field-url'), { target: { value: 'shop.acme.com' } });
    expect(within(form).getByTestId('src-submit')).toBeEnabled();
    fireEvent.click(within(form).getByTestId('src-back'));
    fireEvent.click(screen.getAllByTestId('src-kind-row').find((r) => r.dataset.type === 'github')!);
    await waitFor(() => expect(screen.getByTestId('src-account-row')).toHaveTextContent('mira'));
    await waitFor(() => expect(screen.getByTestId('src-submit')).toBeEnabled());
    await act(async () => { fireEvent.click(screen.getByTestId('src-submit')); });
    expect(calls.find((c) => c.key === 'POST /api/v2/sources')!.body).toMatchObject({ type: 'github', account: 'mira', kinds: { assigned: {} } });
  });
});

describe('a source', () => {
  it('history: titles open sessions, the queue is one row, held items wait for one catch-up', async () => {
    const d = detail({
      id: 'a', catch_up: { count: 1 },
      history: [
        item({ id: 'h', state: 'held', title: 'while paused', arrived_at: '2026-11-16T08:00:00Z' }),
        item({ id: 'w', state: 'waiting', title: 'next', arrived_at: '2026-11-16T07:30:00Z' }),
        item({ id: 'r', state: 'routed', title: 'done one', arrived_at: '2026-11-16T07:00:00Z', count: 3,
          updates: [{ title: 'u1', sender: null, at: '2026-11-16T07:00:00Z' }] }),
      ],
    });
    const { onOpenSession } = await renderPanel([summary({ id: 'a', catch_up: { count: 1 } })], {
      'GET /api/v2/sources/a': () => d,
      'POST /api/v2/sources/a/catch-up': () => ({ ...d, catch_up: null }),
    });
    await act(async () => { fireEvent.click(screen.getByTestId('src-row')); });
    const hist = await screen.findByTestId('src-history');
    expect(within(hist).getByTestId('src-wait')).toHaveTextContent('1 waiting');
    const routed = within(hist).getAllByTestId('src-arv').find((r) => r.dataset.id === 'r')!;
    fireEvent.click(within(routed).getByTestId('src-arv-title'));
    expect(onOpenSession).toHaveBeenCalledWith('p1', 's1');
    expect(within(routed).getByTestId('src-arv-state')).toHaveTextContent('Running');
    fireEvent.click(within(routed).getByTestId('src-arv-updates'));
    expect(routed).toHaveTextContent('u1');
    await act(async () => { fireEvent.click(within(hist).getByText('Catch up')); });
    expect(calls.some((c) => c.key === 'POST /api/v2/sources/a/catch-up')).toBe(true);
  });

  it('a failure shows only the actions the daemon offers; 再检查 runs the check', async () => {
    const failed = detail({ id: 'a', status: 'failed', failure: { code: 'repo_not_found', params: { repo: 'acme/old' }, at: null, actions: ['check_again', 'remove'] } });
    await renderPanel([failed], { 'GET /api/v2/sources/a': () => failed, 'POST /api/v2/sources/a/check': () => detail({ id: 'a' }) });
    await act(async () => { fireEvent.click(screen.getByTestId('src-row')); });
    const fail = await screen.findByTestId('src-fail');
    expect(fail).toHaveTextContent("Can't find acme/old");
    expect(within(fail).queryByTestId('src-fail-reconnect')).toBeNull();
    await act(async () => { fireEvent.click(within(fail).getByTestId('src-fail-check')); });
    await waitFor(() => expect(screen.queryByTestId('src-fail')).toBeNull());
  });

  it('the switch pauses; ⋯ → 删除 asks once, then removes and returns to the list', async () => {
    const d = detail({ id: 'a' });
    await renderPanel([d, summary({ id: 'b' })], {
      'GET /api/v2/sources/a': () => d,
      'PATCH /api/v2/sources/a': (init) => ({ ...d, ...JSON.parse(init!.body!) }),
      'DELETE /api/v2/sources/a': () => ({ status: 'deleted', id: 'a' }),
    });
    await act(async () => { fireEvent.click(screen.getAllByTestId('src-row')[0]); });
    await screen.findByTestId('src-detail');
    await act(async () => { fireEvent.click(screen.getByTestId('src-detail-switch')); });
    expect(calls.find((c) => c.key === 'PATCH /api/v2/sources/a')!.body).toEqual({ on: false });
    fireEvent.click(screen.getByTestId('src-more'));
    fireEvent.click(screen.getByTestId('src-menu-delete'));
    await act(async () => { fireEvent.click(screen.getByTestId('src-delete-yes')); });
    expect(screen.getAllByTestId('src-row').map((r) => r.dataset.id)).toEqual(['b']);
  });

  it('a source an agent registered arrives off with 新 and where it came from', async () => {
    await renderPanel([summary({
      id: 'x', type: 'command', family: 'external', name: 'HN mentions', on: false, is_new: true, label: null,
      registered_by: { by: 'agent', agent: 'claude-code', project_id: 'p1', project_name: 'client-portal', session_id: null, at: null },
    })]);
    const r = screen.getByTestId('src-row');
    expect(r).toHaveTextContent('HN mentions');
    expect(r).toHaveTextContent('client-portal');
    expect(within(r).getByTestId('src-row-new')).toHaveTextContent('New');
    expect(within(r).getByTestId('src-row-switch')).toHaveAttribute('aria-checked', 'false');
  });
});

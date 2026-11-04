// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 154 — the Workbench as home: composer on top, two groups (等你 then
 * 进行中) with one summary line, cards that open, confirm and take a reply,
 * and the send flow (route on 自动 → mint → the chat's own inject). The api
 * client is mocked; the feed store is the real module store.
 */

import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock, isRelayMode: false }));

// The reading pane embeds the real ChatView and FilePreview; here they are
// stubs (the daemon smoke drives the real ones). The ChatView stub opens a
// document through the pane's OpenPathContext and reports a send.
const chatView = vi.hoisted(() => ({ last: null as Record<string, unknown> | null, mounts: 0 }));
vi.mock('./ChatView', async () => {
  const React = await import('react');
  const { OpenPathContext } = await import('./filePreviewContext');
  function ChatViewStub(props: Record<string, unknown>) {
    React.useEffect(() => { chatView.mounts += 1; }, []);
    chatView.last = props;
    const open = React.useContext(OpenPathContext);
    return (
      <div data-testid="chatview-stub" data-session={String(props.sessionId)}>
        <button type="button" data-testid="stub-open-doc" onClick={() => open?.('docs/a.md')}>doc</button>
        <button type="button" data-testid="stub-send" onClick={() => (props.onUserSend as (() => void) | undefined)?.()}>send</button>
      </div>
    );
  }
  return { default: ChatViewStub };
});
vi.mock('./FilePreview', () => ({
  default: (props: { selectedPath: string | null; panelHeader?: { onBack: () => void }; headerExtras?: unknown }) => (
    <div data-testid="file-preview-stub">
      <button type="button" data-testid="stub-doc-back" onClick={props.panelHeader?.onBack}>back</button>
      <span>{props.selectedPath}</span>
      {props.headerExtras as never}
    </div>
  ),
}));

import WorkbenchPage from './WorkbenchPage';
import { __resetWorkbenchFeed, refreshFeed } from '../hooks/useWorkbenchFeed';
import { __resetCheckedMarks } from '../hooks/useSeenReplies';
import { DRAFT_KEY } from './workbench/sendTask';
import type { Project, WorkbenchFeedRow } from '../types';

const NOW = Date.now();
const iso = (minsAgo: number) => new Date(NOW - minsAgo * 60_000).toISOString();

const PROJECTS: Project[] = [
  { project_id: 'proj-a', name: 'Marketing', workspace: '/ws/a' } as Project,
  { project_id: 'proj-b', name: 'Ops', workspace: '/ws/b' } as Project,
  { project_id: 'proj-x', name: 'Private', workspace: '/ws/x', workbench_exclude_global: true } as Project,
];
const AGENTS = [{ slug: 'codex', name: 'Codex' }, { slug: 'claude-code', name: 'Claude Code' }];

function row(over: Partial<WorkbenchFeedRow>): WorkbenchFeedRow {
  return {
    project_id: 'proj-a', session_id: 's1', session_uuid: 's1', name: 'Fix the cwd', agent: null,
    status: 'idle', worker_running: false, last_user_at: iso(30), last_reply_at: iso(20),
    checked_reply_at: null, ask: null, result_line: null, stop: null, ...over,
  };
}

const FEED: WorkbenchFeedRow[] = [
  row({ session_id: 'done', session_uuid: 'done', name: 'Hero copy', agent: 'codex', result_line: 'Changed 3 places' }),
  row({ session_id: 'asked', session_uuid: 'asked', name: 'Sync OSS', ask: { id: 'a1b2c3', text: 'Sync the notes too?', due: null, created: '2026-11-04', overdue: false }, last_reply_at: iso(5) }),
  row({ session_id: 'limit', session_uuid: 'limit', name: 'Windows smoke', agent: 'codex', stop: { cause: 'usage_limit', resets_at: null }, last_user_at: iso(125), last_reply_at: iso(120) }),
  row({ project_id: 'proj-b', session_id: 'run', session_uuid: 'run', name: 'Weekly report', status: 'running', last_reply_at: null, last_user_at: iso(1) }),
  row({ project_id: 'proj-b', session_id: 'appr', session_uuid: 'appr', name: 'Deploy', status: 'pending_approval', last_reply_at: null }),
  row({ session_id: null, session_uuid: null, name: null, last_user_at: null, last_reply_at: null, ask: { id: 'e5e5e5', text: 'Renew the domain', due: '2026-11-02', created: '2026-11-01', overdue: true } }),
];

let routes: Record<string, (init?: { method?: string; body?: string }) => unknown>;

function setApi(map: typeof routes) {
  routes = map;
  apiMock.mockImplementation((path: string, init?: { method?: string; body?: string }) => {
    const key = `${init?.method ?? 'GET'} ${path}`;
    const handler = routes[key];
    if (!handler) return Promise.resolve({});
    try {
      return Promise.resolve(handler(init));
    } catch (e) {
      return Promise.reject(e);
    }
  });
}

async function renderPage(feed: WorkbenchFeedRow[] = FEED, extra: typeof routes = {}, phone = false) {
  setApi({ 'GET /api/v2/workbench/feed': () => ({ rows: feed, window_days: 14 }), ...extra });
  await act(async () => { await refreshFeed(); });
  const setRoute = vi.fn();
  render(<WorkbenchPage setRoute={setRoute} projects={PROJECTS} agents={AGENTS} phone={phone} />);
  return { setRoute };
}

function cards() {
  return screen.queryAllByTestId('wb-card');
}

function cardWith(text: string) {
  const el = cards().find((c) => c.textContent?.includes(text));
  if (!el) throw new Error(`no card with ${text}`);
  return el;
}

afterEach(cleanup);
beforeEach(() => {
  apiMock.mockReset();
  __resetWorkbenchFeed();
  __resetCheckedMarks();
  localStorage.clear();
});

describe('Workbench — list', () => {
  it('two groups, 等你 before 进行中, with one summary line and no headers', async () => {
    await renderPage();
    expect(screen.getByTestId('workbench-summary')).toHaveTextContent('5 waiting on you · 1 running');
    const order = cards().map((c) => c.getAttribute('data-dot'));
    // Overdue ask first, then amber newest first, then blue; running last.
    expect(order).toEqual(['need', 'need', 'need', 'need', 'new', 'run']);
    expect(cards()[0]).toHaveTextContent('Renew the domain');
    expect(cards()[cards().length - 1]).toHaveTextContent('Weekly report');
  });

  it('a card reads logo · project · detail', async () => {
    await renderPage();
    const done = cardWith('Hero copy');
    expect(within(done).getByTestId('wb-card-meta')).toHaveTextContent('Marketing · Changed 3 places');
    expect(within(done).getByTitle('Codex')).toBeInTheDocument();
    expect(within(cardWith('Windows smoke')).getByTestId('wb-card-meta')).toHaveTextContent('Usage limit');
    expect(within(cardWith('Deploy')).getByTestId('wb-card-meta')).toHaveTextContent('Needs approval');
    // An ask: the question quoted as the title, the task in the muted line.
    const asked = cardWith('Sync the notes too?');
    expect(within(asked).getByTestId('wb-card-title')).toHaveTextContent('“Sync the notes too?”');
    expect(within(asked).getByTestId('wb-card-meta')).toHaveTextContent('Sync OSS');
  });

  it('no card has an open-session icon: ✓ is the only icon, on waiting cards only', async () => {
    await renderPage();
    expect(screen.queryAllByTestId('wb-card-open')).toHaveLength(0);
    for (const name of ['Weekly report', 'Deploy']) {
      const c = cardWith(name);
      expect(within(c).queryByTestId('wb-card-confirm')).toBeNull();
      expect(within(c).queryByTestId('wb-card-reply')).toBeNull();
      expect(within(c).queryAllByRole('button').map((b) => b.getAttribute('data-testid'))).toEqual(['wb-card-title']);
    }
    const done = cardWith('Hero copy');
    expect(within(done).getByTestId('wb-card-confirm')).toBeInTheDocument();
    expect(within(done).getByTestId('wb-card-reply')).toBeInTheDocument();
  });

  it('a finished automation run reads "<automation> · <the run\'s first line>", the logo is the agent that ran it', async () => {
    await renderPage([
      row({
        session_id: 'run1', session_uuid: 'run1', trigger_type: 'file_watch', agent: 'codex',
        name: "[Triggered by file_watch '整理收件箱']\n\nChanged files:…",
        result_line: 'PONG',
      }),
    ]);
    const c = cardWith('整理收件箱');
    expect(within(c).getByTestId('wb-card-title')).toHaveTextContent(/^整理收件箱 · PONG$/);
    // The first line is in the title, so the muted line does not repeat it.
    expect(within(c).getByTestId('wb-card-meta')).toHaveTextContent(/^Marketing$/);
    expect(within(c).getByTitle('Codex')).toBeInTheDocument();
    expect(c.getAttribute('data-dot')).toBe('new');
  });

  it('a running automation run is in 进行中 under the automation\'s name', async () => {
    await renderPage([
      row({
        session_id: 'run2', session_uuid: 'run2', trigger_type: 'schedule', status: 'running',
        name: "[Triggered by schedule 'Daily digest' (every day at 9)]…", last_reply_at: null,
      }),
    ]);
    const c = cardWith('Daily digest');
    expect(within(c).getByTestId('wb-card-title')).toHaveTextContent(/^Daily digest$/);
    expect(c.getAttribute('data-dot')).toBe('run');
    expect(screen.getByTestId('workbench-summary')).toHaveTextContent('1 running');
  });

  it('an automation run with no first line yet shows the stop in the muted line', async () => {
    await renderPage([
      row({
        session_id: 'run3', session_uuid: 'run3', trigger_type: 'schedule',
        name: "[Triggered by schedule 'Nightly build'…", stop: { cause: 'usage_limit', resets_at: null },
      }),
    ]);
    const c = cardWith('Nightly build');
    expect(within(c).getByTestId('wb-card-title')).toHaveTextContent(/^Nightly build$/);
    expect(within(c).getByTestId('wb-card-meta')).toHaveTextContent('Usage limit');
  });

  it('empty: the composer alone, its placeholder carries the sentence', async () => {
    await renderPage([]);
    expect(screen.queryByTestId('workbench-summary')).toBeNull();
    expect(cards()).toHaveLength(0);
    expect(screen.getByTestId('wb-composer-input')).toHaveAttribute(
      'placeholder', 'Type a task. Orbital picks the project and the agent.',
    );
  });

  it('non-empty: the placeholder is the question', async () => {
    await renderPage();
    expect(screen.getByTestId('wb-composer-input')).toHaveAttribute('placeholder', 'What do you want done?');
  });
});

describe('Workbench — card actions', () => {
  it('the title opens the session', async () => {
    const { setRoute } = await renderPage();
    fireEvent.click(within(cardWith('Hero copy')).getByTestId('wb-card-title'));
    expect(setRoute).toHaveBeenCalledWith({ name: 'project', projectId: 'proj-a', sessionId: 'done' });
  });

  it('✓ on a finished card checks its reply in the daemon and the card leaves', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderPage();
      fireEvent.click(within(cardWith('Hero copy')).getByTestId('wb-card-confirm'));
      await act(async () => { vi.advanceTimersByTime(300); });
      const call = apiMock.mock.calls.find((c) => c[0] === '/api/v2/workbench/checked');
      expect(call).toBeTruthy();
      expect(JSON.parse(call![1].body)).toMatchObject({ project_id: 'proj-a', session_id: 'done' });
      await waitFor(() => expect(cards().some((c) => c.textContent?.includes('Hero copy'))).toBe(false));
    } finally {
      vi.useRealTimers();
    }
  });

  it('✓ on an ask closes it by:user through the exit route', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderPage();
      fireEvent.click(within(cardWith('Sync the notes too?')).getByTestId('wb-card-confirm'));
      await act(async () => { vi.advanceTimersByTime(300); });
      expect(apiMock).toHaveBeenCalledWith('/api/v2/workbench/proj-a/entries/a1b2c3/exit', {
        method: 'POST', body: JSON.stringify({ kind: 'fulfilled' }),
      });
      await waitFor(() => expect(cards().some((c) => c.textContent?.includes('Sync the notes too?'))).toBe(false));
    } finally {
      vi.useRealTimers();
    }
  });

  it('a reply to an ask goes into that session in the chat\'s exact wording', async () => {
    await renderPage();
    const asked = cardWith('Sync the notes too?');
    fireEvent.change(within(asked).getByTestId('wb-card-reply'), { target: { value: 'yes please' } });
    fireEvent.keyDown(within(asked).getByTestId('wb-card-reply'), { key: 'Enter' });
    await waitFor(() => expect(apiMock.mock.calls.some((c) => c[0] === '/api/v2/agents/proj-a/inject')).toBe(true));
    const body = JSON.parse(apiMock.mock.calls.find((c) => c[0] === '/api/v2/agents/proj-a/inject')![1].body);
    expect(body).toMatchObject({
      content: '[Answer to ask a1b2c3, already closed: "Sync the notes too?"]\nyes please',
      session_id: 'asked', ask_id: 'a1b2c3', ask_answer: 'yes please',
    });
    // 2026-11-04 (supersedes "keeps its place in 等你"): the card moves to
    // 进行中 at once.
    expect(cardWith('Sync OSS').getAttribute('data-dot')).toBe('run');
  });

  it('a card answered from its field sits in 进行中 until the turn ends, then comes back to 等你 unchecked', async () => {
    await renderPage();
    const done = cardWith('Hero copy');
    fireEvent.change(within(done).getByTestId('wb-card-reply'), { target: { value: 'one more pass' } });
    fireEvent.keyDown(within(done).getByTestId('wb-card-reply'), { key: 'Enter' });
    await waitFor(() => expect(cardWith('Hero copy').getAttribute('data-dot')).toBe('run'));
    expect(screen.getByTestId('workbench-summary')).toHaveTextContent('2 running');
    // The turn runs…
    const running = row({ session_id: 'done', session_uuid: 'done', name: 'Hero copy', agent: 'codex',
      status: 'running', last_user_at: iso(0), last_reply_at: iso(20) });
    routes['GET /api/v2/workbench/feed'] = () => ({ rows: [running], window_days: 14 });
    await act(async () => { await refreshFeed(); });
    expect(cardWith('Hero copy').getAttribute('data-dot')).toBe('run');
    // …and ends with a new reply: back to 等你, blue (finished, unchecked).
    const finished = row({ session_id: 'done', session_uuid: 'done', name: 'Hero copy', agent: 'codex',
      result_line: 'Second pass done', last_user_at: iso(1), last_reply_at: new Date().toISOString() });
    routes['GET /api/v2/workbench/feed'] = () => ({ rows: [finished], window_days: 14 });
    await act(async () => { await refreshFeed(); });
    expect(cardWith('Hero copy').getAttribute('data-dot')).toBe('new');
    expect(screen.getByTestId('workbench-summary')).toHaveTextContent('1 waiting on you');
  });

  it('a session-less ask opens through the old doorway (mint + ask quoted)', async () => {
    const { setRoute } = await renderPage(FEED, {
      'POST /api/v2/agents/proj-a/new-session': () => ({ session_id: 'minted-1' }),
    });
    fireEvent.click(within(cardWith('Renew the domain')).getByTestId('wb-card-title'));
    await waitFor(() => expect(setRoute).toHaveBeenCalledWith({
      name: 'project', projectId: 'proj-a', sessionId: 'minted-1',
      askReply: { id: 'e5e5e5', text: 'Renew the domain' },
    }));
  });
});

describe('Workbench — send', () => {
  it('自动: routes, mints, injects to the minted pin; the card appears and the chips reset', async () => {
    let injected: Record<string, unknown> | null = null;
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: 'proj-b', by: 'model' }),
      'POST /api/v2/agents/proj-b/new-session': () => ({ session_id: 'new-1', pinned_target: 'codex' }),
      'POST /api/v2/agents/proj-b/inject': (init) => { injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    const input = screen.getByTestId('wb-composer-input');
    fireEvent.change(input, { target: { value: 'Write the weekly report' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    // The pending card shows at once, with 正在开始….
    expect(cardWith('Write the weekly report')).toHaveTextContent('Starting…');
    await waitFor(() => expect(injected).not.toBeNull());
    expect(injected).toMatchObject({ content: 'Write the weekly report', target: 'codex', session_id: 'new-1', pinned: true });
    expect(input).toHaveValue('');
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Auto');
    expect(localStorage.getItem(DRAFT_KEY)).toBeNull();
  });

  it('a picked project and Orbital skip the route and send to the manager', async () => {
    let injected: Record<string, unknown> | null = null;
    await renderPage([], {
      'POST /api/v2/agents/proj-a/new-session': () => ({ session_id: 'new-2', pinned_target: 'codex' }),
      'POST /api/v2/agents/proj-a/inject': (init) => { injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    fireEvent.click(screen.getByTestId('wb-chip-project'));
    const pop = screen.getByTestId('wb-pop-project');
    // Excluded projects are not on the Workbench at all.
    expect(within(pop).queryByText('Private')).toBeNull();
    fireEvent.click(within(pop).getByText('Marketing'));
    fireEvent.click(screen.getByTestId('wb-chip-agent'));
    fireEvent.click(within(screen.getByTestId('wb-pop-agent')).getByText('Orbital'));
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'hello' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(injected).not.toBeNull());
    expect(apiMock.mock.calls.some((c) => c[0] === '/api/v2/workbench/route')).toBe(false);
    expect(injected).toMatchObject({ content: 'hello', session_id: 'new-2' });
    expect(injected).not.toHaveProperty('target');
  });

  it('a failed mint puts the text back and shows the open error', async () => {
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: 'proj-b', by: 'fallback' }),
      'POST /api/v2/agents/proj-b/new-session': () => { throw new Error('boom'); },
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'keep me' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('workbench-open-error')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId('wb-composer-input')).toHaveValue('keep me'));
    expect(cards()).toHaveLength(0);
  });

  it('an unsent draft from the last page load comes back into the composer', async () => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ text: 'half sent', nonce: 'n', project_id: null, session_id: null }));
    await renderPage([]);
    await waitFor(() => expect(screen.getByTestId('wb-composer-input')).toHaveValue('half sent'));
  });

  it('a draft whose session exists is dropped (the text is already there)', async () => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ text: 'sent', nonce: 'n', project_id: 'proj-a', session_id: 's9' }));
    await renderPage([], {
      'GET /api/v2/projects/proj-a/sessions': () => ({ sessions: [{ session_id: 's9', session_uuid: 's9' }] }),
    });
    await waitFor(() => expect(localStorage.getItem(DRAFT_KEY)).toBeNull());
    expect(screen.getByTestId('wb-composer-input')).toHaveValue('');
  });
});

// "0 s ago" must not sit on a card for half a minute: under a minute the
// card's relative time refreshes at least every 10 s.
describe('Workbench — relative time ticks', () => {
  it('a card a few seconds old moves on without a feed refresh', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: false, toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
    try {
      const start = new Date('2026-11-04T10:00:00Z').getTime();
      vi.setSystemTime(start);
      const fresh = row({
        project_id: 'proj-b', session_id: 'run', session_uuid: 'run', name: 'Weekly report', status: 'running',
        last_reply_at: null, last_user_at: new Date(start).toISOString(),
      });
      await renderPage([fresh]);
      const time = () => cardWith('Weekly report').querySelector('.wb-time')?.textContent;
      expect(time()).toBe('0s ago');
      await act(async () => { vi.advanceTimersByTime(10_000); });
      expect(time()).toBe('10s ago');
      await act(async () => { vi.advanceTimersByTime(10_000); });
      expect(time()).toBe('20s ago');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('Workbench — reading pane (2026-11-04)', () => {
  beforeEach(() => {
    chatView.last = null;
    chatView.mounts = 0;
  });

  const pane = () => screen.queryByTestId('wb-pane');

  it('nothing selected: no pane, the page is unchanged', async () => {
    await renderPage();
    expect(pane()).toBeNull();
    expect(screen.queryByTestId('chatview-stub')).toBeNull();
    expect(cards().some((c) => c.getAttribute('data-selected') === 'true')).toBe(false);
  });

  it('a click on a card body selects it and opens that session\'s chat in the pane', async () => {
    await renderPage();
    fireEvent.click(cardWith('Hero copy'));
    expect(cardWith('Hero copy').getAttribute('data-selected')).toBe('true');
    const p = pane()!;
    expect(p).not.toBeNull();
    expect(within(p).getByTestId('wb-pane-title')).toHaveTextContent('Hero copy');
    expect(within(p).getByTestId('chatview-stub').getAttribute('data-session')).toBe('done');
    expect(chatView.last).toMatchObject({ projectId: 'proj-a', sessionId: 'done', checksOnView: false });
    expect(within(p).getByTestId('wb-pane-open')).toHaveTextContent('Open session');
    expect(within(p).getByTestId('wb-pane-close')).toBeInTheDocument();
  });

  it('the title, ✓ and the reply field do not select', async () => {
    const { setRoute } = await renderPage();
    const done = cardWith('Hero copy');
    fireEvent.click(within(done).getByTestId('wb-card-reply'));
    expect(pane()).toBeNull();
    fireEvent.click(within(done).getByTestId('wb-card-title'));
    expect(pane()).toBeNull();
    expect(setRoute).toHaveBeenCalledWith({ name: 'project', projectId: 'proj-a', sessionId: 'done' });
  });

  it('a second click on the selected card closes the pane, and so does ×', async () => {
    await renderPage();
    fireEvent.click(cardWith('Hero copy'));
    expect(pane()).not.toBeNull();
    fireEvent.click(cardWith('Hero copy'));
    expect(pane()).toBeNull();
    fireEvent.click(cardWith('Weekly report'));
    expect(within(pane()!).getByTestId('wb-pane-title')).toHaveTextContent('Weekly report');
    fireEvent.click(screen.getByTestId('wb-pane-close'));
    expect(pane()).toBeNull();
  });

  it('打开会话 goes to the full session', async () => {
    const { setRoute } = await renderPage();
    fireEvent.click(cardWith('Hero copy'));
    fireEvent.click(screen.getByTestId('wb-pane-open'));
    expect(setRoute).toHaveBeenCalledWith({ name: 'project', projectId: 'proj-a', sessionId: 'done' });
  });

  it('a session-less ask has no session to show, so it does not select', async () => {
    await renderPage();
    fireEvent.click(cardWith('Renew the domain'));
    expect(pane()).toBeNull();
  });

  it('a send from the pane moves the card to 进行中 at once', async () => {
    await renderPage();
    fireEvent.click(cardWith('Hero copy'));
    expect(cardWith('Hero copy').getAttribute('data-dot')).toBe('new');
    fireEvent.click(screen.getByTestId('stub-send'));
    expect(cardWith('Hero copy').getAttribute('data-dot')).toBe('run');
    // Still selected, pane still open.
    expect(cardWith('Hero copy').getAttribute('data-selected')).toBe('true');
    expect(pane()).not.toBeNull();
  });

  it('a document opens OVER the pane with a back chevron; back returns to the same chat', async () => {
    await renderPage(FEED, {
      'GET /api/v2/projects/proj-a/files/content?path=docs%2Fa.md&document_preview=1': () => ({
        path: 'docs/a.md', content: '# A', size: 3, mime: 'text/markdown', truncated: false,
      }),
    });
    fireEvent.click(cardWith('Hero copy'));
    fireEvent.click(screen.getByTestId('stub-open-doc'));
    const doc = await screen.findByTestId('wb-pane-doc');
    expect(within(doc).getByTestId('file-preview-stub')).toHaveTextContent('docs/a.md');
    // Over the pane, not a second panel: inside it, and the chat stays mounted.
    expect(pane()!.contains(doc)).toBe(true);
    expect(screen.getAllByTestId('chatview-stub')).toHaveLength(1);
    fireEvent.click(within(doc).getByTestId('stub-doc-back'));
    expect(screen.queryByTestId('wb-pane-doc')).toBeNull();
    expect(chatView.mounts).toBe(1);
  });

  it('maximize: the document covers the window except the edge strip; again restores it', async () => {
    await renderPage(FEED, {
      'GET /api/v2/projects/proj-a/files/content?path=docs%2Fa.md&document_preview=1': () => ({
        path: 'docs/a.md', content: '# A', size: 3, mime: 'text/markdown', truncated: false,
      }),
    });
    fireEvent.click(cardWith('Hero copy'));
    fireEvent.click(screen.getByTestId('stub-open-doc'));
    const doc = await screen.findByTestId('wb-pane-doc');
    expect(doc.className).not.toContain('panel-max-window');
    fireEvent.click(within(doc).getByTestId('wb-pane-doc-maximize'));
    expect(screen.getByTestId('wb-pane-doc').className).toContain('panel-max-window');
    fireEvent.click(within(screen.getByTestId('wb-pane-doc')).getByTestId('wb-pane-doc-maximize'));
    expect(screen.getByTestId('wb-pane-doc').className).not.toContain('panel-max-window');
  });

  it('✓ on the selected card closes the pane with it', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderPage();
      fireEvent.click(cardWith('Hero copy'));
      fireEvent.click(within(cardWith('Hero copy')).getByTestId('wb-card-confirm'));
      await act(async () => { vi.advanceTimersByTime(300); });
      await waitFor(() => expect(pane()).toBeNull());
    } finally {
      vi.useRealTimers();
    }
  });

  it('phone: the pane is a bottom sheet over a scrim; the scrim closes it', async () => {
    await renderPage(FEED, {}, true);
    fireEvent.click(cardWith('Hero copy'));
    const sheet = pane()!;
    expect(sheet.getAttribute('data-variant')).toBe('sheet');
    expect(within(sheet).queryByTestId('wb-pane-doc-maximize')).toBeNull();
    fireEvent.click(screen.getByTestId('wb-pane-scrim'));
    expect(pane()).toBeNull();
  });
});

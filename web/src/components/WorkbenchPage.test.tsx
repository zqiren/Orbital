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
vi.mock('../config', () => ({ api: apiMock, isRelayMode: false, BASE_URL: '' }));

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
  default: (props: { selectedPath: string | null; panelHeader?: { onBack: () => void; backLabel?: string }; headerExtras?: unknown }) => (
    <div data-testid="file-preview-stub">
      <button type="button" data-testid="stub-doc-back" aria-label={props.panelHeader?.backLabel} onClick={props.panelHeader?.onBack}>back</button>
      <span>{props.selectedPath}</span>
      {props.headerExtras as never}
    </div>
  ),
}));

const uploadMock = vi.hoisted(() => vi.fn());
vi.mock('../lib/attachment-upload', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../lib/attachment-upload')>();
  return { ...actual, uploadFile: uploadMock };
});

import WorkbenchPage from './WorkbenchPage';
import { __resetWorkbenchFeed, refreshFeed } from '../hooks/useWorkbenchFeed';
import { __resetCheckedMarks } from '../hooks/useSeenReplies';
import { getWindowCovered } from '../hooks/useWindowCover';
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
  row({ session_id: 'asked', session_uuid: 'asked', name: 'Sync OSS', ask: { id: 'a1b2c3', text: 'Sync the notes too?', due: null, created: '2026-11-04', overdue: false, options: [], detail: '' }, last_reply_at: iso(5) }),
  row({ session_id: 'limit', session_uuid: 'limit', name: 'Windows smoke', agent: 'codex', stop: { cause: 'usage_limit', resets_at: null }, last_user_at: iso(125), last_reply_at: iso(120) }),
  row({ project_id: 'proj-b', session_id: 'run', session_uuid: 'run', name: 'Weekly report', status: 'running', last_reply_at: null, last_user_at: iso(1) }),
  row({ project_id: 'proj-b', session_id: 'appr', session_uuid: 'appr', name: 'Deploy', status: 'pending_approval', last_reply_at: null }),
  row({ session_id: null, session_uuid: null, name: null, last_user_at: null, last_reply_at: null, ask: { id: 'e5e5e5', text: 'Renew the domain', due: '2026-11-02', created: '2026-11-01', overdue: true, options: [], detail: '' } }),
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
  it('one list by recency, whatever the group, with one summary line and no headers', async () => {
    await renderPage();
    expect(screen.getByTestId('workbench-summary')).toHaveTextContent('5 waiting on you · 1 running');
    const order = cards().map((c) => c.getAttribute('data-dot'));
    // Newest first, whatever the dot (2026-11-05): the run started a minute
    // ago, the 5-minute-old ask, the 20-minute reply, the 2-hour stop, the
    // overdue ask from Nov 1; the approval (no time) last.
    expect(order).toEqual(['run', 'need', 'new', 'need', 'need', 'need']);
    expect(cards()[0]).toHaveTextContent('Weekly report');
    expect(cards()[1]).toHaveTextContent('Sync the notes too?');
    expect(cards()[4]).toHaveTextContent('Renew the domain');
  });

  it('a task just sent is the first card, above every 等你 card', async () => {
    await renderPage(FEED, {
      'POST /api/v2/workbench/route': () => new Promise(() => {}), // still routing
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'Brand new task' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    expect(cards()[0]).toHaveTextContent('Brand new task');
    expect(cards()[0]).toHaveTextContent('Starting…');
  });

  it('desktop: one greeting line heads the composer, the cards follow straight under it (2026-11-05)', async () => {
    await renderPage();
    const greeting = screen.getByTestId('workbench-greeting');
    expect([
      "Say it. I'll route it.", 'Drop it here, it finds its place.', 'One line. I take it from there.',
    ]).toContain(greeting.textContent);
    const kids = [...greeting.parentElement!.children].map((el) => el.getAttribute('data-testid') ?? el.className);
    expect(kids[0]).toBe('workbench-greeting');
    expect(kids.indexOf('workbench-cards')).toBe(kids.indexOf('workbench-summary') + 1);
    expect(screen.queryByTestId('workbench-mark')).toBeNull();
  });

  it('the empty page shows the same greeting above the composer', async () => {
    await renderPage([]);
    expect(screen.getByTestId('workbench-greeting').textContent).toBeTruthy();
    expect(screen.queryAllByTestId('wb-card')).toHaveLength(0);
  });

  it('phone: no greeting and no golden spacer; the composer sits under the header', async () => {
    await renderPage(FEED, {}, true);
    expect(screen.queryByTestId('workbench-greeting')).toBeNull();
    expect(screen.getByTestId('workbench-scroller').querySelector('.wb-golden')).toBeNull();
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

  it('empty: the composer alone, with the same placeholder as always (the greeting says the rest)', async () => {
    await renderPage([]);
    expect(screen.queryByTestId('workbench-summary')).toBeNull();
    expect(cards()).toHaveLength(0);
    expect(screen.getByTestId('wb-composer-input')).toHaveAttribute(
      'placeholder', 'What do you want done?',
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

  it('an ask with detail: the question is the title, the detail renders as markdown, the muted line names the session', async () => {
    const detailed = row({
      session_id: 'acct', session_uuid: 'acct', name: 'Q3 books',
      ask: { id: 'beef01', text: 'Which account first?', due: null, created: '2026-11-04', overdue: false, options: [],
        detail: 'The **Chase** statement covers `Jul–Sep`.\n\nSecond paragraph.' },
      last_reply_at: iso(2),
    });
    await renderPage([detailed, ...FEED]);
    const card = cardWith('Which account first?');
    expect(within(card).getByTestId('wb-card-title')).toHaveTextContent('“Which account first?”');
    const detail = within(card).getByTestId('wb-card-askdetail');
    expect(detail.querySelector('strong')).toHaveTextContent('Chase');
    expect(detail).toHaveTextContent('Second paragraph.');
    const meta = within(card).getByTestId('wb-card-meta');
    expect(meta).toHaveTextContent('Marketing · Q3 books');
    expect(meta.textContent).not.toContain('Chase');
  });

  it('an ask with options offers them on its card; a tap answers with that option (item 160)', async () => {
    const choice = row({
      session_id: 'venue', session_uuid: 'venue', name: 'Offsite',
      ask: { id: 'c0ffee', text: 'Which venue?', due: null, created: '2026-11-04', overdue: false, options: ['Rooftop', 'Hall'], detail: '' },
      last_reply_at: iso(2),
    });
    await renderPage([choice, ...FEED]);
    const card = cardWith('Which venue?');
    expect(within(card).getAllByTestId('wb-card-option').map((b) => b.textContent)).toEqual(['Rooftop', 'Hall']);
    expect(within(cardWith('Sync the notes too?')).queryAllByTestId('wb-card-option')).toHaveLength(0);
    fireEvent.click(within(card).getByRole('button', { name: 'Hall' }));
    await waitFor(() => expect(apiMock.mock.calls.some((c) => c[0] === '/api/v2/agents/proj-a/inject')).toBe(true));
    const body = JSON.parse(apiMock.mock.calls.find((c) => c[0] === '/api/v2/agents/proj-a/inject')![1].body);
    expect(body).toMatchObject({
      content: '[Answer to ask c0ffee, already closed: "Which venue?"]\nHall',
      session_id: 'venue', ask_id: 'c0ffee', ask_answer: 'Hall',
    });
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

  it('the agent picker ends with a row that opens Global Settings at user memory', async () => {
    await renderPage([]);
    const seen: Array<unknown> = [];
    const onOpen = (e: Event) => seen.push((e as CustomEvent).detail);
    window.addEventListener('open-global-settings', onOpen);
    fireEvent.click(screen.getByTestId('wb-chip-agent'));
    const row = within(screen.getByTestId('wb-pop-agent')).getByTestId('wb-agent-prefs');
    expect(row).toHaveTextContent('Tell Orbital who does what');
    fireEvent.click(row);
    window.removeEventListener('open-global-settings', onOpen);
    expect(seen).toEqual([{ section: 'user-memory' }]);
    expect(screen.queryByTestId('wb-pop-agent')).toBeNull();
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

  it('routing failed (2026-11-07): no session, the text is back, one error line on the composer, the picker open', async () => {
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: null, by: 'failed', reason: 'timeout', attempts: 2 }),
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: '记一个问题' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('wb-route-error')).toBeInTheDocument());
    expect(screen.getByTestId('wb-route-error')).toHaveTextContent("Couldn't pick a project. Choose one and send again.");
    await waitFor(() => expect(screen.getByTestId('wb-composer-input')).toHaveValue('记一个问题'));
    expect(screen.getByTestId('wb-pop-project')).toBeInTheDocument();
    // One error line only: not the page's open-error banner as well.
    expect(screen.queryByTestId('workbench-open-error')).toBeNull();
    expect(apiMock.mock.calls.some((c) => String(c[0]).endsWith('/new-session'))).toBe(false);
    expect(cards()).toHaveLength(0);
    // The error sits at the top of the composer box, above the text.
    const box = screen.getByTestId('wb-composer-box');
    expect(box.firstElementChild).toBe(screen.getByTestId('wb-route-error'));
  });

  it('routing failed: picking a project clears the error, and that send goes to the pick', async () => {
    let injected: Record<string, unknown> | null = null;
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: null, by: 'failed', reason: 'call_failed', attempts: 2 }),
      'POST /api/v2/agents/proj-a/new-session': () => ({ session_id: 'new-7', pinned_target: null }),
      'POST /api/v2/agents/proj-a/inject': (init) => { injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'file this' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('wb-pop-project')).toBeInTheDocument());
    fireEvent.click(within(screen.getByTestId('wb-pop-project')).getByText('Marketing'));
    expect(screen.queryByTestId('wb-route-error')).toBeNull();
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(injected).not.toBeNull());
    expect(injected).toMatchObject({ content: 'file this', session_id: 'new-7' });
  });

  it('routing failed: the next send clears the error', async () => {
    let calls = 0;
    await renderPage([], {
      'POST /api/v2/workbench/route': () => {
        calls += 1;
        return calls === 1
          ? { project_id: null, by: 'failed', reason: 'timeout', attempts: 2 }
          : new Promise(() => {}); // the second send is still routing
      },
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'again' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('wb-route-error')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId('wb-composer-input')).toHaveValue('again'));
    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    expect(screen.queryByTestId('wb-route-error')).toBeNull();
  });

  it('routing failed: staged attachments stay in the composer and nothing uploaded', async () => {
    uploadMock.mockReset();
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: null, by: 'failed', reason: 'timeout', attempts: 1 }),
    });
    fireEvent.change(screen.getByTestId('wb-attach-input'), {
      target: { files: [new File(['hi'], 'spec.txt', { type: 'text/plain' })] },
    });
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'with a file' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('wb-route-error')).toBeInTheDocument());
    expect(within(screen.getByTestId('wb-chip-strip')).getByText('spec.txt')).toBeInTheDocument();
    expect(uploadMock).not.toHaveBeenCalled();
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
describe('Workbench — attachments (stage in the composer, upload at send)', () => {
  function stage(...names: string[]) {
    const files = names.map((n) => new File(['hello'], n, { type: 'text/plain' }));
    fireEvent.change(screen.getByTestId('wb-attach-input'), { target: { files } });
  }

  beforeEach(() => {
    uploadMock.mockReset();
    uploadMock.mockImplementation(async ({ file }: { file: File }) => ({ path: `uploads/${file.name}`, size: file.size }));
  });

  it('a picked file is a chip and nothing uploads before the send', async () => {
    await renderPage([]);
    expect(screen.getByTestId('wb-attach')).toHaveAccessibleName('Attach files');
    stage('notes.txt');
    expect(within(screen.getByTestId('wb-chip-strip')).getByText('notes.txt')).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 5));
    expect(uploadMock).not.toHaveBeenCalled();
    // Attachment-only is sendable, as in the chat.
    expect(screen.getByTestId('wb-composer-send')).toBeEnabled();
  });

  it('自动: the route sees the filenames, the files upload into the routed project, the inject carries them', async () => {
    let routed: Record<string, unknown> | null = null;
    let injected: Record<string, unknown> | null = null;
    const order: string[] = [];
    uploadMock.mockImplementation(async ({ file, projectId }: { file: File; projectId: string }) => {
      order.push(`upload ${projectId}`);
      return { path: `uploads/${file.name}`, size: file.size };
    });
    await renderPage([], {
      'POST /api/v2/workbench/route': (init) => { order.push('route'); routed = JSON.parse(init!.body!); return { project_id: 'proj-b', by: 'model' }; },
      'POST /api/v2/agents/proj-b/new-session': () => { order.push('mint'); return { session_id: 'new-9', pinned_target: null }; },
      'POST /api/v2/agents/proj-b/inject': (init) => { order.push('inject'); injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    stage('a.txt', 'b.txt');
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'Summarise these' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    expect(cardWith('Summarise these')).toHaveTextContent('Starting…');
    await waitFor(() => expect(injected).not.toBeNull());
    expect(routed).toEqual({ text: 'Summarise these\n\n<attached_files>\n- a.txt\n- b.txt\n</attached_files>' });
    expect(order).toEqual(['route', 'upload proj-b', 'upload proj-b', 'mint', 'inject']);
    // The daemon writes the <attached_files> prefix from the payload, as for the chat.
    expect(injected).toMatchObject({
      content: 'Summarise these',
      session_id: 'new-9',
      attachments: [
        { path: 'uploads/a.txt', mime: 'text/plain', size: 5 },
        { path: 'uploads/b.txt', mime: 'text/plain', size: 5 },
      ],
    });
    await waitFor(() => expect(screen.queryByTestId('wb-chip-strip')).toBeNull());
  });

  it('a picked project still uploads at send, into that project', async () => {
    let injected: Record<string, unknown> | null = null;
    await renderPage([], {
      'POST /api/v2/agents/proj-a/new-session': () => ({ session_id: 'new-3', pinned_target: null }),
      'POST /api/v2/agents/proj-a/inject': (init) => { injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    fireEvent.click(screen.getByTestId('wb-chip-project'));
    fireEvent.click(within(screen.getByTestId('wb-pop-project')).getByText('Marketing'));
    stage('only.txt');
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(injected).not.toBeNull());
    expect(uploadMock.mock.calls[0][0].projectId).toBe('proj-a');
    expect(apiMock.mock.calls.some((c) => c[0] === '/api/v2/workbench/route')).toBe(false);
    // Attachment-only: empty text, the files carry the task.
    expect(injected).toMatchObject({ content: '', attachments: [{ path: 'uploads/only.txt' }] });
  });

  it('a failed upload: no session, the text comes back and the chips stay (the failed one offers retry)', async () => {
    uploadMock.mockRejectedValue(new Error('upload refused'));
    await renderPage([], {
      'POST /api/v2/workbench/route': () => ({ project_id: 'proj-b', by: 'model' }),
    });
    stage('a.txt');
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'keep me too' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(screen.getByTestId('workbench-open-error')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId('wb-composer-input')).toHaveValue('keep me too'));
    expect(apiMock.mock.calls.some((c) => String(c[0]).endsWith('/new-session'))).toBe(false);
    const strip = screen.getByTestId('wb-chip-strip');
    expect(within(strip).getByText('a.txt')).toBeInTheDocument();
    expect(within(strip).getByRole('button', { name: 'Retry upload' })).toBeInTheDocument();
    expect(cards()).toHaveLength(0);
  });
});

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
    // The back goes to the chat in the pane: just "Back" (返回), not "Back to files".
    expect(within(doc).getByTestId('stub-doc-back')).toHaveAttribute('aria-label', 'Back');
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
    // A pinned project list collapses to the strip while this covers the window.
    expect(getWindowCovered()).toBe(true);
    fireEvent.click(within(screen.getByTestId('wb-pane-doc')).getByTestId('wb-pane-doc-maximize'));
    expect(screen.getByTestId('wb-pane-doc').className).not.toContain('panel-max-window');
    expect(getWindowCovered()).toBe(false);
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

describe('Workbench — keyless (no API key: Quick Tasks is not manager_ready)', () => {
  const LAST_KEY = 'orbital.wbLastProject';
  const keyless: Project[] = [
    { project_id: 'proj-q', name: 'Quick Tasks', workspace: '/ws/q', is_scratch: true, manager_ready: false } as Project,
    { project_id: 'proj-a', name: 'Marketing', workspace: '/ws/a', manager_ready: false } as Project,
    { project_id: 'proj-b', name: 'Ops', workspace: '/ws/b', manager_ready: false } as Project,
  ];
  const keyed: Project[] = keyless.map((p) => ({ ...p, manager_ready: true }));

  async function renderWith(projects: Project[], extra: typeof routes = {}) {
    setApi({ 'GET /api/v2/workbench/feed': () => ({ rows: [], window_days: 14 }), ...extra });
    await act(async () => { await refreshFeed(); });
    return render(<WorkbenchPage setRoute={vi.fn()} projects={projects} agents={AGENTS} phone={false} />);
  }

  it('the project chip starts on Quick Tasks, never 自动, and the greeting is hidden', async () => {
    await renderWith(keyless);
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Quick Tasks');
    expect(screen.queryByTestId('workbench-greeting')).toBeNull();
    // Its slot stays (empty), so the composer does not move.
    const slot = screen.getByTestId('workbench-scroller').querySelector('.wb-greeting');
    expect(slot).not.toBeNull();
    expect(slot!.textContent).toBe('');
    expect(screen.getByTestId('workbench-scroller').querySelector('.wb-golden')).not.toBeNull();
  });

  it('the chip starts on the last project sent to from this browser; a gone one falls back to Quick Tasks', async () => {
    localStorage.setItem(LAST_KEY, 'proj-b');
    const { unmount } = await renderWith(keyless);
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Ops');
    unmount();
    localStorage.setItem(LAST_KEY, 'proj-gone');
    await renderWith(keyless);
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Quick Tasks');
  });

  it('自动 stays in the picker but is not selectable: muted, no check, a note; a click opens the model settings', async () => {
    const opened: unknown[] = [];
    const onOpen = (e: Event) => opened.push((e as CustomEvent).detail);
    window.addEventListener('open-global-settings', onOpen);
    try {
      await renderWith(keyless);
      fireEvent.click(screen.getByTestId('wb-chip-project'));
      const auto = screen.getByTestId('wb-opt-auto');
      expect(auto).toHaveAttribute('aria-disabled', 'true');
      expect(auto).toHaveAttribute('aria-selected', 'false');
      expect(auto.querySelector('.wb-ck svg')).toBeNull();
      expect(auto).toHaveTextContent('Needs an API key');
      fireEvent.click(auto);
      expect(opened).toEqual([{ section: 'llm' }]);
      expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Quick Tasks');
    } finally {
      window.removeEventListener('open-global-settings', onOpen);
    }
  });

  it('a keyless send goes to the shown project without routing, and is remembered', async () => {
    localStorage.setItem(LAST_KEY, 'proj-b');
    let injected: Record<string, unknown> | null = null;
    await renderWith(keyless, {
      'POST /api/v2/agents/proj-a/new-session': () => ({ session_id: 'k-1', pinned_target: 'codex' }),
      'POST /api/v2/agents/proj-a/inject': (init) => { injected = JSON.parse(init!.body!); return { status: 'started' }; },
    });
    fireEvent.click(screen.getByTestId('wb-chip-project'));
    fireEvent.click(within(screen.getByTestId('wb-pop-project')).getByText('Marketing'));
    fireEvent.change(screen.getByTestId('wb-composer-input'), { target: { value: 'tidy the brief' } });
    fireEvent.click(screen.getByTestId('wb-composer-send'));
    await waitFor(() => expect(injected).not.toBeNull());
    expect(apiMock.mock.calls.some((c) => c[0] === '/api/v2/workbench/route')).toBe(false);
    expect(localStorage.getItem(LAST_KEY)).toBe('proj-a');
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Marketing');
  });

  it('keyed: nothing changes — 自动 chip, selectable 自动 row with no note, greeting shown', async () => {
    localStorage.setItem(LAST_KEY, 'proj-b');
    await renderWith(keyed);
    expect(screen.getByTestId('wb-chip-project')).toHaveTextContent('Auto');
    expect(screen.getByTestId('workbench-greeting')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('wb-chip-project'));
    const auto = screen.getByTestId('wb-opt-auto');
    expect(auto).not.toHaveAttribute('aria-disabled');
    expect(auto).toHaveAttribute('aria-selected', 'true');
    expect(auto).not.toHaveTextContent('Needs an API key');
  });
});

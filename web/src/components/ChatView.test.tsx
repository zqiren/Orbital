// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Regression: /agents/available was previously fetched in ChatView's mount
// effect, blocking chat render for up to 8 s on cold-cache calls. The fetch
// now lives in App-level state and is delivered to ChatView via the
// `mentionAgents` prop. These tests assert:
//   1. ChatView mount issues NO `/agents/available` request.
//   2. The @-mention dropdown renders the agents passed via prop.
//
// Approach: mock `../config` to count calls per URL, mock the WebSocket and
// Agent hooks to no-op (they do network/event-bus work that's irrelevant
// here), mount ChatView, wait for the mount effect, then inspect the call
// log and rendered DOM.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';

// React requires this flag to allow act(...) in a test runner. Vitest's
// jsdom environment doesn't set it; we opt in here.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiCalls: string[] = [];
const apiWithTotalCalls: string[] = [];

// Spec 074 lock-race fix: the sessions pin PATCH is gated + logged so the
// serialize-behind-PATCH test can hold it open while a send fires.
let sessionsPatchGate: Promise<void> | null = null;
const sessionsPatchCalls: string[] = [];

// Bug #48 (fix C): per-request capture of the AbortSignal handed to
// apiWithTotal, plus an optional gate that holds /chat responses open so
// tests can assert in-flight behavior (cached paint, abort-on-switch).
const apiWithTotalSignals: (AbortSignal | undefined)[] = [];
let chatResponseGate: Promise<void> | null = null;

// Configurable /chat responses keyed by whether the request is the initial
// page (no offset) or a "Load earlier" page (offset present). Tests set these
// to drive ChatView's transform-once history rendering (FE-1/FE-2/FE-3).
let chatInitialResponse: { data: unknown[]; total: number } = { data: [], total: 0 };
let chatOlderResponse: { data: unknown[]; total: number } = { data: [], total: 0 };

// Configurable run-status holder. fetchHolder() and the run-status poll read
// /run-status; tests set this to drive the slot-holder comparison that gates
// live-event rendering. null = no holder.
let runStatusHolder: string | null = null;

// Configurable run-status last_terminal_event — drives the classified-error
// hydration (credential-error surfacing) after a reload.
let runStatusTerminalEvent: unknown = null;

vi.mock('../config', () => ({
  api: vi.fn(async (path: string, opts?: { method?: string }) => {
    apiCalls.push(path);
    // Session pin PATCH (spec 074) — gated for the lock-race test.
    if (opts?.method === 'PATCH' && path.includes('/sessions/')) {
      sessionsPatchCalls.push(path);
      if (sessionsPatchGate) await sessionsPatchGate;
      return {};
    }
    // run-status returns the configured slot holder so ChatView can decide
    // whether the viewed session is the holder.
    if (path.includes('/run-status')) {
      return {
        project_id: 'p1',
        status: 'running',
        current_holder_session_id: runStatusHolder,
        last_terminal_event: runStatusTerminalEvent,
      };
    }
    // pending-approval: never pending in these tests.
    if (path.includes('/pending-approval')) {
      return { pending: false };
    }
    // /chat?limit=1 (REST fallback) and anything else → empty list.
    return [];
  }),
  apiWithTotal: vi.fn(async (path: string, opts?: { signal?: AbortSignal }) => {
    apiWithTotalCalls.push(path);
    apiWithTotalSignals.push(opts?.signal);
    // /chat?limit=... is the history fetch. An `offset=` param marks a
    // "Load earlier" page; otherwise it's the initial page.
    if (path.includes('/chat?limit=')) {
      if (chatResponseGate) await chatResponseGate;
      if (path.includes('offset=')) return chatOlderResponse;
      return chatInitialResponse;
    }
    return { data: [], total: 0 };
  }),
  // Mirror the real ApiError(status, detail) shape — parseProviderError reads
  // `.detail` off instances of this class.
  ApiError: class ApiError extends Error {
    constructor(
      public status: number,
      public detail: string,
    ) {
      super(detail);
      this.name = 'ApiError';
    }
  },
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

// Minimal WS event bus so tests can dispatch live events into ChatView's
// registered handlers. ChatView calls on(type, fn)/off(type, fn).
type WsHandler = (event: unknown) => void;
const wsHandlers = new Map<string, Set<WsHandler>>();
function emitWs(type: string, event: unknown) {
  for (const fn of wsHandlers.get(type) ?? []) fn(event);
}
function resetWs() {
  wsHandlers.clear();
}

vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: (type: string, fn: WsHandler) => {
      if (!wsHandlers.has(type)) wsHandlers.set(type, new Set());
      wsHandlers.get(type)!.add(fn);
    },
    off: (type: string, fn: WsHandler) => {
      wsHandlers.get(type)?.delete(fn);
    },
    connectionState: 'connected',
    subscribe: () => {},
  }),
}));

// Per-test override slot for cancelMessage. The default resolves; tests
// that need rejection or controlled timing replace this between renders.
let cancelMessageMock: (...args: unknown[]) => Promise<unknown> = async () => undefined;
// Records cancelMessage calls (args), so "Run now" tests can assert it fired
// exactly once with the holder and injectMessage zero times.
const cancelMessageCalls: unknown[][] = [];
// Records the args of the most recent injectMessage call so tests can assert
// the viewed sessionId is threaded through. Replaceable per-test for slot_held.
let injectMessageMock: (...args: unknown[]) => Promise<unknown> = async () => undefined;
const injectCalls: unknown[][] = [];
// Records startAgent calls so the auto-start test can assert it does NOT fire
// on a session switch.
const startAgentCalls: string[] = [];
// Pending-input queue (spec 006): GET /pending recovery + cancelPendingInput.
let getPendingMock: (projectId: string) => Promise<unknown> = async () => ({
  holder: null,
  pending: [],
});
// Per-test override for coldStartScan (credential-error surfacing tests make
// it reject with a structured ApiError).
let coldStartScanMock: (...args: unknown[]) => Promise<unknown> = async () => ({
  status: 'ok',
});
// v3: cancel/dequeue is server-authoritative — it returns `removed` so recall
// knows whether it pulled back a still-queued entry (true) or the message had
// already dispatched (false). Default: removed a live entry.
let cancelPendingInputMock: (...args: unknown[]) => Promise<unknown> = async () => ({
  status: 'cancelled',
  removed: true,
});
const cancelPendingInputCalls: unknown[][] = [];

// Return a STABLE instance so useAgent's functions keep referential identity
// across renders (the real hook wraps them in useCallback). The session-load
// effect now depends on reconcilePending → getPending; a fresh object per
// render would re-run it every render and break the "exactly one /chat fetch"
// assertions.
vi.mock('../hooks/useAgent', () => {
  const inst = {
    injectMessage: vi.fn((...args: unknown[]) => {
      injectCalls.push(args);
      return injectMessageMock(...args);
    }),
    startAgent: vi.fn(async (projectId: string) => {
      startAgentCalls.push(projectId);
      return undefined;
    }),
    cancelMessage: vi.fn((...args: unknown[]) => {
      cancelMessageCalls.push(args);
      return cancelMessageMock(...args);
    }),
    newSession: vi.fn(async () => undefined),
    coldStartScan: vi.fn((...args: unknown[]) => coldStartScanMock(...args)),
    getPending: vi.fn((projectId: string) => getPendingMock(projectId)),
    cancelPendingInput: vi.fn((...args: unknown[]) => {
      cancelPendingInputCalls.push(args);
      return cancelPendingInputMock(...args);
    }),
  };
  return { useAgent: () => inst };
});

// Per-test queue state (real backend vocab: running|paused|idle). 'running' =
// active (composer disabled); 'idle'/'paused' = normal composer visible. Tests
// override queueState as needed.
let queueState: 'running' | 'paused' | 'idle' = 'idle';
const stopQueueMock = vi.fn(async () => undefined);

vi.mock('../hooks/useQueue', () => ({
  useQueue: () => ({
    snapshot: { state: queueState, version: 1, items: [], chat_session_id: null },
    loading: false,
    error: null,
    refresh: vi.fn(),
    addItem: vi.fn(),
    removeItem: vi.fn(),
    editItem: vi.fn(),
    stopQueue: stopQueueMock,
    resumeQueue: vi.fn(),
  }),
}));

// Composer attachments upload through the shared useAttachments hook. Stub the
// network call only — humanSize stays real (AttachmentChip renders with it).
let uploadFileMock: (...args: unknown[]) => Promise<{ path: string; size: number }> =
  async () => ({ path: 'uploads/notes.txt', size: 4 });
vi.mock('../lib/attachment-upload', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../lib/attachment-upload')>();
  return { ...actual, uploadFile: (...args: unknown[]) => uploadFileMock(...args) };
});

import ChatView, {
  appendLiveReasoning,
  __clearChatHistoryCacheForTests,
} from './ChatView';
import type { DisplayItem } from '../utils/chatTransform';
import type { Project } from '../types';
// Producer↔renderer parity fixture (backlog #23 D2 / #27) — the same file the
// transform tests and the pytest read. See the marker-row tests at the end.
import subAgentMarkerFixturesData from '../utils/subAgentMarkerFixtures.json';

const project: Project = {
  project_id: 'p1',
  name: 'Test',
  workspace: 'C:/tmp/p1',
  model: '',
  api_key: '',
  base_url: null,
  autonomy: 'check_in',
  instructions: '',
};

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  apiCalls.length = 0;
  apiWithTotalCalls.length = 0;
  sessionsPatchGate = null;
  sessionsPatchCalls.length = 0;
  apiWithTotalSignals.length = 0;
  chatResponseGate = null;
  __clearChatHistoryCacheForTests();
  injectCalls.length = 0;
  startAgentCalls.length = 0;
  cancelMessageCalls.length = 0;
  cancelPendingInputCalls.length = 0;
  injectMessageMock = async () => undefined;
  cancelMessageMock = async () => undefined;
  getPendingMock = async () => ({ holder: null, pending: [] });
  cancelPendingInputMock = async () => ({ status: 'cancelled', removed: true });
  coldStartScanMock = async () => ({ status: 'ok' });
  runStatusHolder = null;
  runStatusTerminalEvent = null;
  chatInitialResponse = { data: [], total: 0 };
  chatOlderResponse = { data: [], total: 0 };
  queueState = 'idle';
  stopQueueMock.mockClear();
  resetWs();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

async function flushEffects() {
  // useEffect runs after commit; awaiting a microtask + a macrotask is
  // sufficient for our synchronous mocks to resolve.
  await act(async () => {
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe('ChatView mount-effect: /agents/available is not fetched', () => {
  it('does NOT call /agents/available on mount (regression)', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();

    const agentsAvailableCalls = [...apiCalls, ...apiWithTotalCalls].filter(
      (p) => p.includes('/agents/available'),
    );
    expect(agentsAvailableCalls).toEqual([]);
  });

  it('issues exactly one /chat?limit= fetch on mount, scoped to the active session', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();

    const chatCalls = apiWithTotalCalls.filter((p) => p.includes('/chat?limit='));
    expect(chatCalls.length).toBe(1);
    expect(chatCalls[0]).toContain('/api/v2/agents/p1/chat?limit=100');
    expect(chatCalls[0]).toContain('session_id=s1');
  });
});

describe('ChatView @-mention dropdown reads from mentionAgents prop', () => {
  it('renders dropdown items from the prop when @ is typed', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[
            { slug: 'reviewer', name: 'Code Reviewer' },
            { slug: 'planner', name: 'Planner' },
          ]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();

    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea).toBeTruthy();

    // Trigger @-mention dropdown by simulating an input change with '@'.
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        'value',
      )!.set!;
      setter.call(textarea, '@');
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // The dropdown lists slugs of installed sub-agents from the prop.
    const text = container.textContent ?? '';
    expect(text).toContain('reviewer');
    expect(text).toContain('Code Reviewer');
    expect(text).toContain('planner');
    expect(text).toContain('Planner');
  });

  it('renders no dropdown items when prop is empty', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();

    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        'value',
      )!.set!;
      setter.call(textarea, '@');
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const text = container.textContent ?? '';
    expect(text).not.toContain('reviewer');
    expect(text).not.toContain('planner');
  });
});

describe('ChatView Stop button: isCancelling optimistic state', () => {
  beforeEach(() => {
    // Reset to the default happy-path resolver between tests.
    cancelMessageMock = async () => undefined;
    vi.useRealTimers();
  });

  function renderRunning() {
    return act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="running"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
  }

  it('sets aria-label="Cancelling" synchronously on Stop click and shows a spinner', async () => {
    // Make cancelMessage hang so isCancelling stays true long enough to observe.
    cancelMessageMock = () => new Promise(() => {});
    await renderRunning();
    await flushEffects();

    const stop = container.querySelector('button[aria-label="Stop"]') as HTMLButtonElement;
    expect(stop).toBeTruthy();

    await act(async () => {
      stop.click();
    });

    // The same button now has aria-label="Cancelling" and is disabled.
    const cancelling = container.querySelector('button[aria-label="Cancelling"]') as HTMLButtonElement;
    expect(cancelling).toBeTruthy();
    expect(cancelling.disabled).toBe(true);

    // Progress is shown by an arc travelling around the glyph's outline
    // (`animate-stop-arc` drives stroke-dashoffset). It is deliberately NOT
    // `animate-spin`: the ring is a rounded square now, and rotating that
    // wobbles the silhouette — the exact jump StopGlyph exists to avoid.
    expect(cancelling.querySelector('.animate-stop-arc')).toBeTruthy();
    expect(cancelling.querySelector('.animate-spin')).toBeNull();

    // Textarea is disabled while cancelling.
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.disabled).toBe(true);
  });

  it('clears isCancelling when agentStatus transitions to idle (WS-driven)', async () => {
    cancelMessageMock = () => new Promise(() => {});
    await renderRunning();
    await flushEffects();

    const stop = container.querySelector('button[aria-label="Stop"]') as HTMLButtonElement;
    await act(async () => {
      stop.click();
    });
    expect(container.querySelector('button[aria-label="Cancelling"]')).toBeTruthy();

    // Simulate the agent.status:idle WS broadcast by re-rendering with idle.
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();

    // The Stop button area is gone (running/waiting branch no longer renders);
    // the textarea is back to enabled and no spinner is visible.
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.disabled).toBe(false);
    expect(container.querySelector('button[aria-label="Cancelling"]')).toBeNull();
  });

  it('clears isCancelling after a 10s timeout and shows a retry notice', async () => {
    vi.useFakeTimers();
    cancelMessageMock = () => new Promise(() => {});
    await renderRunning();
    // Effects ran with real timers up to render time; let pending effects flush
    // before swapping in fake timers' clock for the 10s wait.
    await act(async () => {
      await Promise.resolve();
    });

    const stop = container.querySelector('button[aria-label="Stop"]') as HTMLButtonElement;
    await act(async () => {
      stop.click();
    });
    expect(container.querySelector('button[aria-label="Cancelling"]')).toBeTruthy();

    // Advance fake timers past the 10s fallback.
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });

    // Optimistic state cleared; notice visible.
    expect(container.querySelector('button[aria-label="Cancelling"]')).toBeNull();
    expect(container.querySelector('button[aria-label="Stop"]')).toBeTruthy();
    const notice = container.querySelector('[role="status"]');
    expect(notice?.textContent ?? '').toContain('Cancel took longer than expected');
    vi.useRealTimers();
  });

  it('clears isCancelling immediately on POST failure and surfaces an error notice', async () => {
    // Reject the cancel POST.
    cancelMessageMock = () => Promise.reject(new Error('network down'));
    await renderRunning();
    await flushEffects();

    const stop = container.querySelector('button[aria-label="Stop"]') as HTMLButtonElement;
    await act(async () => {
      stop.click();
      // Let the rejection settle.
      await Promise.resolve();
      await Promise.resolve();
    });

    // Optimistic state is dropped; failure notice is visible.
    expect(container.querySelector('button[aria-label="Cancelling"]')).toBeNull();
    expect(container.querySelector('button[aria-label="Stop"]')).toBeTruthy();
    const notice = container.querySelector('[role="status"]');
    expect(notice?.textContent ?? '').toContain('Cancel request failed');
  });
});

// ─── T5: session-aware ChatView ────────────────────────────────────────────

/** Set a textarea's value through the native setter so React's onChange fires. */
function typeInComposer(text: string) {
  const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
  const setter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype,
    'value',
  )!.set!;
  setter.call(textarea, text);
  textarea.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Dispatch a bubbling native keydown on the composer so React's delegated
 *  onKeyDown (handleKeyDown) runs — used for the ↑-to-recall tests. */
function pressKey(key: string) {
  const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
  textarea.dispatchEvent(
    new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }),
  );
}

function renderChat(props: {
  agentStatus?: string;
  sessionId?: string;
  initialDraft?: string;
  onDraftConsumed?: () => void;
  mentionAgents?: Array<{ slug: string; name: string }>;
}) {
  return act(async () => {
    root.render(
      <ChatView
        projectId="p1"
        project={project}
        agentStatus={(props.agentStatus ?? 'idle') as never}
        mentionAgents={props.mentionAgents ?? []}
        sessionId={props.sessionId}
        initialDraft={props.initialDraft}
        onDraftConsumed={props.onDraftConsumed}
      />,
    );
  });
}

describe('T5 ChatView: per-session history load', () => {
  it('loads /chat scoped to the active session and reloads when sessionId changes', async () => {
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    let chatCalls = apiWithTotalCalls.filter((p) => p.includes('/chat?limit='));
    expect(chatCalls.length).toBe(1);
    expect(chatCalls[0]).toContain('session_id=s1');

    // Switch to a different session — history reloads with the new id.
    await renderChat({ sessionId: 's2' });
    await flushEffects();

    chatCalls = apiWithTotalCalls.filter((p) => p.includes('/chat?limit='));
    expect(chatCalls.length).toBe(2);
    expect(chatCalls[1]).toContain('session_id=s2');
  });

  it('does NOT fetch /chat history when no session is resolved (sessionId undefined)', async () => {
    await renderChat({ sessionId: undefined });
    await flushEffects();

    const chatCalls = apiWithTotalCalls.filter((p) => p.includes('/chat?limit='));
    expect(chatCalls.length).toBe(0);
    // Empty state is shown, not a perpetual loading skeleton.
    expect(container.textContent ?? '').toContain('No messages yet');
  });
});

describe('T5 ChatView: per-session composer draft', () => {
  it('preserves the draft per session across a switch (type in A, switch to B, back to A)', async () => {
    await renderChat({ sessionId: 'A' });
    await flushEffects();

    // Type a draft into session A.
    await act(async () => {
      typeInComposer('hello from A');
    });
    let textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('hello from A');

    // Switch to session B — composer is empty (no draft for B yet).
    await renderChat({ sessionId: 'B' });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('');

    // Type a draft into B.
    await act(async () => {
      typeInComposer('hello from B');
    });
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('hello from B');

    // Switch back to A — A's draft is restored.
    await renderChat({ sessionId: 'A' });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('hello from A');

    // And back to B — B's draft is restored.
    await renderChat({ sessionId: 'B' });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('hello from B');
  });
});

describe('ChatView: one-shot composer prefill (route.draft / initialDraft, Workbench doorway spec 2026-07-24)', () => {
  it('seeds the composer with initialDraft when sessionId is already resolved, focused with the cursor at the end', async () => {
    const onDraftConsumed = vi.fn();
    await renderChat({ sessionId: 's1', initialDraft: 'Workbench · "do the thing"\n\n', onDraftConsumed });
    await flushEffects();

    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('Workbench · "do the thing"\n\n');
    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe(textarea.value.length);
    expect(textarea.selectionEnd).toBe(textarea.value.length);
    expect(onDraftConsumed).toHaveBeenCalledTimes(1);
  });

  it('survives the sessionId undefined→resolved transition (ChatTab async default-session resolution) without being blanked', async () => {
    // Mirrors the real navigation: WorkbenchPage sets route WITHOUT a
    // sessionId, so ChatView first mounts with sessionId=undefined; ChatTab
    // resolves a default session a tick later and the sessionId prop flips
    // to defined. The per-session draft map has no entry for that session
    // (never visited before) — naively that would blank the composer.
    const draft = 'Workbench · "do the thing"\n\n';
    await renderChat({ sessionId: undefined, initialDraft: draft });
    await flushEffects();
    let textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe(draft);

    // Session resolves (same initialDraft prop — ChatTab doesn't change it).
    await renderChat({ sessionId: 's1', initialDraft: draft });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe(draft);
  });

  it('does NOT discard a user edit made while sessionId is still resolving (critical fix: async-resolution overwrite)', async () => {
    // Same setup as the transition test above, but the user types something
    // ELSE into the composer during the window before ChatTab's async
    // default-session resolution completes. The session-swap effect must not
    // stomp that edit with the original (now stale) seeded draft when
    // sessionId finally resolves.
    const draft = 'Workbench · "do the thing"\n\n';
    await renderChat({ sessionId: undefined, initialDraft: draft });
    await flushEffects();

    await act(async () => {
      typeInComposer('actually, let me write something completely different');
    });
    let textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('actually, let me write something completely different');

    // Session resolves — same initialDraft prop, exactly as ChatTab would
    // pass (it doesn't re-derive or change it on this transition).
    await renderChat({ sessionId: 's1', initialDraft: draft });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('actually, let me write something completely different');
  });

  it('does NOT re-apply the draft on a later re-render with the same initialDraft (applies once, preserves user edits)', async () => {
    const draft = 'Workbench · "do the thing"\n\n';
    await renderChat({ sessionId: 's1', initialDraft: draft });
    await flushEffects();

    await act(async () => {
      typeInComposer('actually let me rewrite this');
    });
    let textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('actually let me rewrite this');

    // Parent re-renders with the SAME initialDraft prop (e.g. unrelated
    // App-level state change before route.draft is cleared) — must not stomp
    // the user's edit.
    await renderChat({ sessionId: 's1', initialDraft: draft });
    await flushEffects();
    textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('actually let me rewrite this');
  });

  it('does not auto-send or spawn a session when a draft is applied', async () => {
    await renderChat({ sessionId: undefined, initialDraft: 'Workbench · "do the thing"\n\n' });
    await flushEffects();
    await renderChat({ sessionId: 's1', initialDraft: 'Workbench · "do the thing"\n\n' });
    await flushEffects();

    expect(injectCalls.length).toBe(0);
    expect(startAgentCalls.length).toBe(0);
  });

  it('when no initialDraft is provided, behaves exactly as before (empty composer, no callback)', async () => {
    const onDraftConsumed = vi.fn();
    await renderChat({ sessionId: 's1', onDraftConsumed });
    await flushEffects();

    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('');
    expect(onDraftConsumed).not.toHaveBeenCalled();
  });
});

describe('ChatView: strict session_id event routing (seam 3 / Phase 3)', () => {
  // Live events are routed STRICTLY by session_id. A handler renders an event
  // iff event.session_id === the viewed sessionId. There is no viewingHolder
  // heuristic and no default-to-show path: an event for another session, or
  // with no session_id, is dropped. runStatusHolder is irrelevant to routing
  // now — these tests prove routing follows session_id, not the holder.

  it('renders a stream delta whose session_id matches the viewed session', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'streamed-token-XYZ',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').toContain('streamed-token-XYZ');
  });

  it('renders a matching-session delta even when the holder is unresolved (null)', async () => {
    // Routing is by session_id, not holder: a delta stamped for the viewed
    // session renders regardless of whether fetchHolder has resolved yet.
    // (Replaces the old "lenient null-holder window" behavior, which has been
    // removed because it defaulted to SHOW and leaked across sessions.)
    runStatusHolder = null;
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'window-token-ABC',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').toContain('window-token-ABC');
  });

  it('does NOT render a stream delta with no session_id (no default-to-show)', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        // session_id deliberately omitted — must be dropped, never shown.
        text: 'nosession-token-NOPE',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').not.toContain('nosession-token-NOPE');
  });

  it('does NOT render a stream delta stamped for a DIFFERENT session (even the holder)', async () => {
    // The holder is s2 but we view s1. s2's delta must not leak into s1's pane.
    runStatusHolder = 's2';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's2',
        text: 'other-session-token-XYZ',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').not.toContain('other-session-token-XYZ');
  });

  it('does NOT render a live activity capsule for a DIFFERENT session', async () => {
    runStatusHolder = 's2';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
        session_id: 's2',
        category: 'file_read',
        tool_name: 'Read',
        description: 'reading secret-file.txt',
        id: 'act-1',
        timestamp: new Date().toISOString(),
      });
    });

    expect(container.querySelector('[data-testid="agent_run"]')).toBeNull();
    expect(container.textContent ?? '').not.toContain('secret-file.txt');
  });

  it('renders a live activity capsule for the viewed session', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
        session_id: 's1',
        category: 'file_read',
        tool_name: 'Read',
        description: 'reading visible-file.txt',
        id: 'act-2',
        timestamp: new Date().toISOString(),
      });
    });

    expect(container.querySelector('[data-testid="agent_run"]')).toBeTruthy();
    expect(container.textContent ?? '').toContain('visible-file.txt');
  });

  it('does NOT render a live capsule row for the fanout tool call', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
        session_id: 's1',
        category: 'tool_use',
        tool_name: 'fanout',
        description: 'fanout: 3 tasks',
        id: 'act-fanout-1',
        timestamp: new Date().toISOString(),
      });
    });

    expect(container.textContent ?? '').not.toContain('fanout: 3 tasks');
  });

  it('MULTI-SESSION LEAK TEST: no event type leaks from session s2 into s1 — and s1 events still render', async () => {
    // Covers stream_delta PLUS the two formerly-always-on handlers (approval
    // request + agent.notify). None may default to "show": an event for s2
    // must never appear while viewing s1, and s1's own events must render.
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    // --- s2 events: must all be dropped ---
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta', project_id: 'p1', session_id: 's2',
        text: 'LEAK-DELTA-s2', source: 'assistant', is_final: false,
      });
      emitWs('approval.request', {
        type: 'approval.request', project_id: 'p1', session_id: 's2',
        what: 'LEAK-APPROVAL-s2', tool_name: 'shell', tool_call_id: 'tc-s2',
        tool_args: {}, recent_activity: [],
      });
      emitWs('agent.notify', {
        type: 'agent.notify', project_id: 'p1', session_id: 's2',
        title: 'LEAK-NOTIFY-s2', body: 'should not show', urgency: 'normal',
        timestamp: new Date().toISOString(),
      });
    });

    const afterS2 = container.textContent ?? '';
    expect(afterS2).not.toContain('LEAK-DELTA-s2');
    expect(afterS2).not.toContain('LEAK-APPROVAL-s2');
    expect(afterS2).not.toContain('LEAK-NOTIFY-s2');

    // --- s1 events: must all render (proves routing isn't just dropping everything) ---
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta', project_id: 'p1', session_id: 's1',
        text: 'SHOW-DELTA-s1', source: 'assistant', is_final: false,
      });
      emitWs('approval.request', {
        type: 'approval.request', project_id: 'p1', session_id: 's1',
        // shell would render through the localized activity template
        // ("Ran: {command}") and swallow the marker — use a command arg so
        // the marker survives the derived sentence.
        what: 'SHOW-APPROVAL-s1', tool_name: 'shell', tool_call_id: 'tc-s1',
        tool_args: { command: 'SHOW-APPROVAL-s1' }, recent_activity: [],
      });
      emitWs('agent.notify', {
        type: 'agent.notify', project_id: 'p1', session_id: 's1',
        title: 'SHOW-NOTIFY-s1', body: 'visible', urgency: 'normal',
        timestamp: new Date().toISOString(),
      });
    });

    const afterS1 = container.textContent ?? '';
    expect(afterS1).toContain('SHOW-DELTA-s1');
    expect(afterS1).toContain('SHOW-APPROVAL-s1');
    expect(afterS1).toContain('SHOW-NOTIFY-s1');
    // And the s2 events are STILL absent after rendering s1's.
    expect(afterS1).not.toContain('LEAK-DELTA-s2');
    expect(afterS1).not.toContain('LEAK-APPROVAL-s2');
    expect(afterS1).not.toContain('LEAK-NOTIFY-s2');
  });
});

describe('ChatView: fanout worker messages do not leak into main chat (spec 009 §0.5-8)', () => {
  // Fanout workers are ephemeral task executors (worker:<fanout_id>-<index>
  // handles), not persistent collaborators — their live turn output is
  // surfaced only via the fanout progress card / join summary, never as a
  // main-chat bubble. This mirrors SubAgentStatusBar's own `worker:` filter
  // (chips bar), which already excludes them.

  it('does NOT render a chat.sub_agent_message whose source is a worker: handle', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.sub_agent_message', {
        type: 'chat.sub_agent_message',
        project_id: 'p1',
        session_id: 's1',
        content: 'WORKER-LEAK-TEXT should never render',
        source: 'worker:abcd1234-0',
        timestamp: new Date().toISOString(),
      });
    });

    expect(container.textContent ?? '').not.toContain('WORKER-LEAK-TEXT');
  });

  it('still renders a chat.sub_agent_message from a non-worker (persistent) sub-agent handle', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.sub_agent_message', {
        type: 'chat.sub_agent_message',
        project_id: 'p1',
        session_id: 's1',
        content: 'CLAUDE-CODE-REPLY visible text',
        source: 'claude-code',
        timestamp: new Date().toISOString(),
      });
    });

    expect(container.textContent ?? '').toContain('CLAUDE-CODE-REPLY visible text');
  });
});

// The chat composer's attachment machinery lives in the shared
// useAttachments hook (spec 053). These guard the ChatView side of that
// extraction: the chips still render, send is still gated on in-flight
// uploads, and the uploaded path still ships inline as an <attached_files>
// block — chat's wire format, which the queue path deliberately does NOT use.
describe('ChatView composer attachments', () => {
  /** Pick a file through the hidden composer file input. */
  function pickFile(name = 'notes.txt') {
    const input = container.querySelector(
      '[data-testid="attachment-file-input"]',
    ) as HTMLInputElement;
    const file = new File(['abcd'], name, { type: 'text/plain' });
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  it('renders a chip and sends the uploaded path inline as an attachments block', async () => {
    uploadFileMock = async () => ({ path: 'uploads/notes.txt', size: 4 });
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      pickFile();
    });
    await flushEffects();

    expect(container.querySelector('[data-testid="chip-strip"]')).toBeTruthy();
    expect(container.querySelector('[data-testid="chip-check"]')).toBeTruthy();

    await act(async () => {
      typeInComposer('look at this');
    });
    const send = container.querySelector('button[aria-label="Send"]') as HTMLButtonElement;
    await act(async () => {
      send.click();
      await Promise.resolve();
    });

    expect(injectCalls.length).toBe(1);
    // injectMessage(projectId, content, target, nonce, attachments, sessionId)
    expect(injectCalls[0][4]).toEqual([
      { path: 'uploads/notes.txt', mime: 'text/plain', size: 4 },
    ]);
    // Composer chips clear on send, and the optimistic bubble renders the
    // attachment — which only happens if the client-built block went inline.
    expect(container.querySelector('[data-testid="chip-strip"]')).toBeNull();
    expect(container.textContent ?? '').toContain('notes.txt');
  });

  it('disables Send while an upload is in flight', async () => {
    uploadFileMock = () => new Promise(() => {}); // never settles
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      typeInComposer('waiting on the upload');
    });
    await act(async () => {
      pickFile();
    });
    await flushEffects();

    expect(container.querySelector('[data-testid="chip-spinner"]')).toBeTruthy();
    const send = container.querySelector('button[aria-label="Send"]') as HTMLButtonElement;
    expect(send.disabled).toBe(true);
    await act(async () => {
      send.click();
      await Promise.resolve();
    });
    expect(injectCalls.length).toBe(0);
  });
});

describe('T5 ChatView: inject targets the viewed session', () => {
  it('passes the viewed sessionId as the 6th injectMessage argument', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'idle', sessionId: 's7' });
    await flushEffects();

    await act(async () => {
      typeInComposer('do the thing');
    });
    // Click the Send button (idle branch).
    const send = container.querySelector('button[aria-label="Send"]') as HTMLButtonElement;
    await act(async () => {
      send.click();
      await Promise.resolve();
    });

    expect(injectCalls.length).toBe(1);
    // injectMessage(projectId, content, target, nonce, attachments, sessionId)
    const args = injectCalls[0];
    expect(args[0]).toBe('p1');
    expect(args[1]).toBe('do the thing');
    expect(args[5]).toBe('s7');
  });

  // Spec 006 §6 test 13 (rewrite of the old slot_held test). The 202 happy
  // path is now `queued_pending_slot`: the message was ACCEPTED + queued. The
  // optimistic bubble is KEPT (not removed) and a "Waiting for {holder}…"
  // affordance shows.
  it('keeps the optimistic bubble and shows the waiting affordance on 202 queued_pending_slot', async () => {
    injectMessageMock = async () => ({
      status: 'queued_pending_slot',
      holding_session_id: 'sess-A',
      queued_session_id: 's1',
      nonce: 'ignored-server-nonce',
      position: 0,
    });
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      typeInComposer('queued message');
    });
    const send = container.querySelector('button[aria-label="Send"]') as HTMLButtonElement;
    await act(async () => {
      send.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    // The optimistic bubble is KEPT.
    expect(container.textContent ?? '').toContain('queued message');
    // The new waiting affordance is shown. v3 copy is placeholder-free (no
    // holder id) and a cross-session entry offers [Run now].
    const notice = container.querySelector('[data-testid="pending-input-notice"]');
    expect(notice).toBeTruthy();
    const body = container.querySelector('[data-testid="pending-input-notice-body"]');
    expect(body?.textContent ?? '').toContain('Waiting for other sessions to finish.');
    expect(container.querySelector('[data-testid="pending-input-notice-run-now"]')).toBeTruthy();
    // No Stop-waiting button in v3.
    expect(container.querySelector('[data-testid="pending-input-notice-stop"]')).toBeNull();
    // The OLD slot_held notice is NOT used on the happy path.
    expect(container.querySelector('[data-testid="slot-held-notice"]')).toBeNull();
  });

  it('falls back to SlotHeldNotice when inject returns 202 slot_held (enqueue failure)', async () => {
    // slot_held is retained ONLY as the defensive enqueue-failure fallback.
    injectMessageMock = async () => ({
      status: 'slot_held',
      holding_session_id: 'other-session',
    });
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      typeInComposer('queued message');
    });
    const send = container.querySelector('button[aria-label="Send"]') as HTMLButtonElement;
    await act(async () => {
      send.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    const notice = container.querySelector('[data-testid="slot-held-notice"]');
    expect(notice).toBeTruthy();
    const holder = container.querySelector('[data-testid="slot-held-notice-holder"]');
    expect(holder?.textContent ?? '').toContain('other-session');
    // The fallback uses the OLD notice, NOT the new pending-input affordance.
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });
});

// ─── Spec 006: pending-input queue (frontend §3h, tests 13-19) ─────────────

// Spec 074 lock-race (manual-verification bug, 2026-08-31): picking a pin
// target fires a fire-and-forget PATCH that write-locks the session JSONL
// server-side; a send landing inside that burst was rejected with the
// message dropped. handleSend must serialize behind the in-flight PATCH.
describe('Spec 074: send serializes behind the in-flight pin PATCH', () => {
  it('holds the inject until the pin PATCH settles, then dispatches pinned', async () => {
    let releasePatch!: () => void;
    sessionsPatchGate = new Promise<void>((r) => { releasePatch = r; });

    await renderChat({
      agentStatus: 'idle',
      sessionId: 's1',
      mentionAgents: [{ slug: 'claude-code', name: 'Claude Code' }],
    });
    await flushEffects();

    // Open the pin dropdown and pick the worker — fires the (gated) PATCH.
    const pinToggle = container.querySelector(
      '[data-testid="pin-target-select"] > button',
    ) as HTMLButtonElement;
    await act(async () => { pinToggle.click(); });
    const option = Array.from(
      container.querySelectorAll('[role="option"]'),
    ).find((el) => el.textContent?.includes('Claude Code')) as HTMLButtonElement;
    expect(option).toBeTruthy();
    await act(async () => { option.click(); });
    expect(sessionsPatchCalls.length).toBe(1);

    await act(async () => { typeInComposer('hi worker'); });
    const send = container.querySelector(
      'button[aria-label="Send"]',
    ) as HTMLButtonElement;
    await act(async () => {
      send.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    // PATCH still in flight → the inject must NOT have fired yet.
    expect(injectCalls.length).toBe(0);

    await act(async () => {
      releasePatch();
      await Promise.resolve();
    });
    await flushEffects();

    // PATCH settled → the send goes out, pinned to the picked worker.
    expect(injectCalls.length).toBe(1);
    // injectMessage(projectId, content, target, nonce, attachments, sessionId, pinned)
    expect(injectCalls[0][1]).toBe('hi worker');
    expect(injectCalls[0][2]).toBe('claude-code');
    expect(injectCalls[0][6]).toBe(true);
  });
});

describe('ChatView: pending-input queue (spec 006)', () => {
  // Helper: send a message that gets queued (202 queued_pending_slot) into the
  // viewed session, returning after the optimistic bubble + affordance render.
  async function sendQueued(sessionId: string, holder: string, text: string) {
    injectMessageMock = async () => ({
      status: 'queued_pending_slot',
      holding_session_id: holder,
      queued_session_id: sessionId,
      position: 0,
    });
    await renderChat({ agentStatus: 'running', sessionId });
    await flushEffects();
    await act(async () => {
      typeInComposer(text);
    });
    // agentStatus 'running' renders the "Queue" button (not "Send").
    const btn =
      (container.querySelector('button[aria-label="Send"]') as HTMLButtonElement | null) ??
      ([...container.querySelectorAll('button')].find(
        (b) => (b.textContent ?? '').trim() === 'Queue',
      ) as HTMLButtonElement);
    await act(async () => {
      btn.click();
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  it('test 14: chat.pending_dispatched clears the waiting affordance for the viewed session', async () => {
    await sendQueued('s1', 'sess-A', 'hello B');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();

    // The dispatch echo for our own nonce: we don't know the internally-minted
    // nonce, so simulate the real backend ordering — chat.pending_dispatched is
    // session-scoped and is the SOLE clear trigger. Emit with the matching
    // session; the handler clears by nonce, so emit using the nonce that the
    // origin tab tracked. We can't read it, so assert the cross-tab analogue:
    // emit pending_enqueued from ANOTHER tab/nonce, then dispatch THAT nonce.
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-dispatch', content: 'second queued', position: 1,
      });
    });
    // Two notices now (origin + cross-tab nonce).
    expect(container.querySelectorAll('[data-testid="pending-input-notice"]').length).toBe(2);

    await act(async () => {
      emitWs('chat.pending_dispatched', {
        type: 'chat.pending_dispatched', project_id: 'p1', session_id: 's1', nonce: 'n-dispatch',
      });
    });
    // The dispatched nonce's affordance is cleared; the other remains.
    expect(container.querySelectorAll('[data-testid="pending-input-notice"]').length).toBe(1);
  });

  it('test 14b: chat.pending_dispatched for a non-viewed session does not error', async () => {
    await sendQueued('s1', 'sess-A', 'hello B');
    await act(async () => {
      emitWs('chat.pending_dispatched', {
        type: 'chat.pending_dispatched', project_id: 'p1', session_id: 's-other', nonce: 'whatever',
      });
    });
    // Our viewed-session affordance is untouched (different nonce/session).
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();
  });

  it('test 15: switch away during pending then back restores the affordance from GET /pending', async () => {
    await sendQueued('s1', 'sess-A', 'hello B');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();

    // Backend now reports the pending entry via GET /pending (persist-at-dispatch
    // means it is NOT in B's JSONL, so only /pending recovers it).
    getPendingMock = async () => ({
      holder: 'sess-A',
      pending: [
        { session_id: 's1', nonce: 'n-recovered', content: 'hello B', position: 0, kind: 'cross' },
      ],
    });

    // Switch away to s2 …
    await renderChat({ agentStatus: 'running', sessionId: 's2' });
    await flushEffects();
    // s2 is a different session: no affordance for it.
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();

    // … and back to s1 — the affordance + bubble are restored from GET /pending.
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();
    expect(container.textContent ?? '').toContain('hello B');
  });

  it('test 16: long wait (>30s nonce eviction) then dispatch echo → no duplicate bubble', async () => {
    vi.useFakeTimers();
    try {
      // A cross-tab enqueue renders an optimistic bubble + registers the nonce.
      await renderChat({ agentStatus: 'running', sessionId: 's1' });
      await act(async () => { await Promise.resolve(); });

      await act(async () => {
        emitWs('chat.pending_enqueued', {
          type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
          holder: 'sess-A', nonce: 'n-long', content: 'long-wait-msg', position: 0,
        });
      });
      const countBubbles = () =>
        (container.textContent ?? '').split('long-wait-msg').length - 1;
      expect(countBubbles()).toBe(1);

      // Advance > 30s and trigger the eviction sweep via an unrelated user echo.
      await act(async () => {
        vi.advanceTimersByTime(40_000);
        emitWs('chat.user_message', {
          type: 'chat.user_message', project_id: 'p1', session_id: 's1',
          content: 'unrelated', nonce: 'n-unrelated', timestamp: new Date().toISOString(),
        });
      });

      // Now the dispatch echo arrives for our pending nonce — it must DEDUP
      // (the nonce was exempt from eviction), so no second bubble.
      await act(async () => {
        emitWs('chat.user_message', {
          type: 'chat.user_message', project_id: 'p1', session_id: 's1',
          content: 'long-wait-msg', nonce: 'n-long', timestamp: new Date().toISOString(),
        });
      });
      expect(countBubbles()).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('test 17: Run now calls cancelMessage(holder) exactly once and injectMessage zero times', async () => {
    await sendQueued('s1', 'sess-A', 'hello B');
    // sendQueued already called injectMessage once (the send). Reset the counter
    // so we measure ONLY the Run-now path.
    injectCalls.length = 0;
    cancelMessageCalls.length = 0;

    const runNow = container.querySelector(
      '[data-testid="pending-input-notice-run-now"]',
    ) as HTMLButtonElement;
    await act(async () => {
      runNow.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(cancelMessageCalls.length).toBe(1);
    expect(cancelMessageCalls[0][0]).toBe('p1');
    expect(cancelMessageCalls[0][1]).toBe('sess-A');
    // No re-inject (the queued message dispatches itself).
    expect(injectCalls.length).toBe(0);
  });

  it('test 18: multi-tab — chat.pending_enqueued dedups by nonce (no double bubble)', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    const evt = {
      type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
      holder: 'sess-A', nonce: 'n-dup', content: 'multi-tab-msg', position: 0,
    };
    await act(async () => { emitWs('chat.pending_enqueued', evt); });
    await act(async () => { emitWs('chat.pending_enqueued', evt); }); // relay retry

    const count = (container.textContent ?? '').split('multi-tab-msg').length - 1;
    expect(count).toBe(1);
    expect(container.querySelectorAll('[data-testid="pending-input-notice"]').length).toBe(1);
  });

  it('test 19: Run now vs self-free race — stale cancel(holder) still only calls cancelMessage once, no inject', async () => {
    // The slot may have already freed (holder gone) by the time the user clicks
    // Run now. The frontend still just calls cancelMessage(holder) once (a no-op
    // server-side) and never re-injects — backend guarantees B runs once.
    await sendQueued('s1', 'sess-A', 'hello B');
    injectCalls.length = 0;
    cancelMessageCalls.length = 0;
    // Simulate the holder already being free: cancelMessage resolves as a no-op.
    cancelMessageMock = async () => ({ status: 'idle' });

    const runNow = container.querySelector(
      '[data-testid="pending-input-notice-run-now"]',
    ) as HTMLButtonElement;
    await act(async () => {
      runNow.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(cancelMessageCalls.length).toBe(1);
    expect(injectCalls.length).toBe(0);
  });

  it('v3: the notice has NO Stop-waiting button', async () => {
    await sendQueued('s1', 'sess-A', 'hello B');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();
    expect(container.querySelector('[data-testid="pending-input-notice-stop"]')).toBeNull();
  });

  // ─── v3 §12 R1: mobile tap-to-edit ────────────────────────────────────────

  it('tap-to-edit recalls the queued message (cancelPendingInput, loads text, removes bubble)', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-tap', content: 'tap-me', position: 0,
      });
    });
    expect(container.textContent ?? '').toContain('tap-me');

    const edit = container.querySelector(
      '[data-testid="pending-input-notice-edit"]',
    ) as HTMLButtonElement;
    await act(async () => {
      edit.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(cancelPendingInputCalls.length).toBe(1);
    expect(cancelPendingInputCalls[0]).toEqual(['p1', 's1', 'n-tap']);
    // removed:true (default) → text loaded into the composer, bubble + notice gone.
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('tap-me');
    // The bubble was removed: the empty-state placeholder is back.
    expect(container.textContent ?? '').toContain('No messages yet');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });

  // ─── v3 §11c: ↑-to-recall (desktop accelerator) ──────────────────────────

  it('↑ in an empty composer recalls the newest cross-session queued message', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-up', content: 'up-recall', position: 0,
      });
    });
    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(cancelPendingInputCalls.length).toBe(1);
    expect(cancelPendingInputCalls[0][2]).toBe('n-up');
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('up-recall');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });

  it('↑ recalls a same-session queued message (kind=same)', async () => {
    injectMessageMock = async () => ({ status: 'queued_same_session' });
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      typeInComposer('recall same');
    });
    const btn =
      (container.querySelector('button[aria-label="Send"]') as HTMLButtonElement | null) ??
      ([...container.querySelectorAll('button')].find(
        (b) => (b.textContent ?? '').trim() === 'Queue',
      ) as HTMLButtonElement);
    await act(async () => {
      btn.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();

    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(cancelPendingInputCalls.length).toBe(1);
    expect(cancelPendingInputCalls[0][1]).toBe('s1'); // sessionId
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe('recall same');
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });

  it('↑ recalls the NEWEST queued message when several are pending', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date('2026-06-30T00:00:00.000Z'));
      await renderChat({ agentStatus: 'running', sessionId: 's1' });
      await act(async () => { await Promise.resolve(); });
      await act(async () => {
        emitWs('chat.pending_enqueued', {
          type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
          holder: 'sess-A', nonce: 'n-older', content: 'older-msg', position: 0,
        });
      });
      vi.setSystemTime(new Date('2026-06-30T00:00:05.000Z'));
      await act(async () => {
        emitWs('chat.pending_enqueued', {
          type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
          holder: 'sess-A', nonce: 'n-newer', content: 'newer-msg', position: 1,
        });
      });
      await act(async () => {
        pressKey('ArrowUp');
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(cancelPendingInputCalls.length).toBe(1);
      expect(cancelPendingInputCalls[0][2]).toBe('n-newer');
    } finally {
      vi.useRealTimers();
    }
  });

  it('↑ does nothing when the composer is non-empty (caret still moves)', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-ne', content: 'pending-msg', position: 0,
      });
    });
    await act(async () => { typeInComposer('half-typed'); });
    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
    });
    expect(cancelPendingInputCalls.length).toBe(0);
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();
  });

  it('↑ does nothing when a mention dropdown is open (guard precedes recall)', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-dd', content: 'pending-dd', position: 0,
      });
    });
    // '@' opens the mention dropdown; its ArrowUp guard runs before the recall.
    await act(async () => { typeInComposer('@'); });
    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
    });
    expect(cancelPendingInputCalls.length).toBe(0);
  });

  it('↑ / tap is a no-op when the queued entry has attachments (no chip half-restore)', async () => {
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    const withAttachments =
      '<attached_files>\n- /tmp/a.png (image/png, 1.2 KB)\n</attached_files>\n\nplease look';
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-att', content: withAttachments, position: 0,
      });
    });
    // The edit affordance is disabled for attachment-bearing entries.
    const edit = container.querySelector(
      '[data-testid="pending-input-notice-edit"]',
    ) as HTMLButtonElement;
    expect(edit.disabled).toBe(true);

    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
    });
    expect(cancelPendingInputCalls.length).toBe(0);
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeTruthy();
  });

  // ─── v3 §12 R2: server-authoritative recall (no double-send) ──────────────

  it('recall on removed:false does NOT load text and leaves no duplicate bubble', async () => {
    cancelPendingInputMock = async () => ({ status: 'cancelled', removed: false });
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('chat.pending_enqueued', {
        type: 'chat.pending_enqueued', project_id: 'p1', session_id: 's1',
        holder: 'sess-A', nonce: 'n-race', content: 'already-running', position: 0,
      });
    });
    expect(container.textContent ?? '').toContain('already-running');

    await act(async () => {
      pressKey('ArrowUp');
      await Promise.resolve();
      await Promise.resolve();
    });

    // Cancel was attempted, but the backend said it had already dispatched.
    expect(cancelPendingInputCalls.length).toBe(1);
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea.value).toBe(''); // text NOT pulled back (no double-send)
    // The bubble stays exactly once (now a normal sent message); no duplicate.
    const count = (container.textContent ?? '').split('already-running').length - 1;
    expect(count).toBe(1);
    // The stale waiting overlay is dropped.
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });

  // ─── v3 §11d.4 + §12 R3: same-session queued (200) ────────────────────────

  it('same-session queued (200) keeps the bubble and shows the same-session line (no Run-now)', async () => {
    injectMessageMock = async () => ({ status: 'queued_same_session' });
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      typeInComposer('queued same');
    });
    const btn =
      (container.querySelector('button[aria-label="Send"]') as HTMLButtonElement | null) ??
      ([...container.querySelectorAll('button')].find(
        (b) => (b.textContent ?? '').trim() === 'Queue',
      ) as HTMLButtonElement);
    await act(async () => {
      btn.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(container.textContent ?? '').toContain('queued same'); // bubble kept
    const body = container.querySelector('[data-testid="pending-input-notice-body"]');
    expect(body?.textContent ?? '').toContain('Waiting for the current response to finish.');
    // No holder to cancel → no Run-now.
    expect(container.querySelector('[data-testid="pending-input-notice-run-now"]')).toBeNull();
  });

  it('wait-state queued (plain "queued") shows NO waiting line — not recallable (Finding 1)', async () => {
    injectMessageMock = async () => ({ status: 'queued' });
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      typeInComposer('wait state msg');
    });
    const btn =
      (container.querySelector('button[aria-label="Send"]') as HTMLButtonElement | null) ??
      ([...container.querySelectorAll('button')].find(
        (b) => (b.textContent ?? '').trim() === 'Queue',
      ) as HTMLButtonElement);
    await act(async () => {
      btn.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    // The bubble is kept, but a wait-state "queued" message is NOT a recallable
    // session._queue entry → no waiting line / affordance.
    expect(container.querySelector('[data-testid="pending-input-notice"]')).toBeNull();
  });

  it('reconcilePending restores a same-session queued entry from GET /pending', async () => {
    getPendingMock = async () => ({
      holder: 's1',
      pending: [
        { session_id: 's1', nonce: 'n-same-rec', content: 'same recovered', position: 0, kind: 'same' },
      ],
    });
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();
    // Bubble + same-session waiting line restored from the server.
    expect(container.textContent ?? '').toContain('same recovered');
    const body = container.querySelector('[data-testid="pending-input-notice-body"]');
    expect(body?.textContent ?? '').toContain('Waiting for the current response to finish.');
    expect(container.querySelector('[data-testid="pending-input-notice-run-now"]')).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────
// Bug29 D2: a message sent WHILE the agent is mid-turn pushes the optimistic
// user bubble into rawMessages. That recomputes historyItems and (before this
// fix) triggered the seed effect's wholesale setItems(historyItems) reseed,
// which wiped every in-flight overlay item not backed by rawMessages — the live
// capsule's tool rows, streamed reasoning, sub-agent bubbles, and any final that
// was just finalized by handleSend. The seed effect now skips the reseed while
// the viewed session's turn is in flight and there is rendered content to keep.
// ─────────────────────────────────────────────────────────────────────────
describe('Bug29 D2: mid-turn send preserves the live overlay', () => {
  function findQueueButton(): HTMLButtonElement {
    return (
      (container.querySelector('button[aria-label="Send"]') as HTMLButtonElement | null) ??
      ([...container.querySelectorAll('button')].find(
        (b) => (b.textContent ?? '').trim() === 'Queue',
      ) as HTMLButtonElement)
    );
  }

  // Seed s1 with one history line, view it as the running/waiting HOLDER, then
  // emit a tool activity so a live capsule renders. Asserts the pre-send state
  // (capsule + its tool row visible) as the baseline the send must preserve.
  async function seedHistoryAndLiveCapsule(status: string) {
    runStatusHolder = 's1';
    chatInitialResponse = {
      data: [
        {
          role: 'user',
          content: 'earlier question',
          source: 'user',
          timestamp: '2026-07-23T00:00:00.000Z',
        },
      ],
      total: 1,
    };
    await renderChat({ agentStatus: status, sessionId: 's1' });
    await flushEffects();
    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
        session_id: 's1',
        category: 'file_read',
        tool_name: 'Read',
        description: 'reading live-tool-file.txt',
        id: 'act-d2',
        timestamp: new Date().toISOString(),
      });
    });
    expect(container.querySelector('[data-testid="agent_run"]')).toBeTruthy();
    expect(container.textContent ?? '').toContain('live-tool-file.txt');
  }

  async function sendMidTurn(text: string) {
    await act(async () => {
      typeInComposer(text);
    });
    const btn = findQueueButton();
    await act(async () => {
      btn.click();
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  // The capsule collapses when handleSend finalizes it (it is no longer the
  // last/running item), so surface its preserved tool row by expanding it.
  async function expandCapsuleAndAssertToolRow(capsule: HTMLElement) {
    await act(async () => {
      (capsule.querySelector('button') as HTMLButtonElement).click();
      await Promise.resolve();
    });
    expect(container.textContent ?? '').toContain('live-tool-file.txt');
  }

  it('keeps the live capsule tool rows on a queued_same_session send', async () => {
    injectMessageMock = async () => ({ status: 'queued_same_session' });
    await seedHistoryAndLiveCapsule('running');
    await sendMidTurn('mid-turn message');

    // a. the optimistic user bubble renders.
    expect(container.textContent ?? '').toContain('mid-turn message');
    // b. the live capsule survived the rawMessages-push reseed (the bug wiped
    //    the whole capsule element).
    const capsule = container.querySelector('[data-testid="agent_run"]') as HTMLElement | null;
    expect(capsule).toBeTruthy();
    await expandCapsuleAndAssertToolRow(capsule!);
    // earlier persisted history is intact too.
    expect(container.textContent ?? '').toContain('earlier question');
  });

  it('keeps the live capsule tool rows on a wait-state {"status":"queued"} send', async () => {
    injectMessageMock = async () => ({ status: 'queued' });
    await seedHistoryAndLiveCapsule('waiting');
    await sendMidTurn('wait-state message');

    expect(container.textContent ?? '').toContain('wait-state message');
    const capsule = container.querySelector('[data-testid="agent_run"]') as HTMLElement | null;
    expect(capsule).toBeTruthy();
    await expandCapsuleAndAssertToolRow(capsule!);
  });

  it('tab-switch mid-turn to the running holder renders the NEW session history (criterion c)', async () => {
    // View s1 (holder, running) with history + a live capsule overlay.
    await seedHistoryAndLiveCapsule('running');
    // Switch to s2, itself the running holder, with its own distinct history.
    runStatusHolder = 's2';
    chatInitialResponse = {
      data: [
        {
          role: 'user',
          content: 's2-only-history',
          source: 'user',
          timestamp: '2026-07-23T01:00:00.000Z',
        },
      ],
      total: 1,
    };
    await renderChat({ agentStatus: 'running', sessionId: 's2' });
    await flushEffects();

    // The new session's history replaced s1's stale overlay — the skip must not
    // strand the prior session's content when the viewed session changes.
    expect(container.textContent ?? '').toContain('s2-only-history');
    expect(container.textContent ?? '').not.toContain('earlier question');
    expect(container.textContent ?? '').not.toContain('live-tool-file.txt');
  });
});

describe('T5 ChatView: opening a project never auto-starts an agent', () => {
  it('does not call startAgent on open, regardless of session/agent state', async () => {
    // Supervisor model: opening a project is pure navigation. The agent only
    // starts when the user sends a message — never automatically. The legacy
    // auto-start useEffect (which fired on first open of an empty session) was
    // removed.
    await renderChat({ agentStatus: 'idle', sessionId: 's1' }); // empty + idle
    await flushEffects();
    expect(startAgentCalls.length).toBe(0);

    // Switching to another empty session must not auto-start either.
    await renderChat({ agentStatus: 'idle', sessionId: 's2' });
    await flushEffects();
    expect(startAgentCalls.length).toBe(0);

    // Switching back is still inert.
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();
    expect(startAgentCalls.length).toBe(0);
  });
});

// ─── T6: queue-active composer gating ─────────────────────────────────────

describe('T6 ChatView: queue-active composer gating', () => {
  it('renders ComposerDisabledPrompt instead of the textarea when the queue is running', async () => {
    // Regression for the queue-state vocab mismatch: backend reports 'running'
    // (not 'draining'). queueActive must be driven by the REAL backend value.
    queueState = 'running';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // The composer-disabled prompt must be present.
    const prompt = container.querySelector('[data-testid="composer-disabled-prompt"]');
    expect(prompt).toBeTruthy();
    expect(prompt?.textContent ?? '').toContain('Pause queue to start chatting.');

    // The normal textarea must NOT be present.
    const textarea = container.querySelector('textarea');
    expect(textarea).toBeNull();
  });

  it('renders the normal composer textarea when the queue is paused', async () => {
    queueState = 'paused';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // Normal textarea should be present.
    const textarea = container.querySelector('textarea');
    expect(textarea).toBeTruthy();

    // The disabled prompt should NOT be present.
    const prompt = container.querySelector('[data-testid="composer-disabled-prompt"]');
    expect(prompt).toBeNull();
  });

  it('renders the normal composer textarea when the queue is idle', async () => {
    // 'idle' (run-mode but nothing to dispatch) is NOT active — the composer
    // stays usable. (A new project's queue defaults to idle, so this is the
    // common case that the old 'draining' check wrongly disabled in product.)
    queueState = 'idle';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    expect(container.querySelector('textarea')).toBeTruthy();
    expect(container.querySelector('[data-testid="composer-disabled-prompt"]')).toBeNull();
  });

  it('gating applies regardless of which session is viewed (project-level)', async () => {
    queueState = 'running';
    // Session A
    await renderChat({ agentStatus: 'idle', sessionId: 'A' });
    await flushEffects();
    expect(container.querySelector('[data-testid="composer-disabled-prompt"]')).toBeTruthy();
    expect(container.querySelector('textarea')).toBeNull();

    // Session B — queue is still running, same behaviour
    await renderChat({ agentStatus: 'idle', sessionId: 'B' });
    await flushEffects();
    expect(container.querySelector('[data-testid="composer-disabled-prompt"]')).toBeTruthy();
    expect(container.querySelector('textarea')).toBeNull();
  });

  it('clicking the pause button in ComposerDisabledPrompt calls stopQueue', async () => {
    queueState = 'running';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    const btn = container.querySelector(
      '[data-testid="composer-pause-queue-btn"]',
    ) as HTMLButtonElement;
    expect(btn).toBeTruthy();

    await act(async () => {
      btn.click();
      await Promise.resolve();
    });

    expect(stopQueueMock).toHaveBeenCalledTimes(1);
  });

  it('shows normal composer when queue transitions from running to paused', async () => {
    // Start with queue active
    queueState = 'running';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();
    expect(container.querySelector('[data-testid="composer-disabled-prompt"]')).toBeTruthy();

    // Queue paused — re-render with updated state
    queueState = 'paused';
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();
    expect(container.querySelector('textarea')).toBeTruthy();
    expect(container.querySelector('[data-testid="composer-disabled-prompt"]')).toBeNull();
  });
});

// ─── Frontend chat-rendering fixes (FE-1 / FE-2 / FE-3) ────────────────────

const TS = '2026-05-08T10:00:00Z';
const TS2 = '2026-05-08T10:00:01Z';
const TS3 = '2026-05-08T10:00:02Z';
const TS4 = '2026-05-08T10:00:03Z';

function tc(id: string, name: string, args = '{}') {
  return { id, type: 'function', function: { name, arguments: args } };
}

describe('FE-3 ChatView: trailing capsule spinner reflects active-running state', () => {
  it('idle session ending on a tool result renders a completed capsule (no spinner)', async () => {
    // A historical/idle session whose final turn ended on tool activity.
    // The viewed session is NOT the holder (runStatusHolder null + idle), so
    // isActivelyRunning is false → the trailing capsule must be completed.
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          reasoning_content: 'reading',
          tool_calls: [tc('c1', 'read')],
        },
        { role: 'tool', content: 'r1', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
      ],
      total: 3,
    };
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    const capsule = container.querySelector('[data-testid="agent_run"]');
    expect(capsule).toBeTruthy();
    expect(capsule?.getAttribute('data-capsule-status')).toBe('completed');
    // No running spinner anywhere in the capsule.
    expect(capsule?.querySelector('.animate-spin')).toBeNull();
  });

  it('idle session ending on a system message renders a completed capsule (no spinner)', async () => {
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          tool_calls: [tc('c1', 'agent_message')],
        },
        { role: 'tool', content: 'unknown', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
        { role: 'system', content: 'Repetitive action detected.', source: 'management', timestamp: TS4 },
      ],
      total: 4,
    };
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    const capsule = container.querySelector('[data-testid="agent_run"]');
    expect(capsule).toBeTruthy();
    expect(capsule?.getAttribute('data-capsule-status')).toBe('completed');
    expect(capsule?.querySelector('.animate-spin')).toBeNull();
  });

  it('actively-running holder session keeps the trailing capsule running (spinner shown)', async () => {
    // The viewed session IS the holder and the agent is running →
    // isActivelyRunning true → trailing capsule stays running with a spinner.
    runStatusHolder = 's1';
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          reasoning_content: 'working',
          tool_calls: [tc('c1', 'shell')],
        },
        { role: 'tool', content: 'r1', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
      ],
      total: 3,
    };
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    const capsule = container.querySelector('[data-testid="agent_run"]');
    expect(capsule).toBeTruthy();
    expect(capsule?.getAttribute('data-capsule-status')).toBe('running');
    expect(capsule?.querySelector('.animate-spin')).toBeTruthy();
  });
});

describe('ChatView: capsule expand-by-default behavior', () => {
  it('a tool-only capsule (no reasoning) renders COLLAPSED on idle', async () => {
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          // No reasoning_content — only tool calls. Capsule stays collapsed.
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        },
        { role: 'tool', content: 'TOOL-RESULT-HIDDEN', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
        { role: 'assistant', content: 'done', source: 'management', timestamp: TS4 },
      ],
      total: 4,
    };
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // Tool result is inside the collapsed capsule body — not in the DOM.
    expect(container.textContent ?? '').not.toContain('TOOL-RESULT-HIDDEN');
  });

  it('a content-null capsule WITH reasoning renders COLLAPSED on idle (locked decision: completed ⇒ collapsed)', async () => {
    // LOCKED PRODUCT DECISION (2026-06-03): a COMPLETED turn that reasoned
    // renders as a COLLAPSED reasoning capsule (clean summary). It only expands
    // while actively RUNNING. This session is IDLE (runStatusHolder = null), so
    // the reasoning body must be hidden behind the collapsed capsule. (This test
    // previously asserted reasoning was visible-by-default under the old
    // contract — rewritten to the new invariant.)
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          reasoning_content: 'REASONING-HIDDEN-WHEN-COLLAPSED deciding which file to read',
          tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        },
        { role: 'tool', content: 'TOOL-RESULT-HIDDEN', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
        { role: 'assistant', content: 'done', source: 'management', timestamp: TS4 },
      ],
      total: 4,
    };
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // Completed ⇒ collapsed: the reasoning text is inside the collapsed capsule
    // body and not rendered in the DOM. The capsule itself is still present
    // (the turn never vanishes) and the final answer is visible.
    expect(container.textContent ?? '').not.toContain('REASONING-HIDDEN-WHEN-COLLAPSED');
    expect(container.textContent ?? '').not.toContain('TOOL-RESULT-HIDDEN');
    expect(container.textContent ?? '').toContain('done');
  });
});

describe('FE-1 ChatView: transform-once pairs tool results across page seams', () => {
  it('pairs a boundary tool result after "Load earlier" instead of dropping it', async () => {
    // Initial page (most-recent 50) starts with the tool result whose
    // tool_call lives in the OLDER page. With per-page transform this result
    // would be an orphan and dropped; transform-once on the full raw list
    // pairs it correctly once both pages are loaded.
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        // Boundary tool result (its call is in the older page).
        { role: 'tool', content: 'SEAM-RESULT-CONTENT', source: 'management', timestamp: TS3, tool_call_id: 'seam' },
        { role: 'assistant', content: 'continuing', source: 'management', timestamp: TS4 },
      ],
      // total must exceed loadedOffset (CHAT_PAGE_SIZE=100) so hasMore is true
      // and the "Load earlier" button renders.
      total: 120,
    };
    chatOlderResponse = {
      data: [
        { role: 'user', content: 'older question', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          reasoning_content: 'reading across the seam',
          tool_calls: [tc('seam', 'read', '{"path":"seam.txt"}')],
        },
      ],
      total: 120,
    };

    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // Before loading earlier, the boundary result has no matching call → it is
    // an orphan and not paired into any capsule (capsule render shows it only
    // when paired). The "Load earlier" button is present because hasMore.
    const loadMoreBtn = Array.from(container.querySelectorAll('button')).find((b) =>
      (b.textContent ?? '').toLowerCase().includes('load earlier'),
    ) as HTMLButtonElement | undefined;
    expect(loadMoreBtn).toBeTruthy();

    await act(async () => {
      loadMoreBtn!.click();
      await Promise.resolve();
    });
    await flushEffects();

    // After the older page loads, the full raw list is transformed once and
    // the seam tool_call_row is paired with its result. Expand the capsule to
    // surface the paired result content.
    const capsule = container.querySelector('[data-testid="agent_run"]') as HTMLElement | null;
    expect(capsule).toBeTruthy();
    // This capsule carries reasoning, so it is expanded by default (its id is
    // seeded into expandedCapsules) and its tool rows are already present. Only
    // click the header to expand if it happens to be collapsed — clicking an
    // already-expanded capsule would collapse it and hide the tool rows.
    if (!capsule!.querySelector('.border-t')) {
      await act(async () => {
        (capsule!.querySelector('button') as HTMLButtonElement).click();
        await Promise.resolve();
      });
    }
    // Now expand each tool row (skipping the header) to reveal the paired result.
    const toolRowButtons = (Array.from(capsule!.querySelectorAll('button')) as HTMLButtonElement[]).slice(1);
    await act(async () => {
      for (const b of toolRowButtons) {
        if (!b.disabled) b.click();
      }
      await Promise.resolve();
    });

    // The paired result content is now surfaced — proving the seam tool result
    // was paired (received), not dropped as an orphan.
    expect(container.textContent ?? '').toContain('SEAM-RESULT-CONTENT');
  });
});

// Live-reasoning anchoring: the live stream must emit an agent header before a
// fresh reasoning capsule (parity with chatTransform FE-A3), so live "thinking"
// shows the agent avatar instead of floating under the user message — and, in
// cold-start (no preceding user message), is attributed to the management agent.
describe('appendLiveReasoning — agent header anchor', () => {
  it('cold-start (empty list): inserts a management header before the capsule', () => {
    const out = appendLiveReasoning([], 'thinking…', '2026-06-08T00:00:00Z', 'management');
    expect(out).toHaveLength(2);
    const header = out[0];
    expect(header.type).toBe('agent_message');
    if (header.type === 'agent_message') {
      expect(header.isHeaderOnly).toBe(true);
      expect(header.source).toBe('management');
    }
    expect(out[1].type).toBe('agent_run');
  });

  it('normal turn: reasoning after a user message inserts an agent header (not glued to the user message)', () => {
    const prev: DisplayItem[] = [
      { type: 'user_message', content: 'hi', timestamp: '2026-06-08T00:00:00Z' },
    ];
    const out = appendLiveReasoning(prev, 'thinking…', '2026-06-08T00:00:01Z', 'management');
    expect(out).toHaveLength(3);
    expect(out[0].type).toBe('user_message');
    expect(out[1].type).toBe('agent_message');
    if (out[1].type === 'agent_message') expect(out[1].isHeaderOnly).toBe(true);
    expect(out[2].type).toBe('agent_run');
  });

  it('extending the running capsule does NOT emit a second header', () => {
    const first = appendLiveReasoning([], 'think a', '2026-06-08T00:00:00Z', 'management');
    const second = appendLiveReasoning(first, ' think b', '2026-06-08T00:00:01Z', 'management');
    expect(second.filter((i) => i.type === 'agent_message' && i.isHeaderOnly).length).toBe(1);
    expect(second.filter((i) => i.type === 'agent_run').length).toBe(1);
    const cap = second.find((i) => i.type === 'agent_run');
    expect(cap?.type).toBe('agent_run');
    if (cap && cap.type === 'agent_run') {
      expect(cap.items).toHaveLength(1);
      const block = cap.items[0];
      expect(block.type).toBe('reasoning_block');
      if (block.type === 'reasoning_block') {
        expect(block.content).toBe('think a think b');
      }
    }
  });
});

describe('ChatView: full-history reconcile after a turn settles (TASK-history-reconcile-after-turn)', () => {
  // Captured bug: a session that starts EMPTY is carried entirely by the WS
  // overlay + limit=1 fallback; the full-history limit=100 reconcile never
  // fires, so middle assistant turns silently disappear from the view while
  // remaining in the backend (DIAGNOSIS-history-loss-subcause.md). After the
  // VIEWED session's turn settles, the view must reconcile against full history.
  const FULL_HISTORY = [
    { role: 'user', content: 'Q1-what-is-it', timestamp: '2026-06-15T15:41:38.000000+00:00', session_id: 's1' },
    { role: 'assistant', content: 'A1-MIDDLE-ANSWER-UNIQUE', timestamp: '2026-06-15T15:42:49.000000+00:00', session_id: 's1', source: 'assistant' },
    { role: 'user', content: 'Q2-side-effects', timestamp: '2026-06-15T15:43:04.000000+00:00', session_id: 's1' },
    { role: 'assistant', content: 'A2-LATEST-ANSWER', timestamp: '2026-06-15T15:43:26.000000+00:00', session_id: 's1', source: 'assistant' },
  ];
  const initialChatCalls = () =>
    apiWithTotalCalls.filter((p) => p.includes('/chat?limit=') && !p.includes('offset=')).length;

  it('reconciles the viewed session against full history after its turn settles', async () => {
    // Reproduce the captured condition: session starts EMPTY and the holder is
    // unresolved (null) at the running transition, so the wasRunningRef/viewing
    // latch never fires via the agentStatus path.
    runStatusHolder = null;
    chatInitialResponse = { data: [], total: 0 };

    // 1. Mount idle, empty session -> items=[] (initial limit=100 load).
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // 2. Agent starts running; viewing is false (holder null), so the running
    //    branch does NOT latch wasRunningRef.
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    // 3. A stream delta arrives for the VIEWED session (definitive proof it is
    //    streaming). The overlay does NOT contain the middle answer.
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'A2-LATEST streaming token',
        source: 'assistant',
        is_final: false,
      });
    });
    await flushEffects();

    // 4. Backend now holds the FULL persisted history (all 4 messages).
    chatInitialResponse = { data: FULL_HISTORY, total: FULL_HISTORY.length };
    const before = initialChatCalls();

    // 5. The turn settles: running -> idle.
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // (a) A full-history reconcile fired for the viewed session after the turn.
    expect(initialChatCalls()).toBeGreaterThan(before);
    expect(
      apiWithTotalCalls.some(
        (p) => p.includes('/chat?limit=') && p.includes('session_id=s1') && !p.includes('offset='),
      ),
    ).toBe(true);

    // (b) The view now contains ALL turns, including the middle answer the
    //     overlay lacked -- not just the latest.
    expect(container.textContent ?? '').toContain('A1-MIDDLE-ANSWER-UNIQUE');
  });

  it('does NOT reconcile when switching to a different idle session mid-turn (no fetch storm)', async () => {
    // s1 is the running holder and is being viewed; the user switches to view
    // idle s2 BEFORE s1 finishes. When s1 then goes idle, no full-history
    // reconcile must fire (the "was streaming" latch must reset on switch).
    runStatusHolder = 's1';
    chatInitialResponse = { data: [], total: 0 };

    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 's1 streaming',
        source: 'assistant',
        is_final: false,
      });
    });
    await flushEffects();

    // Switch to view s2 (idle) while s1 is still the running holder.
    await renderChat({ agentStatus: 'running', sessionId: 's2' });
    await flushEffects();
    const before = initialChatCalls();

    // s1's loop ends -> project agentStatus goes idle while viewing s2.
    await renderChat({ agentStatus: 'idle', sessionId: 's2' });
    await flushEffects();

    // No spurious reconcile fired on the idle transition after the switch.
    expect(initialChatCalls()).toBe(before);
  });
});

describe('credential-error surfacing (AgentErrorNotice)', () => {
  it('shows the notice when agent.status error arrives for the viewed session', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();
    expect(container.querySelector('[data-testid="agent-error-notice"]')).toBeNull();

    await act(async () => {
      emitWs('agent.status', {
        type: 'agent.status',
        project_id: 'p1',
        status: 'error',
        reason: 'No LLM API key configured',
        error_code: 'missing_api_key',
        session_id: 's1',
        source: 'management',
      });
    });
    const notice = container.querySelector('[data-testid="agent-error-notice"]');
    expect(notice).not.toBeNull();
    expect(notice!.textContent).toMatch(/API key/i);
    expect(notice!.textContent).toContain('No LLM API key configured');
  });

  it('clears the notice when a new run starts (non-error status)', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();
    await act(async () => {
      emitWs('agent.status', {
        type: 'agent.status',
        project_id: 'p1',
        status: 'error',
        reason: 'x',
        error_code: 'missing_api_key',
        session_id: 's1',
      });
    });
    expect(container.querySelector('[data-testid="agent-error-notice"]')).not.toBeNull();
    await act(async () => {
      emitWs('agent.status', {
        type: 'agent.status',
        project_id: 'p1',
        status: 'running',
        session_id: 's1',
      });
    });
    expect(container.querySelector('[data-testid="agent-error-notice"]')).toBeNull();
  });

  it('hydrates the notice from run-status last_terminal_event on mount', async () => {
    runStatusTerminalEvent = {
      type: 'error',
      details: 'No LLM API key configured',
      error_code: 'missing_api_key',
      timestamp: '2026-07-17T00:00:00Z',
    };
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
    await flushEffects();
    const notice = container.querySelector('[data-testid="agent-error-notice"]');
    expect(notice).not.toBeNull();
    expect(notice!.textContent).toMatch(/API key/i);
  });

  it('shows a classified inline error on the cold-start card when scan fails', async () => {
    const { ApiError } = await import('../config');
    coldStartScanMock = async () => {
      throw new ApiError(
        400,
        JSON.stringify({
          code: 'missing_api_key',
          message: 'No LLM API key configured',
        }),
      );
    };
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={{ ...project, is_empty_workspace: false }}
          agentStatus="idle"
          mentionAgents={[]}
        />,
      );
    });
    await flushEffects();
    const scanBtn = [...container.querySelectorAll('button')].find(
      (b) => /scan/i.test(b.textContent ?? ''),
    );
    expect(scanBtn).toBeTruthy();
    await act(async () => {
      scanBtn!.click();
    });
    await flushEffects();
    const err = container.querySelector('[data-testid="cold-start-error"]');
    expect(err).not.toBeNull();
    expect(err!.textContent).toMatch(/API key/i);
  });
});

// --------------------------------------------------------------------------
// Sub-agent lifecycle marker rows, end to end (backlog #27). The transform's
// own parity tests (utils/chatTransform.test.ts + the paired pytest) prove a
// marker PARSES; they cannot see what the user is told about it, and that is
// where the second half of this bug lived: the label chain here ended in an
// unguarded `/* failed */` fallback, so a marker that parsed correctly still
// rendered as "failed". These tests close that gap by driving the same
// fixture file through the real component.
// --------------------------------------------------------------------------

const markerFixtures = subAgentMarkerFixturesData as Array<{
  shape: string;
  action: string;
  content: string;
}>;

function sysRow(content: string) {
  return {
    role: 'system',
    content,
    source: 'management',
    session_id: 's1',
    timestamp: new Date().toISOString(),
  };
}

describe('ChatView: [Sub-agent] lifecycle marker rows (backlog #27)', () => {
  it('renders a row for EVERY producer shape in the parity fixture', async () => {
    chatInitialResponse = {
      data: markerFixtures.map((f) => sysRow(f.content)),
      total: markerFixtures.length,
    };
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    const rows = [...container.querySelectorAll('[data-testid="sub-agent-activity"]')];
    expect(rows.length).toBe(markerFixtures.length);
    expect(rows.map((r) => r.getAttribute('data-action'))).toEqual(
      markerFixtures.map((f) => f.action),
    );
  });

  it('a user stop is never labelled a failure, and says what it killed', async () => {
    // The whole point of #27's second half: before the exhaustive switch,
    // 'stopped' fell through to the failed copy and the timeline told the
    // user their own stop request had failed.
    chatInitialResponse = {
      data: [
        sysRow(
          '[Sub-agent] claude-code stopped by user. Terminated 2 background ' +
            'process(es): npm run dev; python server.py. This background work did NOT complete.',
        ),
      ],
      total: 1,
    };
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    const row = container.querySelector('[data-testid="sub-agent-activity"]');
    expect(row).not.toBeNull();
    const text = row!.textContent ?? '';
    expect(text).not.toMatch(/fail/i);
    expect(text).toContain('stopped by you');
    // The destroyed background work is the part the user did not ask for,
    // so it must be on the row, not swallowed.
    expect(text).toContain('2');
    expect(text).toContain('npm run dev');
  });

  it('a blocked sub-agent shows its question, never the agent-facing instructions', async () => {
    chatInitialResponse = {
      data: [
        sysRow(
          '[Sub-agent] claude-code requires input (question): Which file should I edit? ' +
            'Respond on the same in-flight request with agent_message(action="respond", ' +
            'agent="claude-code", interaction_id="int-1", ...). For a free-form question, ' +
            'put the answer in message. Do not send a new task.',
        ),
      ],
      total: 1,
    };
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    const row = container.querySelector('[data-testid="sub-agent-activity"]');
    expect(row).not.toBeNull();
    const text = row!.textContent ?? '';
    expect(text).toContain('Which file should I edit?');
    expect(text).not.toContain('agent_message(');
    expect(text).not.toContain('Do not send a new task');
  });

  it('tones rank by severity: lost work and interruptions warn, a plain stop does not', async () => {
    chatInitialResponse = {
      data: [
        sysRow('[Sub-agent] claude-code stopped by user.'),
        sysRow(
          '[Sub-agent] claude-code teardown terminated 1 live background process(es): ' +
            'npm run dev. This background work did NOT complete.',
        ),
        sysRow(
          '[Sub-agent] claude-code was stopped before completing its current task ' +
            '(turn interrupted — e.g. an approval denied with stop). No result was ' +
            'produced. The agent remains available. Transcript: /x/y.jsonl',
        ),
        sysRow('[Sub-agent] claude-code stopped with error: boom'),
      ],
      total: 4,
    };
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    const byAction = Object.fromEntries(
      [...container.querySelectorAll('[data-testid="sub-agent-activity"]')].map((r) => [
        r.getAttribute('data-action'),
        r.className,
      ]),
    );
    // A stop the user asked for is not an alarm.
    expect(byAction['stopped']).toContain('text-secondary');
    expect(byAction['background_lost']).toContain('text-warning');
    expect(byAction['interrupted']).toContain('text-warning');
    expect(byAction['error']).toContain('text-error');
  });

  // backlog #35a: the three things the borrowed `failed` shape got wrong.
  it('a dropped queued message warns in amber and claims no failed dispatch', async () => {
    chatInitialResponse = {
      data: [
        sysRow(
          '[Sub-agent] claude-code queued message dropped: agent stopped before dispatch.',
        ),
      ],
      total: 1,
    };
    await renderChat({ sessionId: 's1' });
    await flushEffects();

    const row = container.querySelector('[data-testid="sub-agent-activity"]');
    expect(row).not.toBeNull();
    expect(row!.getAttribute('data-action')).toBe('queue_dropped');
    // Amber, not red: the user's own stop is the usual cause and nothing
    // malfunctioned — but a message of theirs was thrown away, so not neutral.
    expect(row!.className).toContain('text-warning');
    expect(row!.className).not.toContain('text-error');

    const text = row!.textContent ?? '';
    expect(text).toContain('agent stopped before dispatch');
    expect(text).not.toMatch(/fail/i);
    // on_failed's hardcoded tail claimed a dispatch that never happened.
    expect(text).not.toContain('The dispatched task did not complete');
    // "failed: queued message dropped: …" rendered two colons in a row.
    expect(text).not.toMatch(/:\s*[^:]*:\s/);
  });
});

// backlog #44: during a streaming response the user must be able to scroll up
// to read earlier content without the transcript snapping back to the bottom on
// every token. These tests stub the scroll container's geometry (jsdom cannot
// measure real layout) and drive stream deltas through the existing emitWs bus.
describe('ChatView: scroll-up during streaming (backlog #44)', () => {
  let rafSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Run requestAnimationFrame callbacks synchronously so scrollToBottom's
    // scrollTop write is observable within the same act() flush.
    rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        cb(0);
        return 0;
      });
  });

  afterEach(() => {
    rafSpy.mockRestore();
  });

  /** Give the scroll container measurable geometry with a writable scrollTop. */
  function stubScrollGeometry(
    el: HTMLElement,
    scrollHeight: number,
    clientHeight: number,
    scrollTop: number,
  ) {
    let top = scrollTop;
    Object.defineProperty(el, 'scrollHeight', { configurable: true, get: () => scrollHeight });
    Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => clientHeight });
    Object.defineProperty(el, 'scrollTop', {
      configurable: true,
      get: () => top,
      set: (v: number) => {
        top = v;
      },
    });
  }

  function getScrollEl(): HTMLElement {
    const el = container.querySelector('div.overflow-y-auto') as HTMLElement | null;
    expect(el).not.toBeNull();
    return el!;
  }

  /** Simulate a genuine user scroll-up. The mount reset arms the ~150ms
   *  programmatic-scroll guard, so wait past it first — a real user scrolls
   *  well after the session has settled. */
  async function userScrollTo(scrollEl: HTMLElement, scrollTop: number) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 160));
    });
    scrollEl.scrollTop = scrollTop;
    await act(async () => {
      scrollEl.dispatchEvent(new Event('scroll', { bubbles: true }));
    });
  }

  it('does NOT snap back and surfaces the jump-to-latest pill after the user scrolls up', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    const scrollEl = getScrollEl();
    // Tall content, short viewport, parked at the very top (user scrolled up).
    stubScrollGeometry(scrollEl, 1000, 500, 0);
    await userScrollTo(scrollEl, 0);

    // A stream delta arrives for the viewed session.
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'streamed-token-SCROLL',
        source: 'assistant',
        is_final: false,
      });
    });

    // Content rendered, but the viewport stayed where the user left it.
    expect(container.textContent ?? '').toContain('streamed-token-SCROLL');
    expect(scrollEl.scrollTop).toBe(0);

    // The arrow-only pill appears (no count in its label).
    const pill = container.querySelector('[data-testid="jump-to-latest"]');
    expect(pill).not.toBeNull();
    expect(pill!.getAttribute('aria-label')).toBe('Jump to latest');
    expect(pill!.textContent ?? '').not.toMatch(/\d/);
  });

  it('clicking the pill jumps to the bottom and resumes auto-follow', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    const scrollEl = getScrollEl();
    stubScrollGeometry(scrollEl, 1000, 500, 0);
    await userScrollTo(scrollEl, 0);
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'first-token',
        source: 'assistant',
        is_final: false,
      });
    });

    const pill = container.querySelector('[data-testid="jump-to-latest"]') as HTMLElement;
    expect(pill).not.toBeNull();

    // Click the pill -> forced jump to the bottom.
    await act(async () => {
      pill.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(scrollEl.scrollTop).toBe(1000);
    // Follow resumed: the pill is gone.
    expect(container.querySelector('[data-testid="jump-to-latest"]')).toBeNull();

    // A subsequent delta now keeps the view pinned to the bottom (auto-follow).
    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        session_id: 's1',
        text: 'second-token',
        source: 'assistant',
        is_final: false,
      });
    });
    expect(scrollEl.scrollTop).toBe(1000);
    expect(container.querySelector('[data-testid="jump-to-latest"]')).toBeNull();
  });
});

describe('Bug #48: session-switch history cache + fetch abort', () => {
  const S1_MSG = {
    role: 'user',
    content: 'cached-s1-message',
    source: 'user',
    timestamp: '2026-08-11T00:00:00.000Z',
  };

  function renderSession(sessionId: string) {
    return act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
          sessionId={sessionId}
        />,
      );
    });
  }

  it('switch-back paints cached history immediately (no skeleton) and revalidates in the background', async () => {
    chatInitialResponse = { data: [S1_MSG], total: 1 };
    await renderSession('s1');
    await flushEffects();
    expect(container.textContent).toContain('cached-s1-message');

    // Switch away to s2 (empty history) — s1's transcript leaves the DOM.
    chatInitialResponse = { data: [], total: 0 };
    await renderSession('s2');
    await flushEffects();
    expect(container.textContent).not.toContain('cached-s1-message');

    // Hold the revalidate fetch open, then switch back to s1: the cached
    // transcript must already be visible while the refetch is in flight
    // (i.e. no loading skeleton replacing the pane).
    let releaseChat!: () => void;
    chatResponseGate = new Promise<void>((r) => {
      releaseChat = r;
    });
    chatInitialResponse = { data: [S1_MSG], total: 1 };
    await renderSession('s1');
    expect(container.textContent).toContain('cached-s1-message');

    // Release the gate — the background revalidate for s1 must have fired
    // (initial load + revalidate = exactly two s1 history fetches).
    releaseChat();
    chatResponseGate = null;
    await flushEffects();
    const s1Fetches = apiWithTotalCalls.filter(
      (p) => p.includes('/chat?limit=') && p.includes('session_id=s1'),
    );
    expect(s1Fetches.length).toBe(2);
    expect(container.textContent).toContain('cached-s1-message');
  });

  it('switching away aborts the in-flight history fetch', async () => {
    let releaseChat!: () => void;
    chatResponseGate = new Promise<void>((r) => {
      releaseChat = r;
    });
    chatInitialResponse = { data: [S1_MSG], total: 1 };
    await renderSession('s1');

    const s1Index = apiWithTotalCalls.findIndex(
      (p) => p.includes('/chat?limit=') && p.includes('session_id=s1'),
    );
    expect(s1Index).toBeGreaterThanOrEqual(0);
    const s1Signal = apiWithTotalSignals[s1Index];
    expect(s1Signal).toBeDefined();
    expect(s1Signal!.aborted).toBe(false);

    // Switching sessions must abort the superseded request, not merely
    // ignore its result.
    await renderSession('s2');
    expect(s1Signal!.aborted).toBe(true);

    releaseChat();
    chatResponseGate = null;
    await flushEffects();
  });
});

// Mid-run transcript loss (investigation 2026-08-15). The module-level
// chatHistoryCache is only ever written by the session-load effect —
// refreshRawMessages (the running→idle catch-up) and the live WS appends never
// write back. So an entry can go stale, and a remount mid-run repaints that
// stale snapshot. The seed effect's mid-turn skip then refuses to replace
// `items` for the rest of the turn, pinning the stale paint until a full page
// reload clears the module cache.
//
// The Bug #48 tests above pass only because they run at agentStatus="idle",
// where the mid-turn skip never arms.
describe('mid-run remount: stale history cache pins the transcript', () => {
  const ROUND_1 = {
    role: 'user',
    content: 'round-one-message',
    source: 'user',
    timestamp: '2026-08-15T06:26:23.698Z',
  };
  const ROUND_2 = {
    role: 'user',
    content: 'round-two-message',
    source: 'user',
    timestamp: '2026-08-15T06:28:03.403Z',
  };

  function renderChatView(agentStatus: 'idle' | 'running') {
    return act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus={agentStatus}
          mentionAgents={[]}
          sessionId="s1"
        />,
      );
    });
  }

  function remountFresh() {
    // A tab switch (Queue tab and back) unmounts ChatView. Per-instance refs
    // reset; the module-level chatHistoryCache survives — that asymmetry is
    // the bug's precondition.
    return act(async () => {
      root.unmount();
      container.remove();
      container = document.createElement('div');
      document.body.appendChild(container);
      root = createRoot(container);
    });
  }

  it('renders a round persisted after the cache entry was written', async () => {
    // 1. First view of s1 while idle: only round 1 exists. This is what
    //    populates chatHistoryCache with a one-round snapshot.
    runStatusHolder = 's1';
    chatInitialResponse = { data: [ROUND_1], total: 1 };
    await renderChatView('idle');
    await flushEffects();
    expect(container.textContent).toContain('round-one-message');

    await remountFresh();

    // 2. Round 2 has since been sent and persisted, and its turn is still in
    //    flight with s1 holding the slot.
    chatInitialResponse = { data: [ROUND_1, ROUND_2], total: 2 };

    // 3. Switch back mid-run, holding the revalidate open so the holder fetch
    //    settles first. That is the real ordering: the cache is consulted, then
    //    the holder resolves, then fresh history lands — which is precisely
    //    when the skip is armed.
    let releaseChat!: () => void;
    chatResponseGate = new Promise<void>((r) => {
      releaseChat = r;
    });
    await renderChatView('running');
    await flushEffects();
    // Option C: the stale one-round snapshot must NOT be painted while the turn
    // is in flight — painting it is what arms the seed effect's mid-turn skip
    // and pins the stale transcript for the rest of the turn. The pane shows
    // the skeleton and waits for the real fetch. This mid-run switch-back flash
    // is the accepted cost of the fix, scoped strictly to turn-in-flight.
    expect(container.textContent).not.toContain('round-one-message');
    expect(container.querySelector('.animate-pulse')).not.toBeNull();

    releaseChat();
    chatResponseGate = null;
    await flushEffects();

    // The revalidate returned BOTH rounds and they are on disk, so BOTH must
    // render. Before the fix the mid-turn skip dropped round 2 until a full
    // page reload cleared the module cache.
    expect(container.textContent).toContain('round-one-message');
    expect(container.textContent).toContain('round-two-message');
  });

  // The stale-EMPTY half of the same defect, provable at idle so it is
  // independent of the mid-turn guard above. A session id is minted before the
  // first inject, and the backend legitimately returns [] for an id that has
  // not materialized yet — so a first view caches `messages: []`. The cache-hit
  // paint then called setItems([]) on that entry, hard-blanking a pane whose
  // session has since filled up, for the whole duration of the revalidate.
  // An empty entry is now treated as a miss: skeleton, then real history.
  it('does not blank the pane from a cache entry written before the session materialized', async () => {
    runStatusHolder = null;
    chatInitialResponse = { data: [], total: 0 };
    await renderChatView('idle');
    await flushEffects();
    expect(container.textContent).not.toContain('round-one-message');

    await remountFresh();

    // The session has materialized since; its history is no longer empty.
    chatInitialResponse = { data: [ROUND_1], total: 1 };
    let releaseChat!: () => void;
    chatResponseGate = new Promise<void>((r) => {
      releaseChat = r;
    });
    await renderChatView('idle');
    await flushEffects();

    // Mid-revalidate: a loading skeleton, not a pane blanked by the empty entry.
    expect(container.querySelector('.animate-pulse')).not.toBeNull();

    releaseChat();
    chatResponseGate = null;
    await flushEffects();

    expect(container.textContent).toContain('round-one-message');
  });

  // Control: the identical stale-cache flow at idle. This passes today, which
  // isolates the defect — a stale cache entry alone is survivable because the
  // revalidate reseeds `items`. It is the mid-turn skip that makes the stale
  // paint permanent for the duration of the turn.
  it('recovers from the same stale cache entry when no turn is in flight', async () => {
    runStatusHolder = 's1';
    chatInitialResponse = { data: [ROUND_1], total: 1 };
    await renderChatView('idle');
    await flushEffects();
    expect(container.textContent).toContain('round-one-message');

    await remountFresh();

    chatInitialResponse = { data: [ROUND_1, ROUND_2], total: 2 };
    let releaseChat!: () => void;
    chatResponseGate = new Promise<void>((r) => {
      releaseChat = r;
    });
    await renderChatView('idle');
    await flushEffects();

    releaseChat();
    chatResponseGate = null;
    await flushEffects();

    expect(container.textContent).toContain('round-two-message');
  });
});

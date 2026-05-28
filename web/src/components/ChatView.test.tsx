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

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';

// React requires this flag to allow act(...) in a test runner. Vitest's
// jsdom environment doesn't set it; we opt in here.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiCalls: string[] = [];
const apiWithTotalCalls: string[] = [];

// Configurable /chat responses keyed by whether the request is the initial
// page (no offset) or a "Load earlier" page (offset present). Tests set these
// to drive ChatView's transform-once history rendering (FE-1/FE-2/FE-3).
let chatInitialResponse: { data: unknown[]; total: number } = { data: [], total: 0 };
let chatOlderResponse: { data: unknown[]; total: number } = { data: [], total: 0 };

// Configurable run-status holder. fetchHolder() and the run-status poll read
// /run-status; tests set this to drive the slot-holder comparison that gates
// live-event rendering. null = no holder.
let runStatusHolder: string | null = null;

vi.mock('../config', () => ({
  api: vi.fn(async (path: string) => {
    apiCalls.push(path);
    // run-status returns the configured slot holder so ChatView can decide
    // whether the viewed session is the holder.
    if (path.includes('/run-status')) {
      return {
        project_id: 'p1',
        status: 'running',
        current_holder_session_id: runStatusHolder,
      };
    }
    // pending-approval: never pending in these tests.
    if (path.includes('/pending-approval')) {
      return { pending: false };
    }
    // /chat?limit=1 (REST fallback) and anything else → empty list.
    return [];
  }),
  apiWithTotal: vi.fn(async (path: string) => {
    apiWithTotalCalls.push(path);
    // /chat?limit=... is the history fetch. An `offset=` param marks a
    // "Load earlier" page; otherwise it's the initial page.
    if (path.includes('/chat?limit=')) {
      if (path.includes('offset=')) return chatOlderResponse;
      return chatInitialResponse;
    }
    return { data: [], total: 0 };
  }),
  ApiError: class ApiError extends Error {},
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
let cancelMessageMock: (projectId: string) => Promise<unknown> = async () => undefined;
// Records the args of the most recent injectMessage call so tests can assert
// the viewed sessionId is threaded through. Replaceable per-test for slot_held.
let injectMessageMock: (...args: unknown[]) => Promise<unknown> = async () => undefined;
const injectCalls: unknown[][] = [];
// Records startAgent calls so the auto-start test can assert it does NOT fire
// on a session switch.
const startAgentCalls: string[] = [];

vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    injectMessage: vi.fn((...args: unknown[]) => {
      injectCalls.push(args);
      return injectMessageMock(...args);
    }),
    startAgent: vi.fn(async (projectId: string) => {
      startAgentCalls.push(projectId);
      return undefined;
    }),
    cancelMessage: vi.fn((projectId: string) => cancelMessageMock(projectId)),
    newSession: vi.fn(async () => undefined),
  }),
}));

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

import ChatView from './ChatView';
import type { Project } from '../types';

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
  injectCalls.length = 0;
  startAgentCalls.length = 0;
  injectMessageMock = async () => undefined;
  runStatusHolder = null;
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

    // Spinner is rendered (lucide-react Loader2 carries class `lucide-loader-2`
    // or `lucide-loader-circle`; we assert presence via the animate-spin class
    // we explicitly set on the icon).
    const spinner = cancelling.querySelector('.animate-spin');
    expect(spinner).toBeTruthy();

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

function renderChat(props: { agentStatus?: string; sessionId?: string }) {
  return act(async () => {
    root.render(
      <ChatView
        projectId="p1"
        project={project}
        agentStatus={(props.agentStatus ?? 'idle') as never}
        mentionAgents={[]}
        sessionId={props.sessionId}
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

describe('T5 ChatView: holder-based live-event filtering', () => {
  it('renders a live stream delta when viewing the holder session', async () => {
    // The viewed session s1 IS the slot holder.
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        text: 'streamed-token-XYZ',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').toContain('streamed-token-XYZ');
  });

  it('renders live deltas during the null-holder window when the agent is active (lenient gate)', async () => {
    // Holder not yet resolved (run-status returns null) but the agent is
    // RUNNING — deltas arriving before fetchHolder converges must still render
    // (otherwise the user sees a mid-stream stall). Mirrors the lenient
    // null-holder policy of fetchPendingApproval.
    runStatusHolder = null;
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        text: 'window-token-ABC',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').toContain('window-token-ABC');
  });

  it('does NOT render live deltas when holder is null AND the agent is idle', async () => {
    // Null holder + idle agent means nothing is running → the lenient branch
    // does not apply, so stray deltas are dropped.
    runStatusHolder = null;
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        text: 'idle-token-NOPE',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').not.toContain('idle-token-NOPE');
  });

  it('does NOT render a live stream delta when viewing a NON-holder session', async () => {
    // The slot holder is s2 but the user is viewing s1.
    runStatusHolder = 's2';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('chat.stream_delta', {
        type: 'chat.stream_delta',
        project_id: 'p1',
        text: 'streamed-token-XYZ',
        source: 'assistant',
        is_final: false,
      });
    });

    expect(container.textContent ?? '').not.toContain('streamed-token-XYZ');
  });

  it('does NOT render a live activity capsule when viewing a NON-holder session', async () => {
    runStatusHolder = 's2';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
        category: 'file_read',
        tool_name: 'Read',
        description: 'reading secret-file.txt',
        id: 'act-1',
        timestamp: new Date().toISOString(),
      });
    });

    // No agent_run capsule rendered for the non-holder session.
    expect(container.querySelector('[data-testid="agent_run"]')).toBeNull();
    expect(container.textContent ?? '').not.toContain('secret-file.txt');
  });

  it('renders a live activity capsule when viewing the holder session', async () => {
    runStatusHolder = 's1';
    await renderChat({ agentStatus: 'running', sessionId: 's1' });
    await flushEffects();

    await act(async () => {
      emitWs('agent.activity', {
        type: 'agent.activity',
        project_id: 'p1',
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

  it('renders SlotHeldNotice when inject returns 202 slot_held', async () => {
    // Inject resolves with a slot_held payload (another session holds the slot).
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

  it('a content-null capsule WITH reasoning renders EXPANDED on idle (FE-A3 / FE-2)', async () => {
    runStatusHolder = null;
    chatInitialResponse = {
      data: [
        { role: 'user', content: 'go', source: 'user', timestamp: TS },
        {
          role: 'assistant',
          content: null,
          source: 'management',
          timestamp: TS2,
          reasoning_content: 'REASONING-VISIBLE-BY-DEFAULT deciding which file to read',
          tool_calls: [tc('c1', 'read', '{"path":"a.txt"}')],
        },
        { role: 'tool', content: 'TOOL-RESULT-VISIBLE', source: 'management', timestamp: TS3, tool_call_id: 'c1' },
        { role: 'assistant', content: 'done', source: 'management', timestamp: TS4 },
      ],
      total: 4,
    };
    await renderChat({ agentStatus: 'idle', sessionId: 's1' });
    await flushEffects();

    // Reasoning makes the capsule expand by default so the user sees the
    // thinking without having to click. Tool result inside the body is
    // visible as a consequence.
    expect(container.textContent ?? '').toContain('REASONING-VISIBLE-BY-DEFAULT');
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
    // Capsules are collapsed by default — click the header (first button) to
    // expand the capsule and render its tool-call rows.
    await act(async () => {
      (capsule!.querySelector('button') as HTMLButtonElement).click();
      await Promise.resolve();
    });
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

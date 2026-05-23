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

vi.mock('../config', () => ({
  api: vi.fn(async (path: string) => {
    apiCalls.push(path);
    return [];
  }),
  apiWithTotal: vi.fn(async (path: string) => {
    apiWithTotalCalls.push(path);
    return { data: [], total: 0 };
  }),
  ApiError: class ApiError extends Error {},
  isRelayMode: false,
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000/ws',
}));

vi.mock('../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    on: () => {},
    off: () => {},
    connectionState: 'connected',
    subscribe: () => {},
  }),
}));

// Per-test override slot for cancelMessage. The default resolves; tests
// that need rejection or controlled timing replace this between renders.
let cancelMessageMock: (projectId: string) => Promise<unknown> = async () => undefined;

vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    injectMessage: vi.fn(async () => undefined),
    startAgent: vi.fn(async () => undefined),
    cancelMessage: vi.fn((projectId: string) => cancelMessageMock(projectId)),
    newSession: vi.fn(async () => undefined),
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
        />,
      );
    });
    await flushEffects();

    const agentsAvailableCalls = [...apiCalls, ...apiWithTotalCalls].filter(
      (p) => p.includes('/agents/available'),
    );
    expect(agentsAvailableCalls).toEqual([]);
  });

  it('issues exactly one /chat?limit= fetch on mount (no piggybacked /agents/available)', async () => {
    await act(async () => {
      root.render(
        <ChatView
          projectId="p1"
          project={project}
          agentStatus="idle"
          mentionAgents={[]}
        />,
      );
    });
    await flushEffects();

    const chatCalls = apiWithTotalCalls.filter((p) => p.includes('/chat?limit='));
    expect(chatCalls.length).toBe(1);
    expect(chatCalls[0]).toContain('/api/v2/agents/p1/chat?limit=50');
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

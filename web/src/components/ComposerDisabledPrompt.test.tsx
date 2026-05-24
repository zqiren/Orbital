// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ComposerDisabledPrompt from './ComposerDisabledPrompt';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

function setup() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
}

function teardown() {
  act(() => {
    root.unmount();
  });
  container.remove();
}

describe('ComposerDisabledPrompt', () => {
  it('renders the prompt text', async () => {
    setup();
    await act(async () => {
      root.render(<ComposerDisabledPrompt onPauseQueue={() => {}} />);
    });

    const el = container.querySelector('[data-testid="composer-disabled-prompt"]');
    expect(el).toBeTruthy();
    expect(el?.textContent ?? '').toContain('Pause queue to start chatting.');

    teardown();
  });

  it('renders a pause button', async () => {
    setup();
    await act(async () => {
      root.render(<ComposerDisabledPrompt onPauseQueue={() => {}} />);
    });

    const btn = container.querySelector(
      '[data-testid="composer-pause-queue-btn"]',
    ) as HTMLButtonElement;
    expect(btn).toBeTruthy();
    expect(btn.disabled).toBe(false);

    teardown();
  });

  it('calls onPauseQueue when the pause button is clicked', async () => {
    setup();
    const onPause = vi.fn();
    await act(async () => {
      root.render(<ComposerDisabledPrompt onPauseQueue={onPause} />);
    });

    const btn = container.querySelector(
      '[data-testid="composer-pause-queue-btn"]',
    ) as HTMLButtonElement;
    await act(async () => {
      btn.click();
    });

    expect(onPause).toHaveBeenCalledTimes(1);
    teardown();
  });

  it('shows a pending/disabled state while pausing (async onPauseQueue)', async () => {
    setup();
    let resolve!: () => void;
    const onPause = vi.fn(
      () => new Promise<void>((res) => { resolve = res; }),
    );
    await act(async () => {
      root.render(<ComposerDisabledPrompt onPauseQueue={onPause} />);
    });

    const btn = container.querySelector(
      '[data-testid="composer-pause-queue-btn"]',
    ) as HTMLButtonElement;

    // Click but don't resolve yet
    act(() => {
      btn.click();
    });

    // While the promise is in-flight the button should be disabled
    const btnAfter = container.querySelector(
      '[data-testid="composer-pause-queue-btn"]',
    ) as HTMLButtonElement;
    expect(btnAfter.disabled).toBe(true);
    expect(btnAfter.textContent ?? '').toContain('Pausing');

    // Resolve and allow React to re-render
    await act(async () => {
      resolve();
      await Promise.resolve();
    });

    teardown();
  });
});

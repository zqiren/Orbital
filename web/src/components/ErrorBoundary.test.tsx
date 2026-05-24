// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';

// A child that throws on demand (module-level flag so we can flip it between
// renders to simulate recovery).
let shouldThrow = true;
function Bomb() {
  if (shouldThrow) throw new Error('boom');
  return <div data-testid="bomb-ok">recovered</div>;
}

describe('ErrorBoundary', () => {
  let errSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    shouldThrow = true;
    // componentDidCatch logs to console.error — silence it so the throwing
    // tests don't spam output (and assert it's still called where relevant).
    errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    errSpy.mockRestore();
  });

  it('renders children normally when they do not throw (pass-through)', () => {
    shouldThrow = false;
    render(
      <ErrorBoundary>
        <div data-testid="child">hello</div>
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('child')).toBeTruthy();
    expect(screen.queryByTestId('error-boundary-fallback')).toBeNull();
  });

  it('renders the fallback with a "Try again" button when a child throws', () => {
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeTruthy();
    expect(screen.getByTestId('error-boundary-retry')).toBeTruthy();
    expect(screen.getByTestId('error-boundary-fallback').textContent).toContain(
      'Something went wrong',
    );
    // The error was surfaced to the console.
    expect(errSpy).toHaveBeenCalled();
  });

  it('recovers when "Try again" is clicked after the child stops throwing', () => {
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeTruthy();

    // Child no longer throws; retry should re-mount and render it.
    shouldThrow = false;
    fireEvent.click(screen.getByTestId('error-boundary-retry'));

    expect(screen.getByTestId('bomb-ok')).toBeTruthy();
    expect(screen.queryByTestId('error-boundary-fallback')).toBeNull();
  });

  it('auto-resets the error when resetKey changes (navigation)', () => {
    const { rerender } = render(
      <ErrorBoundary resetKey="a">
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeTruthy();

    // Navigate (resetKey changes) while the child stops throwing.
    shouldThrow = false;
    rerender(
      <ErrorBoundary resetKey="b">
        <Bomb />
      </ErrorBoundary>,
    );

    expect(screen.getByTestId('bomb-ok')).toBeTruthy();
    expect(screen.queryByTestId('error-boundary-fallback')).toBeNull();
  });
});

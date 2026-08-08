// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { translate } from '../i18n/useT';
import { readInitialLocale } from '../i18n/LocaleContext';

/**
 * ErrorBoundary — scoped render-error guard.
 *
 * Wraps a subtree (the project view) so a render error degrades gracefully
 * instead of unmounting the whole app. Phase-1B verification hit exactly this:
 * a `sessions.filter is not a function` throw blanked the entire app because
 * nothing caught it. Scoped here, a crash in one project view leaves the app
 * shell (project list, navigation) intact so the user can recover.
 *
 * - `resetKey`: when it changes (e.g. navigating to a different project/tab),
 *   the boundary auto-clears a prior error so stale crashes don't stick to new
 *   content.
 * - "Try again" manually re-mounts the children.
 */
interface ErrorBoundaryProps {
  children: ReactNode;
  /** When this value changes, a prior error state is cleared automatically. */
  resetKey?: unknown;
  /** Optional override for the fallback heading. */
  message?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface the failure so it's observable (no telemetry service per spec).
    console.error('[ErrorBoundary] render error:', error, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false });
    }
  }

  private handleRetry = (): void => {
    this.setState({ hasError: false });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Class component — no useT(); read the persisted locale directly.
      const locale = readInitialLocale();
      return (
        <div
          data-testid="error-boundary-fallback"
          className="flex flex-col items-center justify-center gap-3 h-full p-8 text-center"
        >
          <p className="text-sm text-secondary">
            {this.props.message ?? translate(locale, 'errorBoundary.message')}
          </p>
          <button
            type="button"
            data-testid="error-boundary-retry"
            onClick={this.handleRetry}
            className="rounded-sm bg-accent text-on-accent text-sm font-medium px-4 py-1.5 hover:bg-accent/85 cursor-pointer"
          >
            {translate(locale, 'errorBoundary.retry')}
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

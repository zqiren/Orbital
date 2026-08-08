// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// Mock the api helper + ApiError (class declared inside the hoisted factory).
let apiFn = vi.fn();
vi.mock('../config', () => {
  class ApiError extends Error {
    constructor(public status: number, public detail: string) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return {
    api: (...args: unknown[]) => apiFn(...args),
    ApiError,
    isRelayMode: false,
    BASE_URL: 'http://localhost:8000',
  };
});

import BrowserSignInCard from './BrowserSignInCard';
import { ApiError as MockApiError } from '../config';

beforeEach(() => {
  apiFn = vi.fn();
});

describe('BrowserSignInCard', () => {
  it('renders the sign-in copy and an Open Browser button', () => {
    render(<BrowserSignInCard />);
    expect(screen.getByText('Browser Sign-In')).toBeInTheDocument();
    expect(screen.getByText(/stay signed in/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Browser' })).toBeInTheDocument();
  });

  it('POSTs warmup on click, polls status, returns to idle when browser closes', async () => {
    vi.useFakeTimers();
    apiFn.mockImplementation(async (path: string) => {
      if (path === '/api/v2/platform/browser/warmup') return { status: 'launched' };
      if (path === '/api/v2/platform/browser/warmup/status') return { active: false };
      throw new Error(`unexpected path: ${path}`);
    });

    render(<BrowserSignInCard />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Browser' }));

    // Let the POST resolve — button switches to the "opening" state
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(apiFn).toHaveBeenCalledWith('/api/v2/platform/browser/warmup', {
      method: 'POST',
      body: JSON.stringify({ url: 'https://accounts.google.com' }),
    });
    expect(screen.queryByRole('button', { name: 'Open Browser' })).not.toBeInTheDocument();

    // First poll tick reports the browser closed — card returns to idle
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(apiFn).toHaveBeenCalledWith('/api/v2/platform/browser/warmup/status');
    expect(screen.getByRole('button', { name: 'Open Browser' })).toBeInTheDocument();
  });

  it('keeps polling while the browser stays open', async () => {
    vi.useFakeTimers();
    let active = true;
    apiFn.mockImplementation(async (path: string) => {
      if (path === '/api/v2/platform/browser/warmup') return { status: 'launched' };
      return { active };
    });

    render(<BrowserSignInCard />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Browser' }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Browser still open after two polls — stays in the opening state
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(screen.queryByRole('button', { name: 'Open Browser' })).not.toBeInTheDocument();

    // Browser closes — next poll restores the idle button
    active = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(screen.getByRole('button', { name: 'Open Browser' })).toBeInTheDocument();
  });

  it('shows the 409 guard detail inline and stays usable', async () => {
    const detail = 'A session is currently running. Stop running sessions, then try again.';
    apiFn.mockRejectedValueOnce(new MockApiError(409, detail));

    render(<BrowserSignInCard />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Browser' }));

    await waitFor(() => {
      expect(screen.getByText(detail)).toBeInTheDocument();
    });
    // Button is back so the user can retry after stopping sessions
    expect(screen.getByRole('button', { name: 'Open Browser' })).toBeInTheDocument();
    // No status polling was started after the failed launch
    expect(apiFn).toHaveBeenCalledTimes(1);
  });
});

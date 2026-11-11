// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AgentErrorNotice from './AgentErrorNotice';

describe('AgentErrorNotice', () => {
  it('renders the classified headline for missing_api_key plus the raw detail', () => {
    render(
      <AgentErrorNotice
        code="missing_api_key"
        message="No LLM API key configured"
        onOpenSettings={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    const notice = screen.getByTestId('agent-error-notice');
    expect(notice.textContent).toMatch(/API key/i);
    // Raw backend detail is shown untranslated as secondary text.
    expect(screen.getByText('No LLM API key configured')).toBeTruthy();
  });

  it('shows the provider sentence, not the raw SDK payload (1111 pass 2, D24)', () => {
    const { unmount } = render(
      <AgentErrorNotice
        code="provider_error"
        message="Error code: 400 - {'error': {'type': 'server_error', 'message': 'Upstream request failed: Model is unavailable.'}}"
        onOpenSettings={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByTestId('agent-error-notice').textContent).toBe(
      'The agent stopped due to a provider error.Upstream request failed: Model is unavailable.Open SettingsDismiss',
    );
    unmount();
    render(
      <AgentErrorNotice
        code="invalid_api_key"
        message="Error code: 401 - {'error': {...}}"
        onOpenSettings={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    const text = screen.getByTestId('agent-error-notice').textContent ?? '';
    expect(text).not.toMatch(/Error code|\{/);
    expect(text).toMatch(/^The provider rejected the API key\./);
  });

  it('falls back to the generic provider-error headline for unknown codes', () => {
    render(
      <AgentErrorNotice
        code="something_new"
        message="boom"
        onOpenSettings={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByTestId('agent-error-notice').textContent).toMatch(/provider error/i);
  });

  it('fires onOpenSettings and onDismiss', () => {
    const onOpenSettings = vi.fn();
    const onDismiss = vi.fn();
    render(
      <AgentErrorNotice
        code="missing_api_key"
        message="x"
        onOpenSettings={onOpenSettings}
        onDismiss={onDismiss}
      />,
    );
    fireEvent.click(screen.getByTestId('agent-error-notice-settings'));
    expect(onOpenSettings).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByTestId('agent-error-notice-dismiss'));
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  describe('internal_error (spec 141)', () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    const detail = "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb4 in position 1649: invalid start byte";

    it('renders the internal copy with no Open settings button', () => {
      render(
        <AgentErrorNotice
          code="internal_error"
          message={detail}
          onOpenSettings={vi.fn()}
          onDismiss={vi.fn()}
        />,
      );
      const notice = screen.getByTestId('agent-error-notice');
      expect(notice.textContent).toMatch(/Something went wrong inside Orbital/);
      expect(notice.textContent).not.toMatch(/provider error/i);
      expect(screen.getByText(detail)).toBeTruthy();
      expect(screen.queryByTestId('agent-error-notice-settings')).toBeNull();
      expect(screen.getByTestId('agent-error-notice-dismiss')).toBeTruthy();
    });

    it('Send feedback opens the dialog with the agent_error context prefilled', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        json: () => Promise.resolve({ text: 'bundle', bytes: 6, telemetry_enabled: true }),
      });
      vi.stubGlobal('fetch', fetchMock);
      render(
        <AgentErrorNotice
          code="internal_error"
          message={detail}
          onOpenSettings={vi.fn()}
          onDismiss={vi.fn()}
        />,
      );
      expect(screen.queryByTestId('feedback-dialog')).toBeNull();
      fireEvent.click(screen.getByTestId('agent-error-notice-feedback'));
      expect(screen.getByTestId('feedback-dialog')).toBeTruthy();
      expect(screen.getByTestId('feedback-context').textContent).toMatch(/internal error/i);
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      const url = new URL(String(fetchMock.mock.calls[0][0]), 'http://x');
      expect(url.pathname).toBe('/api/v2/feedback/preview');
      expect(url.searchParams.get('kind')).toBe('agent_error');
      expect(url.searchParams.get('detail')).toBe(detail);
    });

    it('provider codes keep Open settings and show no Send feedback', () => {
      render(
        <AgentErrorNotice
          code="provider_error"
          message="x"
          onOpenSettings={vi.fn()}
          onDismiss={vi.fn()}
        />,
      );
      expect(screen.getByTestId('agent-error-notice-settings')).toBeTruthy();
      expect(screen.queryByTestId('agent-error-notice-feedback')).toBeNull();
    });
  });
});

// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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
});

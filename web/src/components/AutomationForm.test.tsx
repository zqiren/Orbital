// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// The name field is optional: the prompt defines the automation, the name only
// labels it. These cover what gets submitted when the author leaves it blank.
// The rest of the form's behaviour (cron assembly, the type switch, the server's
// 400 surfacing) lives in AutomationsList.test.tsx, which drives it through the
// pane it is rendered in.

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import AutomationForm, { deriveName } from './AutomationForm';

afterEach(() => cleanup());

describe('deriveName', () => {
  it('takes the first line of the prompt', () => {
    expect(deriveName('Check the build\nthen post to Slack')).toBe('Check the build');
  });

  it('collapses runs of whitespace', () => {
    expect(deriveName('  Check   the    build  ')).toBe('Check the build');
  });

  it('truncates a long first line with an ellipsis', () => {
    const long = 'a'.repeat(80);
    const out = deriveName(long);
    expect(out).toHaveLength(49); // 48 + the ellipsis
    expect(out.endsWith('…')).toBe(true);
  });

  it('does not leave a dangling space before the ellipsis', () => {
    // 48th character lands mid-gap, so a naive slice would read "… …".
    const out = deriveName(`${'a'.repeat(47)} tail`);
    expect(out).toBe(`${'a'.repeat(47)}…`);
  });

  it('returns empty for an empty prompt', () => {
    expect(deriveName('   ')).toBe('');
  });
});

describe('AutomationForm — optional name', () => {
  function renderForm() {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<AutomationForm onSubmit={onSubmit} onCancel={vi.fn()} />);
    return onSubmit;
  }

  it('submits a name derived from the prompt when left blank', async () => {
    const onSubmit = renderForm();
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Summarise yesterday’s support tickets' },
    });
    fireEvent.click(screen.getByTestId('automation-form-save'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0].name).toBe('Summarise yesterday’s support tickets');
  });

  it('keeps a name the author typed', async () => {
    const onSubmit = renderForm();
    fireEvent.change(screen.getByTestId('automation-form-name'), {
      target: { value: 'Ticket triage' },
    });
    fireEvent.change(screen.getByTestId('automation-form-prompt'), {
      target: { value: 'Summarise yesterday’s support tickets' },
    });
    fireEvent.click(screen.getByTestId('automation-form-save'));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0].name).toBe('Ticket triage');
  });

  it('still refuses to submit without a prompt — there is nothing to derive from', () => {
    const onSubmit = renderForm();
    fireEvent.click(screen.getByTestId('automation-form-save'));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByTestId('automation-form-error')).toBeInTheDocument();
  });
});

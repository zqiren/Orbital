// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CopyButton from './CopyButton';

afterEach(() => cleanup());

/**
 * jsdom ships no clipboard, and `userEvent.setup()` installs a stub of its own
 * — so every call here must land AFTER setup() or it gets overwritten.
 */
function setClipboard(writeText: ((t: string) => Promise<void>) | null) {
  if (writeText === null) {
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined, configurable: true, writable: true,
    });
    return;
  }
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText }, configurable: true, writable: true,
  });
}

describe('CopyButton', () => {
  beforeEach(() => setClipboard(async () => undefined));

  it('writes the exact text to the clipboard and flips to Copied', async () => {
    const user = userEvent.setup();
    const writeText = vi.fn(async () => undefined);
    setClipboard(writeText); // after setup() — see the note on setClipboard

    render(<CopyButton text={'line one\nline two'} />);
    await user.click(screen.getByTestId('copy-button'));

    expect(writeText).toHaveBeenCalledWith('line one\nline two');
    await waitFor(() =>
      expect(screen.getByTestId('copy-button')).toHaveAttribute('data-copied', 'true'),
    );
  });

  it('renders NOTHING when the clipboard API is unavailable', () => {
    // The LAN dev surface (http://<LAN-IP>:5173) is not a secure context, so
    // navigator.clipboard is undefined there. A button that silently no-ops
    // reads as a broken feature — so there must be no button at all.
    setClipboard(null);
    const { container } = render(<CopyButton text="anything" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('a rejected write does not throw and leaves the button un-copied', async () => {
    const user = userEvent.setup();
    setClipboard(vi.fn(async () => {
      throw new Error('NotAllowedError');
    }));

    render(<CopyButton text="x" />);
    await user.click(screen.getByTestId('copy-button'));

    expect(screen.getByTestId('copy-button')).not.toHaveAttribute('data-copied');
  });

  it('is type="button" so it cannot submit an enclosing settings form', () => {
    render(<CopyButton text="x" />);
    expect(screen.getByTestId('copy-button')).toHaveAttribute('type', 'button');
  });
});

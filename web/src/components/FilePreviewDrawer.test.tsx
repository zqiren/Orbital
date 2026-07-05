// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import FilePreviewDrawer from './FilePreviewDrawer';

function renderDrawer(open: boolean, onClose = vi.fn()) {
  return render(
    <FilePreviewDrawer
      open={open}
      selectedPath={null}
      fileContent={null}
      loading={false}
      onClose={onClose}
    />,
  );
}

describe('FilePreviewDrawer — focus management (a11y)', () => {
  it('is inert and keeps its controls out of the tab order when closed', () => {
    const { container } = renderDrawer(false);
    const panel = container.querySelector('[role="dialog"]');
    expect(panel).not.toBeNull();
    expect(panel).toHaveAttribute('inert');
    // The close button is not in the accessibility tree / tab order while closed.
    expect(screen.queryByRole('button', { name: 'Close preview' })).toBeNull();
  });

  it('moves focus into the drawer (close button) when it opens', () => {
    renderDrawer(true);
    const closeBtn = screen.getByRole('button', { name: 'Close preview' });
    expect(document.activeElement).toBe(closeBtn);
    expect(screen.getByRole('dialog')).not.toHaveAttribute('inert');
  });

  it('restores focus to the previously-focused element when it closes', () => {
    function Wrapper({ open }: { open: boolean }) {
      return (
        <>
          <button data-testid="opener">opener</button>
          <FilePreviewDrawer
            open={open}
            selectedPath={null}
            fileContent={null}
            loading={false}
            onClose={() => {}}
          />
        </>
      );
    }
    const { rerender } = render(<Wrapper open={false} />);
    const opener = screen.getByTestId('opener');
    opener.focus();
    expect(document.activeElement).toBe(opener);

    rerender(<Wrapper open={true} />);
    expect(document.activeElement).toBe(
      screen.getByRole('button', { name: 'Close preview' }),
    );

    rerender(<Wrapper open={false} />);
    expect(document.activeElement).toBe(opener);
  });

  it('traps Tab / Shift+Tab within the open drawer', () => {
    renderDrawer(true);
    const closeBtn = screen.getByRole('button', { name: 'Close preview' });
    expect(document.activeElement).toBe(closeBtn);

    // Only the close button is focusable here, so Tab wraps back to it.
    fireEvent.keyDown(closeBtn, { key: 'Tab' });
    expect(document.activeElement).toBe(closeBtn);
    fireEvent.keyDown(closeBtn, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(closeBtn);
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    renderDrawer(true, onClose);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

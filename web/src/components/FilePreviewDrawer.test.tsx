// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import FilePreviewDrawer from './FilePreviewDrawer';

const FILE_PREVIEW_DRAWER_WIDTH_KEY = 'orbital:filePreviewDrawerWidth';

afterEach(() => {
  cleanup();
  localStorage.clear();
});

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
    const resizeHandle = screen.getByRole('separator', { name: 'Resize file preview' });
    expect(document.activeElement).toBe(closeBtn);

    // The close button is last, so Tab wraps to the resize handle; Shift+Tab
    // from that first control wraps back to Close.
    fireEvent.keyDown(closeBtn, { key: 'Tab' });
    expect(document.activeElement).toBe(resizeHandle);
    fireEvent.keyDown(resizeHandle, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(closeBtn);
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    renderDrawer(true, onClose);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('FilePreviewDrawer — desktop resizing', () => {
  it('expands when the left edge is dragged left and persists the width', () => {
    renderDrawer(true);
    const panel = screen.getByRole('dialog');
    const handle = screen.getByRole('separator', { name: 'Resize file preview' });

    expect(panel).toHaveStyle({ '--file-preview-drawer-width': '420px' });

    fireEvent.pointerDown(handle, { button: 0, clientX: 500 });
    fireEvent.pointerMove(window, { clientX: 300 });
    fireEvent.pointerUp(window);

    expect(panel).toHaveStyle({ '--file-preview-drawer-width': '620px' });
    expect(localStorage.getItem(FILE_PREVIEW_DRAWER_WIDTH_KEY)).toBe('620');
  });

  it('clamps pointer resizing to 320px–80% of the available content area', () => {
    renderDrawer(true);
    const panel = screen.getByRole('dialog');
    const handle = screen.getByRole('separator', { name: 'Resize file preview' });

    fireEvent.pointerDown(handle, { button: 0, clientX: 500 });
    fireEvent.pointerMove(window, { clientX: 2000 });
    expect(panel).toHaveStyle({ '--file-preview-drawer-width': '320px' });
    fireEvent.pointerMove(window, { clientX: -1000 });
    expect(panel).toHaveStyle({ '--file-preview-drawer-width': '819px' });
    fireEvent.pointerUp(window);
  });

  it('supports keyboard resizing and restores the saved width', () => {
    const first = renderDrawer(true);
    const handle = screen.getByRole('separator', { name: 'Resize file preview' });

    fireEvent.keyDown(handle, { key: 'ArrowLeft' });
    expect(screen.getByRole('dialog')).toHaveStyle({
      '--file-preview-drawer-width': '444px',
    });
    first.unmount();

    renderDrawer(true);
    expect(screen.getByRole('dialog')).toHaveStyle({
      '--file-preview-drawer-width': '444px',
    });
  });
});

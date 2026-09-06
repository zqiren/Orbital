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

// ---------------------------------------------------------------------------
// Spec 078 D13 — docked mode (ChatTab's third column)
// ---------------------------------------------------------------------------

function renderDocked(
  overrides: Partial<React.ComponentProps<typeof FilePreviewDrawer>> = {},
) {
  return render(
    <FilePreviewDrawer
      docked
      open
      selectedPath={null}
      fileContent={null}
      loading={false}
      onClose={vi.fn()}
      {...overrides}
    />,
  );
}

describe('FilePreviewDrawer — docked mode', () => {
  it('renders in flow: no dialog, no modal semantics, no scrim', () => {
    const { container } = renderDocked();
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByTestId('workspace-panel-column')).toBeInTheDocument();
    expect(container.querySelector('.bg-black\\/30')).toBeNull();
    const column = screen.getByTestId('workspace-panel-column');
    expect(column).not.toHaveAttribute('aria-modal');
    expect(column).not.toHaveAttribute('inert');
    expect(column.className).not.toContain('absolute');
    expect(column.className).not.toContain('translate-x');
  });

  it('defaults to 360px wide instead of the overlay drawer’s 420px', () => {
    renderDocked();
    expect(screen.getByTestId('workspace-panel-column')).toHaveStyle({
      '--file-preview-drawer-width': '360px',
    });
  });

  it('remembers its own width, separate from the overlay drawer', () => {
    // The overlay drawer's width must not leak into the docked panel …
    localStorage.setItem(FILE_PREVIEW_DRAWER_WIDTH_KEY, '520');
    const first = renderDocked();
    expect(screen.getByTestId('workspace-panel-column')).not.toHaveStyle({
      '--file-preview-drawer-width': '520px',
    });
    first.unmount();
    // … while the panel's own remembered width is honoured.
    localStorage.setItem('orbital:workspacePanelWidth', '520');
    renderDocked();
    expect(screen.getByTestId('workspace-panel-column')).toHaveStyle({
      '--file-preview-drawer-width': '520px',
    });
  });

  it('defaults to half the window, capped so chat keeps its minimum', () => {
    // jsdom's window is 1024 wide: half is 512 but 1024 − 800 reserved = 224,
    // so the 360 floor wins.
    renderDocked();
    expect(screen.getByTestId('workspace-panel-column')).toHaveStyle({
      '--file-preview-drawer-width': '360px',
    });
  });

  it('keeps the resize handle working and persists the new width', () => {
    renderDocked();
    const handle = screen.getByRole('separator', { name: 'Resize file preview' });
    fireEvent.pointerDown(handle, { button: 0, clientX: 500 });
    fireEvent.pointerMove(window, { clientX: 400 });
    fireEvent.pointerUp(window);

    expect(screen.getByTestId('workspace-panel-column')).toHaveStyle({
      '--file-preview-drawer-width': '460px',
    });
    expect(localStorage.getItem('orbital:workspacePanelWidth')).toBe('460');
    expect(localStorage.getItem(FILE_PREVIEW_DRAWER_WIDTH_KEY)).toBeNull();
  });

  it('renders `children` instead of FilePreview, under the header slot', () => {
    renderDocked({
      header: <span data-testid="panel-header">switch</span>,
      children: <div data-testid="panel-body">the panel</div>,
    });
    expect(screen.getByTestId('panel-header')).toBeInTheDocument();
    expect(screen.getByTestId('panel-body')).toBeInTheDocument();
    // FilePreview's empty state must not be rendered underneath.
    expect(screen.queryByText('Select a file to preview')).toBeNull();
  });

  it('collapses via a "Hide workspace" button rather than "Close preview"', () => {
    const onClose = vi.fn();
    renderDocked({ onClose });
    expect(screen.queryByRole('button', { name: 'Close preview' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Hide workspace' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does not trap focus, steal focus, or close on Escape', () => {
    const onClose = vi.fn();
    render(
      <>
        <button data-testid="composer">composer</button>
        <FilePreviewDrawer
          docked
          open
          selectedPath={null}
          fileContent={null}
          loading={false}
          onClose={onClose}
        >
          <div>panel</div>
        </FilePreviewDrawer>
      </>,
    );
    const composer = screen.getByTestId('composer');
    composer.focus();
    expect(document.activeElement).toBe(composer);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });
});

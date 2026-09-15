// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 078 §5 / D7, reshaped by spec 088 — the panel bar (exactly a
 * Files|Browser switch and a Browser-only Annotate toggle) is its own
 * component so it can sit in the drawer's header row; the body shows the one
 * selected view and paints no bar of its own.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import WorkspacePanel, { WorkspacePanelBar } from './WorkspacePanel';

afterEach(() => cleanup());

function renderBar(overrides: Partial<React.ComponentProps<typeof WorkspacePanelBar>> = {}) {
  const props = {
    view: 'files' as const,
    onViewChange: vi.fn(),
    annotating: false,
    onToggleAnnotate: vi.fn(),
    ...overrides,
  };
  const utils = render(<WorkspacePanelBar {...props} />);
  return { ...utils, props };
}

describe('WorkspacePanelBar — the view switch', () => {
  it('renders exactly two tabs in one tablist', () => {
    renderBar();
    expect(screen.getAllByRole('tablist')).toHaveLength(1);
    expect(screen.getAllByRole('tab')).toHaveLength(2);
    expect(screen.getByRole('tab', { name: 'Files' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Browser' })).toBeInTheDocument();
  });

  it('marks the selected view with aria-selected', () => {
    renderBar({ view: 'browser' });
    expect(screen.getByRole('tab', { name: 'Browser' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'false');
  });

  it('reports the picked view to the parent', () => {
    const { props } = renderBar({ view: 'files' });
    fireEvent.click(screen.getByRole('tab', { name: 'Browser' }));
    expect(props.onViewChange).toHaveBeenCalledWith('browser');
  });
});

describe('WorkspacePanelBar — the Annotate toggle', () => {
  it('reads "Annotate" when off and "Done" when on, with aria-pressed following', () => {
    const { rerender, props } = renderBar({ view: 'browser' });
    expect(screen.getByRole('button', { name: 'Annotate' })).toHaveAttribute('aria-pressed', 'false');

    rerender(<WorkspacePanelBar {...props} annotating />);
    expect(screen.getByRole('button', { name: 'Done' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('calls onToggleAnnotate when pressed', () => {
    const { props } = renderBar({ view: 'browser' });
    fireEvent.click(screen.getByTestId('panel-annotate'));
    expect(props.onToggleAnnotate).toHaveBeenCalledTimes(1);
  });

  it('shows no Annotate button on the Files view (selection + Quote covers files)', () => {
    renderBar({ view: 'files' });
    expect(screen.queryByRole('button', { name: 'Annotate' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Done' })).toBeNull();
  });
});

describe('WorkspacePanel — the body', () => {
  it('shows only the selected view, never both', () => {
    const { rerender } = render(
      <WorkspacePanel
        view="files"
        browser={<div data-testid="browser-body" />}
        files={<div data-testid="files-body" />}
      />,
    );
    expect(screen.getByTestId('files-body')).toBeInTheDocument();
    expect(screen.queryByTestId('browser-body')).toBeNull();

    rerender(
      <WorkspacePanel
        view="browser"
        browser={<div data-testid="browser-body" />}
        files={<div data-testid="files-body" />}
      />,
    );
    expect(screen.getByTestId('browser-body')).toBeInTheDocument();
    expect(screen.queryByTestId('files-body')).toBeNull();
  });

  it('paints no bar of its own — the switch lives in the drawer header (spec 088)', () => {
    render(<WorkspacePanel view="browser" browser={<div />} files={<div />} />);
    expect(screen.queryByRole('tablist')).toBeNull();
    expect(screen.queryByRole('button')).toBeNull();
  });
});
